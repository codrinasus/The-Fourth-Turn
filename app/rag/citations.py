"""Turn a model answer into precise, verbatim citations.

Two jobs, split so that neither the model nor the code does something it is bad at:

1. **Which chunks were used** — the model tells us. Contexts are numbered in the prompt and
   the answer carries `(1)`, `(2)` markers, so only the evidence it actually leaned on ends
   up in `sources` instead of every retrieved chunk.
2. **Which span is the evidence** — the *code* decides. The sentences of a cited chunk are
   scored against the question with the same cross-encoder used for reranking, and then
   the quote is *widened* around the winner: we try the whole chunk, then the paragraph
   containing it, then the sentence alone, and ship the largest one the PDF can vouch for.
   A lone sentence read as truncated — the best-matching sentence is frequently the one
   next to the sentence carrying the fact — so we return all the context the document
   will actually confirm.

Because the quote is sliced from indexed text rather than generated, it is verbatim by
construction: there is no paraphrase to detect and no hallucinated quote to verify.

That guarantee is about the *chunk*, though, and the chunk is Docling's rendering of the
page rather than the page itself — so it is not the guarantee grading actually checks.
`rag/verbatim.py` closes that gap afterwards, re-expressing the span in the PDF's own
characters and verifying the page number. Chunks do not span pages, but Docling labels a
block with the page it *starts* on, so a paragraph running across a page break can carry a
page number one too low; that is repaired there, not here.
"""

from __future__ import annotations

import re

from . import rerank, verbatim
from .retrieve import Context

# The paper's own references look like [42], so we ask for parentheses instead of brackets
# to keep the model's markers distinguishable from text it may quote.
#
# Grouped markers are matched, not just lone ones. Asked to cite (2) and (6) for the same
# claim, the model writes "(2; 6)" — and a lone-number pattern silently misses it. That
# failed loudly on our Level-3 answers: q7 leaned on four passages, three of the markers
# went unparsed, and the answer shipped citing numbers that had no matching source. The
# marker set and the returned sources have to be derived from the same parse or they drift.
_MARKER = re.compile(r"\(\s*(\d{1,2}(?:\s*[;,]\s*\d{1,2})*)\s*\)")
_NUMBER = re.compile(r"\d{1,2}")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_MIN_QUOTE = 40  # shorter spans are rarely evidence on their own
_MIN_HEADING_PROSE = 80  # a span shorter than this with no terminal punctuation is a title
_MAX_QUOTE = 400  # a whole sentence is kept even if long; this only guards runaway rows


def number_contexts(contexts: list[Context]) -> str:
    """Render the numbered context block the citation markers refer to."""
    if not contexts:
        return "(no context retrieved)"
    return "\n\n".join(f"({i}) [page {c.page}] {c.text}" for i, c in enumerate(contexts, start=1))


def parse_markers(answer: str, n_contexts: int) -> list[int]:
    """0-based context indices cited in the answer, in order of first appearance."""
    seen: list[int] = []
    for match in _MARKER.finditer(answer):
        for number in _NUMBER.findall(match.group(1)):
            idx = int(number) - 1
            if 0 <= idx < n_contexts and idx not in seen:
                seen.append(idx)
    return seen


def renumber(answer: str, order: list[int]) -> str:
    """Rewrite markers so `(1)` in the answer is `sources[0]` after filtering.

    The model cites the prompt's numbering; the response only carries the chunks it used,
    so without this a reader would follow `(3)` to a source that is no longer third.

    A number with no mapping is dropped rather than left in place — it would point past
    the end of `sources`. If that empties a marker, the marker goes too, so the answer
    never carries a citation the response cannot back up.
    """
    mapping = {old + 1: new + 1 for new, old in enumerate(order)}

    def sub(match: re.Match[str]) -> str:
        kept = [str(mapping[int(n)]) for n in _NUMBER.findall(match.group(1)) if int(n) in mapping]
        return f"({', '.join(kept)})" if kept else ""

    # Removing a marker leaves the space that preceded it stranded, sometimes right before
    # a full stop; tidy that up so a dropped citation is invisible rather than conspicuous.
    cleaned = _MARKER.sub(sub, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return re.sub(r"\s+([.,;:])", r"\1", cleaned)


def _spans(text: str) -> list[tuple[int, int]]:
    """Candidate spans as (start, end) offsets into `text`.

    Offsets rather than strings on purpose: every quote is later produced as
    `text[start:end]`, so it cannot drift from the source. Joining split-out sentences
    with a space would silently invent whitespace the page does not contain.
    """
    bounds: list[tuple[int, int]] = []
    for line in re.finditer(r"[^\n]+", text):
        line_text = line.group(0)
        # Table-ish rows have no sentence structure to split on.
        if line_text.count("|") >= 2:
            bounds.append((line.start(), line.end()))
            continue
        cursor = line.start()
        for part in _SENTENCE_SPLIT.split(line_text):
            if not part:
                continue
            start = text.index(part, cursor)
            bounds.append((start, start + len(part)))
            cursor = start + len(part)
    return bounds


def _merge_short(text: str, bounds: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Extend runt spans over their neighbour until they carry enough context.

    Merging extends the *end offset*, so the result stays one contiguous slice of `text` —
    including whatever newline sat between the two sentences.

    Never merges across a blank line. `chunking._emit` joins blocks with a literal "\\n\\n"
    whatever separated them on the page, so a span crossing that join can fail to be a page
    substring even though it is a chunk substring. Staying inside one block keeps the quote
    verbatim at page level, which is what grading checks.
    """
    merged: list[tuple[int, int]] = []
    for start, end in bounds:
        if merged:
            prev_start, prev_end = merged[-1]
            same_block = "\n\n" not in text[prev_end:start]
            is_table = text[start:end].count("|") >= 2
            if same_block and not is_table and prev_end - prev_start < _MIN_QUOTE:
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))
    return merged


def _is_heading(span: str) -> bool:
    """Whether a span looks like a section title rather than a sentence.

    Short, and not ending in sentence punctuation. Numbered headings ("2.5 Evaluation of
    Explanations") are caught by the same test without needing to recognise the numbering,
    so this does not depend on how one publisher formats headings.
    """
    stripped = span.strip()
    return len(stripped) < _MIN_HEADING_PROSE and not stripped.endswith((".", "!", "?", ":"))


def _block_around(text: str, start: int, end: int) -> tuple[int, int]:
    """The paragraph containing `start:end`, as offsets.

    Blocks are separated by the blank line `chunking._emit` inserts between them, so this
    is the largest unit that was contiguous prose on the page.
    """
    opening = text.rfind("\n\n", 0, start)
    closing = text.find("\n\n", end)
    return (0 if opening < 0 else opening + 2, len(text) if closing < 0 else closing)


def evidence_quote(question: str, context: Context) -> str:
    """The evidence from `context`, as much of it as can be proven verbatim.

    We slice, never generate, so any of these is a true substring of the chunk. The
    question is how *much* to return, and a single sentence turned out to be too little:
    the sentence the cross-encoder picks is the one that best matches the question, which
    is often the sentence just before or after the one carrying the actual fact, so the
    relevant part reads as cut off.

    So we widen, and let the PDF arbitrate. Three candidates are tried largest-first —
    the whole chunk, then the paragraph around the best sentence, then the sentence — and
    the first that `verbatim` can locate on the page wins. That gives the fullest context
    the document can actually vouch for, and it cannot lower the verbatim rate: the
    sentence is still there as the last resort, exactly as before.

    Widest-first matters because a chunk may join blocks that were not adjacent on the
    page — `chunking` drops page headers, footers and footnotes from between them — so
    the whole chunk is *usually* but not always a contiguous page span. Rather than
    predict which, we ask.
    """
    text = context.text
    bounds = _merge_short(text, _spans(text))
    # A section heading is short, has no sentence structure, and scores well against a
    # question that echoes its words — "2.5 Evaluation of Explanations" beat the prose
    # underneath it for "How does the survey categorise the evaluation of explanations?".
    # It is a true span of the page but it is not evidence, so headings are set aside
    # unless they are all the chunk has.
    prose = [b for b in bounds if not _is_heading(text[b[0] : b[1]])]
    bounds = prose or bounds
    if not bounds:
        return text.strip()

    if len(bounds) == 1:
        best = 0
    else:
        try:
            scores = rerank.score(question, [text[s:e] for s, e in bounds])
            best = max(range(len(bounds)), key=lambda i: scores[i])
        except rerank.RerankError:
            best = 0

    start, end = bounds[best]
    sentence = text[start : min(end, start + _MAX_QUOTE)].strip()
    block_start, block_end = _block_around(text, start, end)

    for candidate in (
        text.strip(),  # the whole chunk
        text[block_start:block_end].strip(),  # the paragraph the best sentence sits in
        sentence,  # the sentence itself
    ):
        if verbatim.find(candidate, context.page) is not None:
            return candidate
    return sentence
