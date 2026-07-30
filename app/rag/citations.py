"""Turn a model answer into precise, verbatim citations.

Two jobs, split so that neither the model nor the code does something it is bad at:

1. **Which chunks were used** — the model tells us. Contexts are numbered in the prompt and
   the answer carries `(1)`, `(2)` markers, so only the evidence it actually leaned on ends
   up in `sources` instead of every retrieved chunk.
2. **Which span is the evidence** — the *code* decides, by scoring the sentences of a cited
   chunk against the question with the same cross-encoder used for reranking, then slicing
   that sentence straight out of the chunk text.

Because the quote is sliced from indexed text rather than generated, it is verbatim by
construction: there is no paraphrase to detect and no hallucinated quote to verify. Chunks
never span pages, so the sentence carries its chunk's page number and the pair is correct.
"""

from __future__ import annotations

import re

from . import rerank
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
        kept = [
            str(mapping[int(n)]) for n in _NUMBER.findall(match.group(1)) if int(n) in mapping
        ]
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


def evidence_quote(question: str, context: Context) -> str:
    """The span of `context` that best supports an answer to `question`.

    Always an exact substring of the chunk — and therefore of the page, since chunks are
    built from page text and never span pages. We slice, never generate. Falls back to the
    chunk's opening span if the cross-encoder is unavailable.
    """
    text = context.text
    bounds = _merge_short(text, _spans(text))
    if not bounds:
        return text.strip()
    if len(bounds) == 1:
        start, end = bounds[0]
        return text[start : min(end, start + _MAX_QUOTE)].strip()

    try:
        scores = rerank.score(question, [text[s:e] for s, e in bounds])
        best = max(range(len(bounds)), key=lambda i: scores[i])
    except rerank.RerankError:
        best = 0

    start, end = bounds[best]
    return text[start : min(end, start + _MAX_QUOTE)].strip()
