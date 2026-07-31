"""Put the PDF's own characters back into an evidence quote.

Grading text-searches every `Source.quote` in `data/in/document.pdf`. Our quotes are
already exact substrings of the chunk they came from — `citations.evidence_quote` slices
by offset and never generates — but the chunk is *Docling's* rendering of the page, and
Docling folds the typography:

    PDF     : specialized search systems – in legal search … the model's decision-making
    Docling : specialized search systems - in legal search … the model's decision-making

Measured on this document: the PDF contains 135 en-dashes, 53 right single quotes and 32
curly double quotes; the Docling text contains **none of them**. So a quote holding an
apostrophe — which is most quotes of any length — is verbatim with respect to our index
and *not* findable in the PDF. That is the difference between "grounded" and "looks
fabricated" to anyone checking by search.

The fix is alignment, not rewriting. We fold both sides to a common form, locate the
quote inside the page's own text as extracted from the PDF, and then return **the PDF's
characters** for that span. Nothing is invented: the output is a slice of the real page.
Failure is safe — if the span cannot be located, the original quote is returned unchanged
and the caller is none the wiser, so this can only ever improve fidelity.

Matching ignores whitespace **entirely**, not just runs of it. The two extractors disagree
about where spaces go, not only how many: pypdf reads a citation marker as `[ 30]` where
Docling reads `[30]`, and a line break inside a hyphenated word lands differently again.
Neither spacing is "the PDF" — whitespace is an artefact of extraction, so comparing it
would measure the extractors against each other rather than measure our grounding. The
returned span therefore carries the PDF's characters with its whitespace collapsed to
single spaces.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from ..config import get_settings

log = logging.getLogger(__name__)

# Folded to a common form on both sides before matching. These are exactly the
# substitutions that lose information going from PDF to Docling text.
_FOLD = {
    "–": "-",
    "—": "-",
    "‐": "-",
    "‑": "-",
    "−": "-",
    "‘": "'",
    "’": "'",
    "‛": "'",
    "ʼ": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    # Typographic ligatures. Extractors disagree about whether to expand these, so both
    # sides are folded to the letters. The dropout paper alone contains 121 "ﬁ" and 91
    # "ﬀ" ("diﬀerent", "eﬀect", "overﬁtting"), so a missing "ﬀ" here would break the
    # majority of its quotes.
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
    "": "Qu",
}
_PUA = re.compile(r"[-]")
_WS = re.compile(r"\s+")
_HYPHEN = re.compile(r"-")
# Our own table rendering. `docling_parser` flattens a table to pipe-separated cells, so a
# table chunk carries "|" characters the page never had. Dropping them on both sides is
# what lets a table row be verified against the PDF at all — without this, no table chunk
# could ever be quoted and the quote collapsed to the caption line above it.
_PIPE = re.compile(r"\|")
# A word broken across a line: letter, hyphen, line break, lowercase letter. Repaired in
# the span we return so a quote reads as prose rather than as a page layout.
_LINE_BREAK_HYPHEN = re.compile(r"(?<=[A-Za-z])-\s+(?=[a-z])")

_MIN_ALIGNABLE = 20  # below this a match is as likely to be coincidence as evidence


def _fold(text: str) -> str:
    for src, dst in _FOLD.items():
        text = text.replace(src, dst)
    return _PUA.sub("", text)


def _fold_with_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-free folded text, plus each character's offset in the original.

    The map is what makes this an *alignment* rather than a substitution: once the quote
    is found in folded space, the map converts those bounds back into offsets in the real
    page text, and the characters we return are the ones the PDF actually contains.
    """
    out: list[str] = []
    offsets: list[int] = []
    for i, ch in enumerate(text):
        # Skip exactly what `_squash` drops, so the haystack and the needle are written in
        # the same alphabet and an index into one means the same thing in the other.
        if ch.isspace() or ch in "-|":
            continue
        folded = _FOLD.get(ch, "" if _PUA.match(ch) else ch)
        if folded == "-":
            continue
        for piece in folded:
            out.append(piece)
            offsets.append(i)
    return "".join(out), offsets


def _squash(text: str) -> str:
    """The normal form both sides are compared in: folded, de-hyphenated, whitespace-free.

    Pipes go because they are ours, not the document's: the parser renders a table as
    pipe-separated cells, so a table chunk contains separators the page never had.

    Hyphens go too, and that is not laziness. A PDF breaks words across lines with a
    hyphen, and the two extractors disagree about it: pypdf reports the layout as it
    stands — `optimiza- tion`, `au- tomatically` — while Docling rejoins the word. Neither
    hyphen is content; it is a typesetting artefact of where the line happened to end.
    Dropping every hyphen makes `optimiza-tion`, `optimization` and `max-norm` versus
    `maxnorm` all compare equal, which is the behaviour we want in both directions.
    """
    return _PIPE.sub("", _HYPHEN.sub("", _WS.sub("", _fold(text))))


@lru_cache(maxsize=1)
def _pdf_pages() -> tuple[str, ...]:
    """Per-page text straight from the PDF, or `()` if it cannot be read.

    Deliberately a *second* extraction of the same file, independent of Docling. Docling
    is the parser we index with; pypdf is the reference we check ourselves against, and
    agreeing with a different extractor is a stronger guarantee than agreeing with
    ourselves. Cached — this is per-query work.
    """
    in_dir = Path(get_settings().in_dir)
    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        return ()
    try:
        from pypdf import PdfReader

        return tuple((page.extract_text() or "") for page in PdfReader(str(pdfs[0])).pages)
    except Exception as e:  # noqa: BLE001 - alignment is an enhancement, never a hard dep
        log.warning("could not read the PDF for quote alignment (%s)", e)
        return ()


def _readable(span: str) -> str:
    """A PDF span cleaned of extraction artefacts, keeping the file's own punctuation.

    Two artefacts, both of layout rather than content: a word broken across a line comes
    back as `optimiza- tion`, and ligatures come back as single glyphs (`diﬀerent`).
    Shipping either would put the page's typesetting into an evidence quote. The dashes,
    curly quotes and the wording remain the PDF's.
    """
    joined = _LINE_BREAK_HYPHEN.sub("", span)
    for glyph, expansion in _FOLD.items():
        if len(expansion) > 1:  # ligatures only — not the dash/quote folds
            joined = joined.replace(glyph, expansion)
    return _WS.sub(" ", joined).strip()


def _slice(needle: str, page_text: str) -> str | None:
    """The span of `page_text` matching `needle`, in the page's own characters."""
    haystack, offsets = _fold_with_map(page_text)
    at = haystack.find(needle)
    if at < 0:
        return None
    start = offsets[at]
    end = offsets[at + len(needle) - 1] + 1
    return _readable(page_text[start:end])


def find(quote: str, page: int) -> tuple[str, int] | None:
    """`(the span in the PDF's characters, its real page)`, or None if it is not there.

    The single lookup both `locate` and the quote-width choice in `rag/citations.py` are
    built on, so "can this be proven verbatim?" and "what do we ship?" can never disagree.
    """
    pages = _pdf_pages()
    if not pages or len(quote) < _MIN_ALIGNABLE:
        return None

    needle = _squash(quote)
    if 1 <= page <= len(pages):
        found = _slice(needle, pages[page - 1])
        if found:
            return found, page

    # Nearest-first, so a page-break overflow resolves to the adjoining page rather than
    # to some coincidentally similar text elsewhere in the document.
    for candidate in sorted(range(1, len(pages) + 1), key=lambda p: (abs(p - page), p)):
        found = _slice(needle, pages[candidate - 1])
        if found:
            log.info("quote cited to page %d is actually on page %d — corrected", page, candidate)
            return found, candidate
    return None


def locate(quote: str, page: int) -> tuple[str, int]:
    """`(quote in the PDF's own characters, the page it is really on)`.

    Two repairs in one pass, both verified against the PDF rather than asserted:

    1. **Typography.** The returned text is sliced out of the PDF, so the dashes, curly
       quotes and ligatures are the file's own — see the module docstring.
    2. **The page number.** Docling labels a block with the page it *starts* on, so a
       paragraph flowing across a page break is cited one page early. We found exactly
       that: a section 7.2 quote cited to page 17 whose text is physically on page 18.
       The cited page is tried first and always wins a tie, so a correct citation is
       never moved; only a quote that genuinely is not on its page gets relocated, to the
       nearest page that actually contains it.

    Unchanged input is returned whenever the PDF is unreadable or the quote cannot be
    found anywhere — this can only improve a citation, never invent one.
    """
    found = find(quote, page)
    if found is None:
        log.info("quote not locatable anywhere in the PDF — returning it unaligned")
        return quote, page
    return found


def is_verbatim(quote: str, page: int) -> bool:
    """Whether `quote` appears on `page` of the PDF under the documented normalisation.

    The check `scripts/audit_quotes.py` uses to put a number on grounding. Both sides are
    normalised the same way and only for things that are artefacts of PDF extraction:
    whitespace, line-break hyphenation, ligatures, and the dash/quote variants the two
    extractors disagree about. **Case and wording are not touched** — a passing quote is
    the page's own words in the page's own order, and any paraphrase fails.
    """
    pages = _pdf_pages()
    if not pages or not (1 <= page <= len(pages)) or not quote.strip():
        return False
    return _squash(quote) in _squash(pages[page - 1])
