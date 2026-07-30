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
    "–": "-", "—": "-", "‐": "-", "‑": "-", "−": "-",
    "‘": "'", "’": "'", "‛": "'", "ʼ": "'",
    "“": '"', "”": '"', "„": '"',
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "ﬁ": "fi", "ﬂ": "fl", "": "Qu",
}
_PUA = re.compile(r"[-]")
_WS = re.compile(r"\s+")

_MIN_ALIGNABLE = 20   # below this a match is as likely to be coincidence as evidence


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
        if ch.isspace():
            continue
        for piece in _FOLD.get(ch, "" if _PUA.match(ch) else ch):
            out.append(piece)
            offsets.append(i)
    return "".join(out), offsets


def _squash(text: str) -> str:
    """Folded and stripped of all whitespace — the form both sides are compared in."""
    return _WS.sub("", _fold(text))


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


def _slice(needle: str, page_text: str) -> str | None:
    """The span of `page_text` matching `needle`, in the page's own characters."""
    haystack, offsets = _fold_with_map(page_text)
    at = haystack.find(needle)
    if at < 0:
        return None
    start = offsets[at]
    end = offsets[at + len(needle) - 1] + 1
    return _WS.sub(" ", page_text[start:end]).strip()


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
    pages = _pdf_pages()
    if not pages or len(quote) < _MIN_ALIGNABLE:
        return quote, page

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

    log.info("quote not locatable anywhere in the PDF — returning it unaligned")
    return quote, page


def is_verbatim(quote: str, page: int) -> bool:
    """Whether `quote` appears on `page` of the PDF, ignoring whitespace.

    The check `scripts/audit_quotes.py` uses to put a number on grounding. Punctuation and
    case are **not** folded — only whitespace, for the reason given in the module
    docstring — so a quote that passes matches the PDF character for character.
    """
    pages = _pdf_pages()
    if not pages or not (1 <= page <= len(pages)) or not quote.strip():
        return False
    return _WS.sub("", quote) in _WS.sub("", pages[page - 1])
