"""Turn a parsed PDF into page-grounded retrieval units.

Two entry points:

- `chunk_blocks` is the real one. It consumes Docling's labelled blocks, so chunk
  boundaries fall on the document's own section headings, tables stay whole, and the
  bibliography is tagged rather than mixed in with prose.
- `chunk_pages` is the fallback for a parser that can only produce page strings. It
  re-derives headings with a regex and is strictly worse; keep it only as a safety net.

Both keep every chunk tied to one PDF page, so a citation's page is never ambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import get_settings
from .docling_parser import Block

_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\s+[A-Z][^\n]{2,120}$")
_SHORT_CAPTION_RE = re.compile(r"^(Fig\.|Figure|Table)\s*\d+[.:]?\s+", re.IGNORECASE)
_MIN_CHUNK_CHARS = 180

# A numbered heading like "3.1 Model-agnostic Feature Attribution": the dotted prefix
# gives us the nesting depth that Docling's `level` (always 1 here) does not.
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+\S")

# Headings after which everything is back matter — indexed, but tagged so retrieval can
# tell a reference list entry from a claim the paper actually makes.
_BACK_MATTER = {"references", "bibliography", "acknowledgments", "acknowledgements"}


@dataclass
class Chunk:
    text: str
    page: int  # 1-indexed
    index: int  # position within the document
    section: str = ""
    kind: str = "text"
    # What gets embedded. The breadcrumb is prepended here rather than to `text` so the
    # vector knows where the passage sits, while `text` stays verbatim for quoting.
    embed_text: str = ""
    heading_path: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.embed_text:
            self.embed_text = f"{self.section}\n\n{self.text}" if self.section else self.text


def _blocks(page_text: str) -> list[str]:
    """Split one page into paragraph/table-ish blocks without losing text."""
    text = page_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    return [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]


def _kind(block: str) -> str:
    if _SHORT_CAPTION_RE.match(block):
        return "caption"
    # Docling tables often arrive as dense rows with repeated separators or many columns.
    if block.count("|") >= 4 or block.count("\t") >= 3:
        return "table"
    return "text"


def _is_heading(block: str) -> bool:
    first = block.splitlines()[0].strip()
    return bool(_HEADING_RE.match(first)) and len(block) < 180


def _split_long_block(block: str, max_chars: int) -> list[str]:
    """Split long paragraphs on sentence boundaries before falling back to hard cuts."""
    if len(block) <= max_chars:
        return [block]

    sentences = re.split(r"(?<=[.!?])\s+", block)
    out: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                out.append(current.strip())
                current = ""
            out.extend(_hard_wrap(sentence, max_chars))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            out.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current:
        out.append(current.strip())
    return [part for part in out if part]


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    """Last-resort split for a single oversized sentence, on whitespace not mid-word."""
    out: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        out.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        out.append(remaining)
    return [part for part in out if part]


def _overlap_tail(blocks: list[str], overlap_chars: int) -> list[str]:
    """Carry the last `overlap_chars` of a chunk into the next one.

    Whole blocks are preferred, but a block longer than the budget is trimmed to its
    trailing sentences rather than copied wholesale — copying it duplicated most of a
    chunk and skewed BM25 term statistics.
    """
    if overlap_chars <= 0:
        return []
    tail: list[str] = []
    total = 0
    for block in reversed(blocks):
        if total + len(block) > overlap_chars:
            if not tail:
                trimmed = _trailing_sentences(block, overlap_chars)
                if trimmed:
                    tail.insert(0, trimmed)
            break
        tail.insert(0, block)
        total += len(block) + 2
    return tail


def _trailing_sentences(block: str, budget: int) -> str:
    """The last whole sentences of `block` that fit in `budget` characters."""
    sentences = re.split(r"(?<=[.!?])\s+", block.strip())
    out: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > budget and out:
            break
        out.insert(0, sentence)
        total += len(sentence) + 1
    tail = " ".join(out).strip()
    return tail if tail and tail != block.strip() else ""


def _heading_depth(text: str, level: int | None) -> int:
    """Nesting depth of a heading, from its section number where it has one."""
    match = _NUMBERED_HEADING_RE.match(text.strip())
    if match:
        return match.group(1).count(".") + 1
    return max(1, level or 1)


def _is_real_heading(text: str) -> bool:
    """Reject what Docling mislabels as a heading.

    Layout models occasionally promote a stray fragment — a wrapped URL in the
    bibliography, say — to `section_header`. Letting one through would reset the
    breadcrumb for every chunk after it.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 160:
        return False
    if stripped.startswith(("//", "http://", "https://", "doi:")):
        return False
    letters = sum(character.isalpha() for character in stripped)
    return letters >= 3 and letters >= len(stripped) * 0.4


def _breadcrumb(path: list[str]) -> str:
    return " > ".join(path)


def _emit(
    chunks: list[Chunk],
    parts: list[str],
    page_no: int,
    index: int,
    section: str,
    kind: str = "",
    heading_path: list[str] | None = None,
    context: str = "",
) -> int:
    text = "\n\n".join(part.strip() for part in parts if part.strip()).strip()
    if not text:
        return index
    if not kind:
        kinds = {_kind(part) for part in parts}
        kind = "table" if "table" in kinds else "caption" if "caption" in kinds else "text"
    # `context` is retrieval-only scaffolding (a table's caption and header row). It
    # belongs in the vector, never in `text`, which has to stay quotable verbatim.
    embed_text = "\n\n".join(part for part in (section, context, text) if part)
    chunks.append(
        Chunk(
            text=text,
            page=page_no,
            index=index,
            section=section,
            kind=kind,
            embed_text=embed_text,
            heading_path=list(heading_path or []),
        )
    )
    return index + 1


def _split_table(block: Block, max_chars: int) -> list[tuple[str, str]]:
    """Split an oversized table on row boundaries. Returns (verbatim text, context).

    A table cut at an arbitrary character offset leaves orphan rows with no idea what
    their columns mean, which is exactly the text a "which methods do X" question needs.
    So every piece after the first gets the caption and header row — but as *context*,
    added only to what we embed. Repeating them inside `text` would make the chunk a
    span that appears nowhere on the page, and `Source.quote` has to stay verbatim.
    """
    caption = block.caption.strip()
    rows = [row for row in block.text.splitlines() if row.strip()]
    if not rows:
        return []

    # The caption is emitted immediately before its table in the page text, so
    # "caption\ntable" is itself a contiguous, quotable span.
    whole = f"{caption}\n{block.text}".strip() if caption else block.text
    if len(whole) <= max_chars:
        return [(whole, "")]

    header = rows[0]
    context = "\n".join(part for part in (caption, header) if part)
    budget = max(_MIN_CHUNK_CHARS, max_chars - len(context))

    pieces: list[tuple[str, str]] = []
    current: list[str] = []
    current_len = 0
    for row in rows:
        if current and current_len + len(row) + 1 > budget:
            pieces.append(("\n".join(current), context))
            current = []
            current_len = 0
        current.append(row)
        current_len += len(row) + 1
    if current:
        pieces.append(("\n".join(current), context))

    if pieces and caption:
        # The first piece starts at the header, so it can carry the caption verbatim.
        first, _ = pieces[0]
        pieces[0] = (f"{caption}\n{first}", "")
    return pieces or [(whole, "")]


def chunk_blocks(blocks: list[Block]) -> list[Chunk]:
    """Split labelled blocks into section-aware, page-grounded chunks.

    Strategy:
    - a chunk never spans a section heading or a page, so `section` and `page` are always
      exactly right for every sentence inside it;
    - headings build a breadcrumb ("3 FEATURE ATTRIBUTION > 3.1 Model-agnostic ..."), kept
      in `section` and prepended to `embed_text` so each vector knows where it sits;
    - tables are their own chunks, carrying their caption, split on rows not characters;
    - back matter (references, acknowledgments) is tagged `kind="reference"` so retrieval
      can down-weight a third of the corpus that answers none of the questions.
    """
    settings = get_settings()
    max_chars = max(settings.chunk_size, _MIN_CHUNK_CHARS)
    overlap_chars = max(0, min(settings.chunk_overlap, max_chars // 2))

    chunks: list[Chunk] = []
    idx = 0
    heading_path: list[str] = []
    in_back_matter = False

    # The heading is held apart from the body blocks: it opens its section's first chunk
    # but never becomes a chunk of its own, and it does not count against `max_chars`.
    pending_heading: str | None = None
    pending_heading_page: int | None = None
    pending: list[str] = []
    pending_len = 0
    pending_page: int | None = None

    def take_heading(page: int) -> list[str]:
        """Claim the pending heading if it sits on the same page as the chunk using it."""
        nonlocal pending_heading, pending_heading_page
        if pending_heading is None or pending_heading_page != page:
            return []
        claimed = [pending_heading]
        pending_heading = pending_heading_page = None
        return claimed

    def flush() -> None:
        nonlocal idx, pending, pending_len, pending_page
        if pending and pending_page is not None:
            parts = take_heading(pending_page) + pending
            kind = "reference" if in_back_matter else ""
            idx = _emit(
                chunks, parts, pending_page, idx, _breadcrumb(heading_path), kind, heading_path
            )
            pending = _overlap_tail(pending, overlap_chars)
            pending_len = sum(len(part) + 2 for part in pending)
        else:
            pending = []
            pending_len = 0
        if not pending:
            pending_page = None

    def drop_stale_heading(page: int) -> None:
        """Forget a heading whose section body starts on the next page.

        Emitting it alone would make a 20-character chunk; carrying it onto the next page
        would put a page-N quote under a page-N+1 citation. The breadcrumb in `section`
        and `embed_text` keeps the information either way.
        """
        nonlocal pending_heading, pending_heading_page
        if pending_heading is not None and pending_heading_page != page:
            pending_heading = pending_heading_page = None

    for block in blocks:
        if block.label == "section_header" and _is_real_heading(block.text):
            flush()
            pending, pending_len, pending_page = [], 0, None

            depth = _heading_depth(block.text, block.level)
            heading_path = heading_path[: depth - 1]
            heading_path.append(block.text.strip())
            # Back matter never returns to body text, so this latch is one-way.
            if block.text.strip().lower() in _BACK_MATTER:
                in_back_matter = True
            pending_heading = block.text.strip()
            pending_heading_page = block.page
            continue

        if block.label == "table":
            flush()
            pending, pending_len, pending_page = [], 0, None
            prefix = take_heading(block.page)
            for piece, context in _split_table(block, max_chars):
                idx = _emit(
                    chunks,
                    prefix + [piece],
                    block.page,
                    idx,
                    _breadcrumb(heading_path),
                    "table",
                    heading_path,
                    context,
                )
                prefix = []
            drop_stale_heading(block.page)
            continue

        if block.label == "picture":
            flush()
            pending, pending_len, pending_page = [], 0, None
            idx = _emit(
                chunks,
                take_heading(block.page) + [block.text],
                block.page,
                idx,
                _breadcrumb(heading_path),
                "caption",
                heading_path,
            )
            drop_stale_heading(block.page)
            continue

        # A chunk stays on one page so its citation page is unambiguous.
        if pending_page is not None and block.page != pending_page:
            flush()
            pending, pending_len, pending_page = [], 0, None
        drop_stale_heading(block.page)

        for part in _split_long_block(block.text, max_chars):
            part_len = len(part) + 2
            if pending and pending_len + part_len > max_chars:
                flush()
                if pending and pending_len + part_len > max_chars:
                    pending, pending_len = [], 0
            pending.append(part)
            pending_len += part_len
            pending_page = block.page

    flush()
    return chunks


def chunk_pages(pages: list[str]) -> list[Chunk]:
    """Fallback chunker for parsers that only yield page strings.

    Headings are guessed with a regex, so sections are less reliable and tables are cut
    on characters. Prefer `chunk_blocks`.
    """
    settings = get_settings()
    max_chars = max(settings.chunk_size, _MIN_CHUNK_CHARS)
    overlap_chars = max(0, min(settings.chunk_overlap, max_chars // 2))

    chunks: list[Chunk] = []
    idx = 0
    current_section = ""

    for page_no, text in enumerate(pages, start=1):
        page_blocks: list[str] = []
        for block in _blocks(text):
            page_blocks.extend(_split_long_block(block, max_chars))
        if not page_blocks:
            continue

        current_parts: list[str] = []
        current_len = 0
        for block in page_blocks:
            if _is_heading(block):
                if current_parts:
                    idx = _emit(chunks, current_parts, page_no, idx, current_section)
                    current_parts = _overlap_tail(current_parts, overlap_chars)
                    current_len = sum(len(p) + 2 for p in current_parts)
                current_section = block.splitlines()[0].strip()

            block_len = len(block) + 2
            if current_parts and current_len + block_len > max_chars:
                idx = _emit(chunks, current_parts, page_no, idx, current_section)
                current_parts = _overlap_tail(current_parts, overlap_chars)
                current_len = sum(len(p) + 2 for p in current_parts)
                if current_parts and current_len + block_len > max_chars:
                    current_parts = []
                    current_len = 0

            current_parts.append(block)
            current_len += block_len

        if current_parts:
            idx = _emit(chunks, current_parts, page_no, idx, current_section)

    return chunks
