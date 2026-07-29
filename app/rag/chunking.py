"""Turn PDF pages into page-grounded retrieval units.

The chunker keeps chunks small enough for dense embeddings, while preserving page
numbers for citations. It writes ordinary text chunks, so the same files can later
feed BM25 and a reranker without adding a second parsing path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import get_settings

_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\s+[A-Z][^\n]{2,120}$")
_SHORT_CAPTION_RE = re.compile(r"^(Fig\.|Figure|Table)\s*\d+[.:]?\s+", re.IGNORECASE)
_MIN_CHUNK_CHARS = 180


@dataclass
class Chunk:
    text: str
    page: int      # 1-indexed
    index: int     # position within the document
    section: str = ""
    kind: str = "text"


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
            for i in range(0, len(sentence), max_chars):
                out.append(sentence[i : i + max_chars].strip())
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


def _overlap_tail(blocks: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0:
        return []
    tail: list[str] = []
    total = 0
    for block in reversed(blocks):
        if total + len(block) > overlap_chars and total >= _MIN_CHUNK_CHARS:
            break
        tail.insert(0, block)
        total += len(block) + 2
    return tail


def _emit(
    chunks: list[Chunk],
    parts: list[str],
    page_no: int,
    index: int,
    section: str,
) -> int:
    text = "\n\n".join(part.strip() for part in parts if part.strip()).strip()
    if not text:
        return index
    kinds = {_kind(part) for part in parts}
    kind = "table" if "table" in kinds else "caption" if "caption" in kinds else "text"
    chunks.append(Chunk(text=text, page=page_no, index=index, section=section, kind=kind))
    return index + 1


def chunk_pages(pages: list[str]) -> list[Chunk]:
    """Split pages into paragraph-aware chunks with bounded overlap.

    Strategy:
    - keep every chunk tied to one PDF page, so citations remain simple;
    - split on Docling's blank-line blocks, which usually preserve headings,
      captions and table-ish text;
    - keep chunks near CHUNK_SIZE characters, with configurable overlap defaulting to zero;
    - store section/kind metadata for later BM25 + reranker integration.
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
