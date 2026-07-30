"""Select the PDF parser used by `/ingest`.

The challenge contract is still simple: `/ingest` receives a PDF from `data/in`,
turns it into page-grounded text, chunks it, embeds it, and indexes it. We keep
the parser behind this tiny switch so the team can compare extraction quality
without touching the rest of the RAG pipeline.
"""

from __future__ import annotations

from pathlib import Path

from . import docling_parser


def active_parser_name() -> str:
    return "docling"


def extract_pages(pdf_path: Path) -> list[str]:
    return docling_parser.extract_pages(pdf_path)


def extract_blocks(pdf_path: Path) -> list[docling_parser.Block] | None:
    """Labelled blocks in reading order, or None if the active parser has no structure.

    `chunk_blocks` needs this; a parser that can only produce page strings returns None
    and `/ingest` falls back to `chunk_pages`.
    """
    return docling_parser.extract_blocks(pdf_path)
