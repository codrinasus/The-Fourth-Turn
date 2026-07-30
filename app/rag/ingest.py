"""Load the committed PDF into the vector store.

    parse PDF (data/in) -> pages -> chunks -> embeddings -> Qdrant

Parser note: Docling is the active path because `/ingest` can call its local
HTTP service directly.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from qdrant_client import models

from ..config import get_settings
from ..models import IngestResponse
from ..vectorstore.qdrant_store import get_store
from . import sections
from .chunking import Chunk, chunk_blocks, chunk_pages
from .embeddings import get_embedder
from .pdf_parser import active_parser_name, extract_blocks, extract_pages
from .retrieve import reset_retrieval_indexes

# A fixed namespace so re-ingesting the same document overwrites its points
# (idempotent ids) instead of duplicating them.
_NAMESPACE = uuid.UUID("6f0d9b1e-3b7a-4c2e-9a1d-000000000000")

log = logging.getLogger(__name__)


def _find_pdf(filename: str | None) -> Path:
    in_dir = Path(get_settings().in_dir)
    if filename:
        path = in_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"no such PDF: {path}")
        return path
    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no *.pdf found in {in_dir}/ — put your document there first")
    return pdfs[0]


def _fresh_dir(name: str) -> Path:
    """An empty data/<name>/ — wiped and recreated so it always matches the last ingest."""
    path = Path(get_settings().in_dir).parent / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_pages(pages: list[str]) -> Path:
    """Write each extracted page to data/pages/page-NNN.txt for inspection."""
    pages_dir = _fresh_dir("pages")
    for i, text in enumerate(pages, start=1):
        (pages_dir / f"page-{i:03d}.txt").write_text(text, encoding="utf-8")
    return pages_dir


def dump_chunks(chunks: list[Chunk]) -> Path:
    """Write each chunk to data/chunks/chunk-NNNN_page-NNN.txt for inspection.

    The page number is in the filename so you can eyeball whether citations will
    line up with the PDF.
    """
    chunks_dir = _fresh_dir("chunks")
    for c in chunks:
        name = f"chunk-{c.index:04d}_page-{c.page:03d}_{c.kind}.txt"
        (chunks_dir / name).write_text(c.text, encoding="utf-8")
    return chunks_dir


def ingest(filename: str | None = None, reset: bool = False) -> IngestResponse:
    settings = get_settings()
    embedder = get_embedder()
    store = get_store()

    path = _find_pdf(filename)
    pages = extract_pages(path)
    dump_pages(pages)

    # Prefer the structure-aware path: chunk boundaries then fall on the document's own
    # section headings instead of on a character count.
    blocks = extract_blocks(path)
    chunks = chunk_blocks(blocks) if blocks else chunk_pages(pages)
    if not chunks:
        raise ValueError(f"{path.name} produced no text — is it a scanned/image PDF?")
    dump_chunks(chunks)
    reset_retrieval_indexes()

    # Embed in batches. is_query=False marks these as documents ("passage:" for e5).
    # `embed_text` carries the section breadcrumb, so a chunk from deep inside a section
    # still vectorises as belonging to it; the payload keeps `text` verbatim for quoting.
    vectors: list[list[float]] = []
    batch = 32
    for i in range(0, len(chunks), batch):
        texts = [c.embed_text for c in chunks[i : i + batch]]
        vectors.extend(embedder.embed(texts, is_query=False))

    store.ensure_collection(dim=len(vectors[0]), reset=reset)

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(_NAMESPACE, f"{path.name}:{c.index}")),
            vector=vec,
            payload={
                "text": c.text,
                "page": c.page,
                "chunk_index": c.index,
                "section": c.section,
                "heading_path": c.heading_path,
                "kind": c.kind,
                "source": path.name,
                "parser": active_parser_name(),
            },
        )
        for c, vec in zip(chunks, vectors)
    ]
    store.upsert(points)

    # The Level-3 second index: one summary per section, over the same chunks. Built last
    # so a failure here still leaves a complete, queryable chunk index behind.
    try:
        n_sections = sections.build_index(chunks, source=path.name, reset=reset)
        log.info("indexed %d section summaries", n_sections)
    except Exception as e:  # noqa: BLE001 - the chunk index is what /query needs to work
        log.warning("section index failed (%s) — level-3 outline will be empty", e)

    return IngestResponse(
        document=path.name,
        pages=len(pages),
        chunks=len(chunks),
        collection=settings.qdrant_collection,
    )
