"""Load the committed PDF into the vector store.

    parse PDF (data/in) -> pages -> [chunk] -> embeddings -> Qdrant

Parser note: we tested GROBID and Marker separately, then chose Marker for this
document because its Markdown kept page boundaries and tables in a RAG-friendly form.
GROBID code was removed so there is one parser path to reason about.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from qdrant_client import models

from ..config import get_settings
from ..models import IngestResponse
from ..vectorstore.qdrant_store import get_store
from .chunking import Chunk, chunk_pages
from .embeddings import get_embedder
from .marker_parser import extract_pages

# A fixed namespace so re-ingesting the same document overwrites its points
# (idempotent ids) instead of duplicating them.
_NAMESPACE = uuid.UUID("6f0d9b1e-3b7a-4c2e-9a1d-000000000000")


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


<<<<<<< HEAD
=======
def extract_pages(path: Path) -> list[str]:
    """Per-page text via pypdf.

    TODO(level-1): pypdf is fine for clean digital PDFs and poor on complex layout
      (two columns, tables, ligatures, math). If your citations won't match the
      document, your extractor is usually why. Try pdfplumber, PyMuPDF, Docling,
      GROBID or Marker and keep whichever reads your document best.
    """
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


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
        name = f"chunk-{c.index:04d}_page-{c.page:03d}.txt"
        (chunks_dir / name).write_text(c.text, encoding="utf-8")
    return chunks_dir


>>>>>>> c511ca992b072522db5a4249b4b5bccb012a4855
def ingest(filename: str | None = None, reset: bool = False) -> IngestResponse:
    settings = get_settings()
    embedder = get_embedder()
    store = get_store()

    path = _find_pdf(filename)
    pages = extract_pages(path)
    dump_pages(pages)

    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError(f"{path.name} produced no text — is it a scanned/image PDF?")
    dump_chunks(chunks)

    # Embed in batches. is_query=False marks these as documents ("passage:" for e5).
    vectors: list[list[float]] = []
    batch = 32
    for i in range(0, len(chunks), batch):
        texts = [c.text for c in chunks[i : i + batch]]
        vectors.extend(embedder.embed(texts, is_query=False))

    store.ensure_collection(dim=len(vectors[0]), reset=reset)

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(_NAMESPACE, f"{path.name}:{c.index}")),
            vector=vec,
            payload={"text": c.text, "page": c.page, "source": path.name, "parser": "marker"},
        )
        for c, vec in zip(chunks, vectors)
    ]
    store.upsert(points)

    return IngestResponse(
        document=path.name,
        pages=len(pages),
        chunks=len(chunks),
        collection=settings.qdrant_collection,
    )
