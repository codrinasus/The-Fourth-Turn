"""FastAPI application.

    uv run uvicorn app.main:app --port 8791 --reload    # local
    docker compose up --build                            # containers + Qdrant

Swagger UI:  http://localhost:8791/docs
ReDoc:       http://localhost:8791/redoc
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from . import __version__
from .routes import collections, health, ingest, query

DESCRIPTION = """
A retrieval-augmented-generation backend over **one PDF you choose**, for the AI Multimedia
Lab hackathon at ESSIR 2026 (*The Fourth Turn*).

Team **KrautWineSarmale**. Everything runs locally — Ollama (`qwen3.6`, `bge-m3`), a
self-hosted Docling for parsing, `bge-reranker-v2-m3` in llama.cpp, and Qdrant. No hosted
API is used.

### Flow

1. The document is committed at `data/in/document.pdf`.
2. **`POST /ingest`** — Docling parses it into labelled blocks; a section-aware chunker
   produces page-bounded chunks (269 here); `bge-m3` embeds them into Qdrant. A **second
   index** of one summary per section (44) is built alongside, for Level 3.
3. **`POST /query`** — ask a question at a level (1, 2 or 3). You get an answer, the
   passages it actually cited, and a copy is written to `data/out/`.

### The three levels

- **1 · retrieval** — a standalone question. Hybrid dense + BM25, fused with RRF and
  reranked by a cross-encoder.
- **2 · memory** — a follow-up. Level-2 questions share a conversation, and the follow-up is
  resolved into a standalone query **before** retrieval — the prompt gets the history, but
  the retriever only ever sees the query. See `diagnostics.retrieval_query`.
- **3 · reasoning** — questions no single passage answers. The question is decomposed into
  sub-questions retrieved separately (`diagnostics.sub_queries`), and the section outline is
  supplied as a document map.

Every `sources[].quote` is *sliced* from indexed page text, never generated, then located in
the PDF itself so it is returned in the file's own characters with its page verified.
`uv run python scripts/audit_quotes.py` checks all of them against a second PDF extractor.
"""


def _configure_logging() -> None:
    """Let the pipeline's own INFO logs reach the console.

    uvicorn configures only its own loggers, so without this every `log.info` in
    `app/rag/` is swallowed — including the ones that make the interesting steps
    observable: what a follow-up was rewritten to, how a Level-3 question was decomposed,
    and when a citation's page number was corrected against the PDF. Those lines are the
    cheapest evidence that the system is doing what the technical note claims.
    """
    # Attach to our own package logger rather than calling basicConfig: uvicorn configures
    # the root logger *before* importing this module, so basicConfig would find handlers
    # already present and silently do nothing.
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s | %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)


def create_app() -> FastAPI:
    _configure_logging()
    app = FastAPI(
        title="The Fourth Turn — AIMultimediaLab @ ESSIR 2026",
        version=__version__,
        description=DESCRIPTION,
    )

    app.include_router(health.router)
    app.include_router(collections.router)
    app.include_router(ingest.router)
    app.include_router(query.router)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
