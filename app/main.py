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

### Flow

1. Put your document in `data/in/`.
2. **`POST /ingest`** — parse it, embed it, index it in Qdrant. (Out of the box: one vector
   per page, local Ollama embeddings. No chunking yet — that is yours to add.)
3. **`POST /query`** — ask a question at a level (1, 2 or 3). You get an answer with its
   sources, and a copy is written to `data/out/`.

### The three levels

- **1 · retrieval** — a standalone question. The baseline handles these.
- **2 · memory** — a follow-up; level-2 questions share a conversation so your system can
  use the history. Resolve the follow-up *before* retrieval.
- **3 · reasoning** — questions no single passage answers. This is where you build.

Everything worth improving is marked `TODO(level-N)` in `app/rag/` and `app/rag/embeddings.py`.
This is an **open-source** challenge — any model you can host on your own machine is fair game.
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
