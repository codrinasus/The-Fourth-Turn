"""Find the chunks most relevant to a question."""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import Message
from .embeddings import get_embedder
from .bm25_index import BM25Index


@dataclass
class Context:
    text: str
    page: int
    score: float


def rewrite_query(question: str, history: list[Message]) -> str:
    """Resolve a follow-up into a standalone search query.

    TODO(level-2): THIS IS THE KEY FUNCTION FOR CONVERSATIONAL RETRIEVAL and right
      now it is a no-op. "And the test split?" has no retrievable content on its own,
      so embedding it returns noise. Use the client's chat model to rewrite the
      question against `history` into something self-contained
      ("How large is the test split of <the dataset from the previous turn>?"),
      then retrieve with that. Leave genuinely standalone questions unchanged.
    """
    if not history:
        return question
    # Baseline: ignores history. Replace this.
    return question


def retrieve(question: str, top_k: int, history: list[Message] | None = None) -> list[Context]:
    embedder = get_embedder()
    store = get_store()
    query = rewrite_query(question, history or [])

    # If BM25 is enabled in settings, prefer sparse retrieval from the chunk files.
    # build a singleton BM25 index lazily
    if not hasattr(retrieve, "_bm25_index") or retrieve._bm25_index is None:
        retrieve._bm25_index = BM25Index()
        retrieve._bm25_index.build_from_dir("data/pages", glob="**/*.txt")
    bm25_idx: BM25Index = retrieve._bm25_index
    hits = bm25_idx.search(query, top_k=top_k)
    return [
        Context(text=h.text, page=int(h.page or 0), score=float(h.score)) for h in hits
    ]

    # Default: dense vector retrieval via the embedder + Qdrant store
    # TODO(level-3): one query + one search is not enough for whole-document
    #   questions ("summarise every chapter", "combine the table on p.40 with the
    #   reference on p.90"). Consider multi-query fan-out, iterative/agentic retrieval
    #   (retrieve -> reason -> retrieve again), or a second index (e.g. a graph or a
    #   per-section summary index) alongside this one.
    vector = embedder.embed([query], is_query=True)[0]
    hits = store.search(vector, top_k)

    return [
        Context(
            text=str(h.payload.get("text", "")),
            page=int(h.payload.get("page", 0)),
            score=float(h.score),
        )
        for h in hits
    ]
