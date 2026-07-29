"""Find the chunks most relevant to a question."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..llm.base import Message
from ..vectorstore.qdrant_store import get_store
from .bm25_index import BM25Index
from .embeddings import get_embedder


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


def _bm25_search(query: str, top_k: int) -> list[Context]:
    chunks_dir = Path("data/chunks")
    if not chunks_dir.exists():
        return []
    if not hasattr(retrieve, "_bm25_index") or retrieve._bm25_index is None:
        retrieve._bm25_index = BM25Index()
        try:
            retrieve._bm25_index.build_from_dir(chunks_dir, glob="**/*.txt")
        except (FileNotFoundError, ValueError):
            return []
    hits = retrieve._bm25_index.search(query, top_k=top_k)
    return [Context(text=h.text, page=int(h.page or 0), score=float(h.score)) for h in hits]


def reset_retrieval_indexes() -> None:
    """Clear file-backed retrieval caches after ingest rewrites data/chunks."""
    retrieve._bm25_index = None


def _dense_search(query: str, top_k: int) -> list[Context]:
    embedder = get_embedder()
    store = get_store()
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


def _hybrid_fuse(dense: list[Context], sparse: list[Context], top_k: int) -> list[Context]:
    """Merge dense and BM25 candidates with reciprocal-rank fusion.

    This is deliberately simple. The next scoring step should be a reranker over this
    merged candidate pool, not a second parser or a hand-written answer shortcut.
    """
    by_key: dict[tuple[int, str], Context] = {}
    fused_scores: dict[tuple[int, str], float] = {}

    for weight, candidates in ((1.0, dense), (1.0, sparse)):
        for rank, ctx in enumerate(candidates, start=1):
            key = (ctx.page, ctx.text)
            by_key.setdefault(key, ctx)
            fused_scores[key] = fused_scores.get(key, 0.0) + weight / (60 + rank)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        Context(
            text=by_key[key].text,
            page=by_key[key].page,
            score=score,
        )
        for key, score in ranked
    ]


def retrieve(question: str, top_k: int, history: list[Message] | None = None) -> list[Context]:
    query = rewrite_query(question, history or [])

    # Retrieve more candidates than we finally show. This gives the future reranker a
    # useful pool while keeping the LLM context limited to top_k chunks.
    candidate_k = max(top_k, min(top_k * 3, 20))
    dense = _dense_search(query, candidate_k)
    sparse = _bm25_search(query, candidate_k)
    if not sparse:
        return dense[:top_k]
    return _hybrid_fuse(dense, sparse, top_k)
