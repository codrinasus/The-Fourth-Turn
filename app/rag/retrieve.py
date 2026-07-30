"""Find the chunks most relevant to a question."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings
from ..llm.base import Message
from ..vectorstore.qdrant_store import get_store
from . import rerank
from .bm25_index import BM25Index
from .embeddings import get_embedder

log = logging.getLogger(__name__)


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

    Deliberately simple: fusion decides which candidates the cross-encoder gets to see
    (`rerank_pool` below), and the cross-encoder decides the final order.
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


def _rerank(query: str, candidates: list[Context], top_k: int) -> list[Context]:
    """Rescore the pool with the cross-encoder, keeping fusion order as the fallback."""
    if not get_settings().reranker_enabled or len(candidates) <= 1:
        return candidates[:top_k]
    try:
        scores = rerank.score(query, [c.text for c in candidates])
    except rerank.RerankError as e:
        log.warning("%s — keeping fusion order", e)
        return candidates[:top_k]

    rescored = [
        Context(text=c.text, page=c.page, score=s) for c, s in zip(candidates, scores)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored[:top_k]


def retrieve(question: str, top_k: int, history: list[Message] | None = None) -> list[Context]:
    settings = get_settings()
    query = rewrite_query(question, history or [])

    # Retrieve more candidates than we finally show: the wider pool is what gives the
    # cross-encoder something to fix, while the prompt still only sees top_k chunks.
    pool_size = max(top_k, settings.rerank_candidates if settings.reranker_enabled else top_k * 3)
    dense = _dense_search(query, pool_size)
    sparse = _bm25_search(query, pool_size)
    if not sparse:
        log.warning("BM25 returned nothing (is data/chunks populated?) — dense-only retrieval")
        pool = dense
    else:
        pool = _hybrid_fuse(dense, sparse, pool_size)

    return _rerank(query, pool, top_k)
