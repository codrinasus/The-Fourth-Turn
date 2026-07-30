"""Cross-encoder reranking of the retrieved candidates.

Dense search and BM25 each score a chunk *without* looking at the question and the chunk
together — one compares two embeddings, the other counts terms. A cross-encoder reads the
pair jointly, so it can tell that a chunk sharing many query words is still off-topic.
That is why reranking usually buys more precision than tuning either retrieval arm.

We serve BAAI/bge-reranker-v2-m3 (the cross-encoder sibling of the bge-m3 embedder used for
retrieval) through llama.cpp's OpenAI-shaped `/v1/rerank`. See docker-compose.reranker.yml.

If the service is unreachable the caller keeps the fusion order — a missing reranker
degrades quality, it must never fail a query.
"""

from __future__ import annotations

import logging
import math

import httpx

from ..config import get_settings

log = logging.getLogger(__name__)


class RerankError(RuntimeError):
    """The reranker could not score this batch."""


def _relevance(logit: float) -> float:
    """Squash the cross-encoder logit into 0-1 so `Source.score` stays readable."""
    return 1.0 / (1.0 + math.exp(-logit))


def score(query: str, documents: list[str]) -> list[float]:
    """Return one relevance score per document, in the order given."""
    settings = get_settings()
    url = settings.reranker_base_url.rstrip("/") + "/v1/rerank"
    try:
        resp = httpx.post(
            url,
            json={"query": query, "documents": documents, "top_n": len(documents)},
            timeout=settings.reranker_timeout,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
    except Exception as e:  # any failure here simply means "no reranker this turn"
        raise RerankError(f"rerank failed: {e}") from e

    scores = [0.0] * len(documents)
    for r in results:
        idx = int(r["index"])
        if 0 <= idx < len(scores):
            scores[idx] = _relevance(float(r["relevance_score"]))
    return scores
