"""Find the chunks most relevant to a question."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings
from ..llm.base import Message
from ..vectorstore.qdrant_store import get_store
from . import decompose, rerank, rewrite
from .bm25_index import BM25Index
from .embeddings import get_embedder

log = logging.getLogger(__name__)

_RRF_K = 60  # the standard reciprocal-rank-fusion damping constant
_REFERENCE_WEIGHT = 0.3  # bibliography chunks compete, but only when nothing else does


@dataclass
class Context:
    text: str
    page: int
    score: float
    kind: str = "text"


@dataclass
class Retrieval:
    """What retrieval found, plus how it got there.

    `query` is what was actually embedded — for a level-2 follow-up that is the rewritten
    standalone question, not what the user typed. Carrying it back out is what lets the
    response report the resolution instead of leaving it invisible in the logs.
    """

    contexts: list[Context]
    query: str
    sub_queries: list[str] = field(default_factory=list)


def rewrite_query(question: str, history: list[Message]) -> str:
    """Resolve a follow-up into a standalone search query — the Level-2 seam.

    Delegates to `rag/rewrite.py`, which gates on whether the question can be searched as
    written and only then asks the chat model to substitute the antecedents. Standalone
    questions come back untouched.
    """
    return rewrite.standalone_query(question, history)


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
    return [
        Context(
            text=h.text,
            page=int(h.page or 0),
            score=float(h.score),
            kind=h.kind or "text",
        )
        for h in hits
    ]


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
            kind=str(h.payload.get("kind", "text")),
        )
        for h in hits
    ]


def _fuse(rankings: list[list[Context]], top_k: int) -> list[Context]:
    """Reciprocal-rank fusion over any number of ranked lists.

    Generic in the number of lists because it now merges more than two things: dense and
    BM25 for one query, and then one such pair per sub-query when a Level-3 question is
    decomposed. RRF is the right tool for both — it needs only ranks, so it can combine
    lists whose scores are on completely different scales (cosine vs BM25 vs another
    query's cosine) without any normalisation.

    Bibliography entries are down-weighted rather than excluded. They are 85 of our 269
    chunks and the cross-encoder scores them ~0.007, so left alone they crowd the pool
    the answer is built from — but "how many papers does the survey review?" is a fair
    question whose evidence lives in exactly those chunks, so a hard filter would be
    wrong. A weight lets them compete when nothing in the body matches.
    """
    by_key: dict[tuple[int, str], Context] = {}
    fused: dict[tuple[int, str], float] = {}

    for candidates in rankings:
        for rank, ctx in enumerate(candidates, start=1):
            key = (ctx.page, ctx.text)
            by_key.setdefault(key, ctx)
            weight = _REFERENCE_WEIGHT if ctx.kind == "reference" else 1.0
            fused[key] = fused.get(key, 0.0) + weight / (_RRF_K + rank)

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        Context(
            text=by_key[key].text,
            page=by_key[key].page,
            score=score,
            kind=by_key[key].kind,
        )
        for key, score in ranked
    ]


def _dedup(contexts: list[Context]) -> list[Context]:
    """Drop chunks whose text is already covered by a higher-ranked one.

    CHUNK_OVERLAP=150 means adjacent chunks share a sentence or two, so the same page-1
    passage could take two of the five prompt slots — we saw exactly that. Containment
    rather than equality, because the overlap makes one chunk a prefix/suffix of another
    rather than its twin.
    """
    kept: list[Context] = []
    for ctx in contexts:
        body = " ".join(ctx.text.split())
        if any(body in " ".join(k.text.split()) or " ".join(k.text.split()) in body for k in kept):
            continue
        kept.append(ctx)
    return kept


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
        Context(text=c.text, page=c.page, score=s, kind=c.kind) for c, s in zip(candidates, scores)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored[:top_k]


def _arms(query: str, depth: int) -> list[list[Context]]:
    """One query's two ranked lists: dense and BM25.

    Returned unfused so the caller can fuse everything at once — with several sub-queries
    in play, fusing per query and then fusing the results would apply RRF twice and bury
    a chunk that only one sub-query found, which is exactly the chunk decomposition
    exists to rescue.
    """
    sparse = _bm25_search(query, depth)
    if not sparse:
        log.warning("BM25 returned nothing (is data/chunks populated?) — dense-only retrieval")
        return [_dense_search(query, depth)]
    return [_dense_search(query, depth), sparse]


def retrieve(
    question: str,
    top_k: int,
    history: list[Message] | None = None,
    level: int = 1,
) -> Retrieval:
    settings = get_settings()
    # Resolve the follow-up BEFORE embedding. The prompt gets the history either way, but
    # the retriever only ever sees this string, so an unresolved "why does that happen?"
    # would search for nothing at all.
    query = rewrite_query(question, history or [])

    # Retrieve more candidates than we finally show: the wider pool is what gives the
    # cross-encoder something to fix, while the prompt still only sees top_k chunks.
    depth = max(top_k, settings.rerank_candidates if settings.reranker_enabled else top_k * 3)

    # Level 3 only: give each hop of a multi-part question its own search, so a chunk that
    # answers just one hop is not out-ranked by chunks matching the question's bulk.
    subs = decompose.sub_queries(question) if level >= 3 else []
    rankings = [ranking for sub in subs for ranking in _arms(sub, depth)]
    rankings.extend(_arms(query, depth))  # the question itself always gets a vote

    # Fuse the whole union and let the cross-encoder see all of it. Previously fusion cut
    # to `depth` first, throwing away a third of the unique candidates before the only
    # component with calibrated scores got a look at them.
    pool = _dedup(_fuse(rankings, settings.max_rerank_pool))

    # Rerank against the resolved query for the same reason we searched with it: scoring
    # a chunk against "why does that happen?" tells the cross-encoder nothing either.
    return Retrieval(
        contexts=_rerank(query, pool, top_k),
        query=query,
        sub_queries=subs,
    )
