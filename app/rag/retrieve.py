"""Find the chunks most relevant to a question."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings
from ..llm.base import Message
from ..vectorstore.qdrant_store import get_store
from . import agent, decompose, rerank, rewrite
from .bm25_index import BM25Index
from .embeddings import get_embedder

log = logging.getLogger(__name__)

_RRF_K = 60  # the standard reciprocal-rank-fusion damping constant
_REFERENCE_WEIGHT = 0.3  # bibliography chunks compete, but only when nothing else does

# How loudly each kind of query votes in fusion. The user's own question is the only
# string we know states what they want; a decomposition is our reading of it, and a
# reflective follow-up is a guess about what is still missing. Ordering them this way is
# what keeps extra retrieval rounds additive instead of disruptive.
_PRIMARY_WEIGHT = 1.0
_SUB_QUERY_WEIGHT = 0.7
_FOLLOW_UP_WEIGHT = 0.4


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
    steps: list[str] = field(default_factory=list)


def rewrite_query(question: str, history: list[Message]) -> str:
    """Resolve a follow-up into a standalone search query — the Level-2 seam.

    Delegates to `rag/rewrite.py`, where the model reads the conversation and writes the
    query. Self-contained questions come back untouched — that judgement is the model's,
    not a heuristic's.

    `REWRITE_ENABLED=false` restores the un-resolved behaviour, so the Level-2 ablation in
    TECHNICAL_NOTE.md can be reproduced rather than taken on trust.
    """
    if not get_settings().rewrite_enabled:
        return question
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


def _fuse(rankings: list[tuple[float, list[Context]]], top_k: int) -> list[Context]:
    """Reciprocal-rank fusion over any number of ranked lists.

    Generic in the number of lists because it now merges more than two things: dense and
    BM25 for one query, and then one such pair per sub-query when a Level-3 question is
    decomposed. RRF is the right tool for both — it needs only ranks, so it can combine
    lists whose scores are on completely different scales (cosine vs BM25 vs another
    query's cosine) without any normalisation.

    Each ranking carries a weight, because not every query deserves an equal vote. The
    question the user actually asked, and its decomposition, are what the answer must
    address; the follow-up queries a Level-3 round invents are speculative. Weighting them
    equally measurably hurt: with the reflective loop on, q8's top passage fell from 0.89
    to ~0.00, because six invented section-specific queries out-voted the one paragraph
    that actually describes the paper's structure. Adding lists shifts every RRF score, so
    "a wasted round cannot displace an earlier hit" is only true if the extra lists count
    for less. Now they do.

    Bibliography entries are down-weighted for a related reason. In a survey they were 85
    of 269 chunks and the cross-encoder scored them ~0.007, so left alone they crowd the
    pool the answer is built from — but "how many papers does this review?" is a fair
    question whose evidence lives in exactly those chunks, so a hard filter would be
    wrong. A weight lets them compete when nothing in the body matches.
    """
    by_key: dict[tuple[int, str], Context] = {}
    fused: dict[tuple[int, str], float] = {}

    for list_weight, candidates in rankings:
        for rank, ctx in enumerate(candidates, start=1):
            key = (ctx.page, ctx.text)
            by_key.setdefault(key, ctx)
            weight = list_weight * (_REFERENCE_WEIGHT if ctx.kind == "reference" else 1.0)
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


def _pool(
    base: list[tuple[float, list[Context]]],
    extra: list[tuple[float, list[Context]]],
    limit: int,
) -> list[Context]:
    """The candidates the cross-encoder gets to see, with the base query's best reserved.

    This exists because of a measured failure. The cross-encoder scores every candidate
    *independently*, so more candidates can never lower an existing one's score — the only
    way a good passage loses is by never reaching the reranker at all. That is exactly what
    the reflective loop was doing: on q8 the paragraph listing what each section does
    scored **0.894** with the loop off, and with it on that chunk had been out-voted in
    fusion, fell below the pool cut, and was never scored. The answer was built from
    passages scoring 0.03.

    Weighting the speculative queries down helped but could not fix it, because the
    problem is a hard cut, not a soft ordering. So half the pool is reserved for the
    fusion of the question and its decomposition; follow-up rounds fill the rest. Extra
    retrieval is then genuinely additive — it can contribute a passage, never evict one.
    """
    # With nothing speculative to make room for, the base query gets the whole pool —
    # reserving half of it would otherwise silently halve Levels 1 and 2, which never
    # produce follow-ups at all.
    if not extra:
        return _dedup(_fuse(base, limit))[:limit]

    reserved = _fuse(base, max(1, limit // 2))
    seen = {(c.page, c.text) for c in reserved}
    rest = [c for c in _fuse(base + extra, limit) if (c.page, c.text) not in seen]
    return _dedup(reserved + rest)[:limit]


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
    base = [(_SUB_QUERY_WEIGHT, r) for sub in subs for r in _arms(sub, depth)]
    # The question itself always gets a vote, and the loudest one: it is the only string we
    # know states what the user wants. Everything else is our paraphrase of it.
    base.extend((_PRIMARY_WEIGHT, r) for r in _arms(query, depth))
    follow_up_rankings: list[tuple[float, list[Context]]] = []

    searched = [query, *subs]
    steps = [f"search: {len(searched)} quer{'y' if len(searched) == 1 else 'ies'}"]

    # Level 3 only: read what came back and search again for whatever is missing. Follow-up
    # rounds are kept separate from `base` so `_pool` can reserve room for the question's
    # own results — extra retrieval contributes passages, it never evicts them.
    if level >= 3 and settings.agent_enabled:
        for step in range(settings.agent_max_steps):
            # Judge against the best of what we have, not the raw union: the top of the
            # pool is what the answer would actually be built from right now.
            so_far = _pool(base, follow_up_rankings, settings.max_rerank_pool)
            best = _rerank(query, so_far, top_k)
            enough, missing, follow_ups = agent.next_queries(
                question, [c.text for c in best], searched
            )
            if enough:
                steps.append(f"step {step + 1}: evidence sufficient")
                break
            log.info("step %d: missing %r — searching %s", step + 1, missing, follow_ups)
            steps.append(f"step {step + 1}: missing '{missing}' → {len(follow_ups)} more queries")
            follow_up_rankings.extend(
                (_FOLLOW_UP_WEIGHT, r) for q in follow_ups for r in _arms(q, depth)
            )
            searched.extend(follow_ups)
        else:
            steps.append(f"step budget ({settings.agent_max_steps}) reached")

    # The cross-encoder sees the whole pool. Fusion used to cut to `depth` first, throwing
    # away a third of the unique candidates before the only calibrated component saw them.
    pool = _pool(base, follow_up_rankings, settings.max_rerank_pool)

    # Rerank against the resolved query for the same reason we searched with it: scoring
    # a chunk against "why does that happen?" tells the cross-encoder nothing either.
    return Retrieval(
        contexts=_rerank(query, pool, top_k),
        query=query,
        steps=steps,
        sub_queries=subs,
    )
