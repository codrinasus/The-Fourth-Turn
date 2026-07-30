"""Answer a question end to end.

    history + retrieved context  ->  prompt  ->  LLM  ->  grounded answer

This is what `POST /query` calls. You send only a question and its level; the system
assigns the id, threads the conversation (level-2 follow-ups share memory), produces the
answer, and writes it to `data/out/` as a JSON file you can later copy into `submission/`.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_settings
from ..llm.base import LLMError, Message
from ..llm.factory import get_client
from ..models import Diagnostics, QueryRequest, QueryResponse, Source
from . import citations, memory, sections, verbatim
from .retrieve import Context, retrieve

SYSTEM_PROMPT = (
    "You answer questions about a single document using only the context provided. "
    "If the context does not contain the answer, say so plainly rather than guessing. "
    "Be specific and concise.\n"
    "The context passages are numbered (1), (2), (3). Cite the passage you took each claim "
    "from by writing its number in parentheses right after the claim — for example: "
    "'The authors evaluate on MS MARCO (2).' For several passages supporting one claim, put "
    "them in a single pair of parentheses separated by commas: '(2, 6)'. Put nothing else "
    "inside a citation's parentheses — no section numbers, no words.\n"
    "Cite only numbered passages you actually used; do not cite a passage you did not rely "
    "on, and do not invent numbers. If an outline of the document is provided, it is "
    "orientation only: it has no number and must never be cited."
)


def _build_messages(
    question: str,
    contexts: list[Context],
    history: list[Message],
    outline: str = "",
) -> list[Message]:
    context_block = citations.number_contexts(contexts)
    messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Prior turns give the model the conversation so far (Level 2). Retrieval still
    # needs the rewritten query — history in the prompt is necessary but not sufficient.
    messages.extend(history)

    # The Level-3 outline goes in unnumbered and explicitly labelled as generated. Only
    # the numbered passages can be cited, so a summary can orient the answer but can
    # never end up as a quote — see rag/sections.py on that boundary.
    outline_block = (
        "Outline of the whole document (generated section summaries — use this to orient "
        f"yourself; it is NOT quotable evidence and has no citation number):\n{outline}\n\n"
        if outline
        else ""
    )
    messages.append(
        {
            "role": "user",
            "content": (
                f"{outline_block}Context from the document:\n{context_block}"
                f"\n\nQuestion: {question}"
            ),
        }
    )
    return messages


def _sources_from(question: str, contexts: list[Context], cited: list[int]) -> list[Source]:
    """One Source per *cited* passage, quoting the sentence that supports the answer.

    `cited` holds the context indices the model marked. When it cites nothing (or the LLM
    was unavailable) we fall back to the retrieval order, so an answer is never left
    unevidenced — but the quote is still a real sentence rather than a truncated chunk.
    """
    chosen = cited or list(range(len(contexts)))
    out: list[Source] = []
    for i in chosen:
        c = contexts[i]
        # Sliced from the chunk (so it cannot be generated), then located in the PDF
        # itself: the text comes back in the file's own characters and the page number is
        # verified — and corrected — against where the span actually sits.
        quote, page = verbatim.locate(citations.evidence_quote(question, c), c.page)
        out.append(Source(page=page, quote=quote, score=round(c.score, 4)))
    return out


def _save(response: QueryResponse, when: datetime) -> None:
    """Write the answer to data/out/q_<id>_level_<level>_<datetime>.json."""
    out_dir = Path(get_settings().out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = when.strftime("%Y%m%d-%H%M%S")
    name = f"q_{response.question_id}_level_{response.level}_{stamp}.json"
    (out_dir / name).write_text(response.model_dump_json(indent=2), encoding="utf-8")


def answer(req: QueryRequest) -> QueryResponse:
    settings = get_settings()
    client = get_client()
    top_k = req.top_k or settings.top_k

    # The system owns the ids. Level-N questions share one conversation, so level-2
    # follow-ups automatically see the earlier turns of the same level.
    question_id = "q" + uuid.uuid4().hex[:6]
    conversation_id = f"level-{req.level}"

    history = memory.get_history(conversation_id)
    now = datetime.now(UTC)
    started = time.perf_counter()

    # A whole-document question needs more evidence in front of it than a lookup does,
    # and the sub-queries have supplied candidates from more of the paper to fill it.
    if req.level >= 3 and not req.top_k:
        top_k = max(top_k, settings.top_k_level3)

    retrieval = retrieve(req.question, top_k, history, level=req.level)
    contexts = retrieval.contexts
    outline = sections.outline() if req.level >= 3 else ""
    messages = _build_messages(req.question, contexts, history, outline)

    try:
        answer_text = client.chat(messages)
    except LLMError as e:
        # Degrade rather than 500: return the retrieved evidence so the pipeline is
        # still usable (and debuggable) without a running LLM.
        answer_text = (
            f"[LLM unavailable: {e}] Retrieved context is attached as sources; no generated answer."
        )

    # The model cites the prompt's numbering; keep only what it used and renumber so
    # "(1)" in the answer points at sources[0].
    cited = citations.parse_markers(answer_text, len(contexts))
    answer_text = citations.renumber(answer_text, cited)

    memory.append(conversation_id, req.question, answer_text)
    latency_ms = int((time.perf_counter() - started) * 1000)

    response = QueryResponse(
        question_id=question_id,
        level=req.level,
        question=req.question,
        answer=answer_text,
        conversation_id=conversation_id,
        # Evidence is picked against the resolved query for the same reason retrieval used
        # it: "why does that happen?" cannot tell one sentence of a chunk from another.
        sources=_sources_from(retrieval.query, contexts, cited),
        diagnostics=Diagnostics(
            provider=settings.llm_provider,
            chat_model=settings.chat_model,
            embedding_model=settings.embedding_model,
            retrieved_chunks=len(contexts),
            retrieval_query=retrieval.query,
            sub_queries=retrieval.sub_queries,
            retrieval_steps=retrieval.steps,
            tokens=None,  # TODO: report real token usage if your provider returns it
            latency_ms=latency_ms,
            timestamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    )
    _save(response, now)
    return response
