"""The main endpoint: answer a question about the document."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..models import QueryRequest, QueryResponse
from ..rag.pipeline import answer

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Ask a question about the ingested document.

    Send just the **question** and its **level** (1, 2 or 3). The system assigns an id and
    threads the conversation for you — you do not pass a conversation id.

    - **Level 1 — retrieval.** A standalone question: hybrid dense + BM25, fused and
      reranked by a cross-encoder.
    - **Level 2 — memory.** A follow-up ("and how large is it?"). All level-2 questions
      share one conversation, and the follow-up is resolved into a standalone query
      *before* retrieval — `diagnostics.retrieval_query` reports what was searched.
    - **Level 3 — whole-document.** The question is decomposed into sub-questions retrieved
      separately (`diagnostics.sub_queries`), with the section outline as a document map.

    The full response is returned **and** written to `data/out/` as a JSON file. To submit,
    copy your preferred answers from `data/out/` into `submission/`. See `submission/README.md`.
    """
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    try:
        return answer(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query failed: {e}") from e
