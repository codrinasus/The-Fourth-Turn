"""Request and response schemas.

The `QueryResponse` below IS the format we grade. Every `POST /query` call also writes
its response to `data/out/` as a JSON file. To submit, copy the content of your preferred
answers from `data/out/` into the matching files under `submission/` and push. See
`submission/README.md`. You never have to hand-write this JSON — it comes straight out of
the endpoint (visible in Swagger at `/docs` and in the Postman collection).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    """One piece of evidence supporting an answer.

    `quote` must be text that actually appears in the PDF on `page` — it is how we (and
    you) check the answer is grounded in the document rather than invented.
    """

    page: int = Field(..., description="1-indexed PDF page the quote is on", examples=[14])
    quote: str = Field(
        ..., description="verbatim span from the PDF that supports the answer",
        examples=["we evaluate on the MS MARCO passage ranking collection"],
    )
    score: float | None = Field(None, description="retrieval score, if you have one", examples=[0.83])


class Diagnostics(BaseModel):
    """Self-reported context for the answer. Not graded for correctness — but you should
    be able to explain these numbers in your technical note and defence."""

    provider: str = Field(..., description="chat provider used", examples=["lmstudio"])
    chat_model: str = Field(..., description="the model that wrote the answer", examples=["google/gemma-4-e2b"])
    embedding_model: str = Field(..., description="the model that produced the vectors", examples=["intfloat/multilingual-e5-large"])
    retrieved_chunks: int = Field(..., description="how many passages were retrieved", examples=[5])
    tokens: int | None = Field(None, description="total tokens for this turn, if known", examples=[3200])
    latency_ms: int | None = Field(None, description="wall-clock time to produce the answer", examples=[1840])
    timestamp: str = Field(..., description="ISO-8601 UTC time the answer was produced", examples=["2026-07-31T09:41:12Z"])


class QueryRequest(BaseModel):
    """What you send to ask a question. Just the question and its level — the system
    handles ids and conversation threading for you."""

    question: str = Field(
        ...,
        description="The question to answer, in natural language.",
        examples=["What dataset do the authors evaluate their method on?"],
    )
    level: int = Field(
        ...,
        ge=1,
        le=3,
        description=(
            "The challenge level: 1 = retrieval (standalone question), 2 = conversational "
            "memory (a follow-up — the system threads it with earlier level-2 questions), "
            "3 = whole-document reasoning. Only 1, 2 or 3 are accepted."
        ),
        examples=[1],
    )
    top_k: int | None = Field(None, description="override the configured retrieval depth", examples=[5])


class QueryResponse(BaseModel):
    """The answer. This is the graded object; it is also saved to `data/out/`."""

    question_id: str = Field(..., description="auto-generated id for this question", examples=["q7f3a1"])
    level: int = Field(..., description="the level you asked at", examples=[1])
    question: str
    answer: str = Field(..., description="the system's answer", examples=["They evaluate on MS MARCO."])
    conversation_id: str = Field(
        ..., description="auto-assigned; level-2 follow-ups share one so memory carries across",
        examples=["level-2"],
    )
    sources: list[Source] = Field(default_factory=list, description="evidence grounding the answer")
    diagnostics: Diagnostics | None = None


class IngestRequest(BaseModel):
    filename: str | None = Field(
        None, description="a PDF under data/in/; defaults to the first *.pdf found there",
        examples=["paper.pdf"],
    )
    reset: bool = Field(False, description="drop and recreate the collection before ingesting")


class IngestResponse(BaseModel):
    document: str = Field(..., description="the file that was ingested")
    pages: int = Field(..., description="pages parsed")
    chunks: int = Field(..., description="vectors indexed after page-grounded chunking")
    collection: str = Field(..., description="the Qdrant collection they went into")
