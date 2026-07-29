"""Load the corpus PDF into the vector store."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..models import IngestRequest, IngestResponse
from ..rag.ingest import ingest

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(req: IngestRequest) -> IngestResponse:
    """Parse, embed and index the PDF sitting in `data/in/`.

    Run this once before querying (and again after you change chunking or embeddings).
    It indexes page-grounded chunks and dumps the same chunks to `data/chunks` for
    inspection, BM25, and future reranking. Pass `reset: true` to rebuild the
    collection from scratch, and `filename` to pick a specific PDF in `data/in/`.
    """
    try:
        return ingest(filename=req.filename, reset=req.reset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}") from e
