"""Application settings.

Values come from environment variables, or from a `.env` file (copy `.env.example`
to `.env` first). Read once and cached — call `get_settings()` anywhere.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Chat provider (the model that writes answers) ----------------------
    # One of: "lmstudio", "ollama", "litellm". This team uses local Ollama.
    # qwen3.6 is a reasoning model and needs noticeably more RAM than qwen3:8b — if Docker
    # cannot give Ollama enough, fall back to CHAT_MODEL=qwen3:8b in .env.
    llm_provider: str = "ollama"
    chat_model: str = "qwen3.6"

    # Thinking is ON: qwen3 reasons before answering and the answers are better for it.
    # The <think> block itself is stripped in app/llm/base.py so it never reaches the
    # graded `answer` field. Set false only to trade quality for speed.
    ollama_thinking: bool = True

    ollama_base_url: str = "http://localhost:11434"  # keep Ollama's default port
    lmstudio_base_url: str = "http://localhost:1234"  # keep LM Studio's default port
    litellm_api_base: str | None = None
    litellm_api_key: str | None = None

    # --- Embeddings (turning text into vectors) -----------------------------
    # Use provider embeddings so Ollama owns both chat and vectors — no
    # sentence-transformers in the image. Pull `bge-m3` once before ingest.
    # bge-m3 is 1024-dim (nomic-embed-text was 768), so switching needs a re-ingest
    # with reset: true. It is a symmetric model: no query/passage prefixes.
    # IMPROVING EMBEDDINGS IS PART OF THE CHALLENGE — see app/rag/embeddings.py.
    embedding_backend: str = "provider"
    embedding_model: str = "bge-m3"

    # --- PDF parsing --------------------------------------------------------
    # Docling is active because `/ingest` can trigger parsing through a local
    # HTTP service.
    pdf_parser: str = "docling"
    docling_base_url: str = "http://localhost:5001"
    docling_output_dir: str = "data/extracted/docling"
    docling_timeout: float = 600.0
    docling_table_mode: str = "accurate"
    docling_image_export_mode: str = "referenced"
    docling_do_ocr: bool = False
    docling_use_cache: bool = True

    # --- Qdrant (vector store) ----------------------------------------------
    # Host port is deliberately unusual to avoid clashes; docker-compose overrides
    # this to the in-container hostname. The container itself still speaks 6333.
    qdrant_url: str = "http://localhost:6391"
    qdrant_collection: str = "aim_hackathon"

    # --- Reranking ----------------------------------------------------------
    # BAAI/bge-reranker-v2-m3 served by llama.cpp (docker-compose.reranker.yml). The
    # cross-encoder rescores the fused dense+BM25 pool; TOP_K survive into the prompt.
    # If the service is down, retrieval falls back to fusion order.
    reranker_enabled: bool = True
    reranker_base_url: str = "http://localhost:8792"
    reranker_timeout: float = 90.0
    rerank_candidates: int = 20  # depth of each individual dense/BM25 search
    # Ceiling on the fused union the cross-encoder rescores. One query yields ~30-39
    # unique chunks from two arms; a decomposed Level-3 question yields several times
    # that. Scoring 60 pairs costs the reranker well under a second, so the cap is here
    # to bound the worst case rather than to save time.
    max_rerank_pool: int = 60

    # --- Retrieval ----------------------------------------------------------
    top_k: int = 5  # how many results a query retrieves
    top_k_level3: int = 8  # whole-document questions need evidence from more places

    # Level 3 reflective retrieval (app/rag/agent.py): after retrieving, the model reads
    # the evidence and searches again for whatever is missing. Each step costs one short
    # LLM call plus a retrieval pass. The budget is a hard stop — the loop also ends as
    # soon as the model says the evidence is sufficient or proposes nothing new.
    agent_enabled: bool = True
    agent_max_steps: int = 2

    # The Docling pages are split into page-grounded chunks before indexing.
    chunk_size: int = 800  # target characters per chunk
    chunk_overlap: int = 0  # neighbour context will come from retrieval expansion later

    # --- Data folders -------------------------------------------------------
    in_dir: str = "data/in"  # put the PDF here; /ingest reads from it
    out_dir: str = "data/out"  # every /query answer is written here as JSON

    request_timeout: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
