"""Application settings.

Values come from environment variables, or from a `.env` file (copy `.env.example`
to `.env` first). Read once and cached — call `get_settings()` anywhere.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Chat provider (the model that writes answers) ----------------------
    # One of: "lmstudio", "ollama", "litellm". This team uses local Ollama + qwen3.6.
    llm_provider: str = "ollama"
    chat_model: str = "qwen3.6"

    ollama_base_url: str = "http://localhost:11434"     # keep Ollama's default port
    lmstudio_base_url: str = "http://localhost:1234"    # keep LM Studio's default port
    litellm_api_base: str | None = None
    litellm_api_key: str | None = None

    # --- Embeddings (turning text into vectors) -----------------------------
    # Out of the box we use a local sentence-transformers model so the app runs with
    # no embedding API and no setup. It downloads on first ingest (~1 GB, once).
    #   "sentence-transformers"  -> app/rag/embeddings.py, model = embedding_model below
    #   "provider"               -> reuse the chat provider's embed() (set embedding_model
    #                               to that provider's embedding model, e.g. nomic-embed-text)
    # IMPROVING EMBEDDINGS IS PART OF THE CHALLENGE — see app/rag/embeddings.py.
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "intfloat/multilingual-e5-large"

    # --- PDF parsing --------------------------------------------------------
    # We compared GROBID and Marker on the committed PDF and kept Marker as the
    # parser. Marker runs as a separate Docker job because installing it in the
    # FastAPI image would pull large ML/OCR dependencies into the serving stack.
    marker_markdown_dir: str = "data/extracted/marker"
    marker_json_dir: str = "data/extracted/marker-json"

    # --- Qdrant (vector store) ----------------------------------------------
    # Host port is deliberately unusual to avoid clashes; docker-compose overrides
    # this to the in-container hostname. The container itself still speaks 6333.
    qdrant_url: str = "http://localhost:6391"
    qdrant_collection: str = "aim_hackathon"

    # --- Retrieval ----------------------------------------------------------
    top_k: int = 5                 # how many results a query retrieves

    # Chunking is OFF by default: ingest indexes one vector per page (see
    # app/rag/chunking.py). When you implement real chunking, these are your dials.
    chunk_size: int = 800          # characters per chunk (once you chunk)
    chunk_overlap: int = 150       # characters shared between neighbours

    # --- Data folders -------------------------------------------------------
    in_dir: str = "data/in"        # put the PDF here; /ingest reads from it
    out_dir: str = "data/out"      # every /query answer is written here as JSON

    request_timeout: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
