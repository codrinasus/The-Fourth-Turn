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
    # One of: "lmstudio", "ollama", "litellm". This team uses local Ollama.
    # qwen3:8b is the default because qwen3.6 needs more RAM than this Docker setup has.
    llm_provider: str = "ollama"
    chat_model: str = "qwen3:8b"

    ollama_base_url: str = "http://localhost:11434"     # keep Ollama's default port
    lmstudio_base_url: str = "http://localhost:1234"    # keep LM Studio's default port
    litellm_api_base: str | None = None
    litellm_api_key: str | None = None

    # --- Embeddings (turning text into vectors) -----------------------------
    # Use provider embeddings so Ollama owns both chat and vectors.
    # Pull `nomic-embed-text` once before ingest.
    # IMPROVING EMBEDDINGS IS PART OF THE CHALLENGE — see app/rag/embeddings.py.
    embedding_backend: str = "provider"
    embedding_model: str = "nomic-embed-text"

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
