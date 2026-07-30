"""Embeddings: turning text into vectors.

We embed through the configured chat provider — for this team, Ollama with `bge-m3`
(1024-dim, multilingual, 8192-token context). One inference stack for both chat and
vectors keeps the FastAPI image light: no sentence-transformers, no model weights baked
into the container. The choice of model and how you build the text you embed are yours
to tune.

`get_embedder()` returns an object with `embed(texts, is_query=False)`. The only backend
is "provider", which reuses the chat provider's `embed()` (Ollama / LM Studio / litellm).
"""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings


class ProviderEmbedder:
    """Embeddings from the configured provider."""

    def __init__(self):
        from ..llm.factory import get_client

        self._client = get_client()

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        # bge-m3 is symmetric — it wants no query:/passage: prefix, so is_query is
        # deliberately unused here. Add prefixing if you switch to an asymmetric model
        # (nomic-embed-text wants search_query:/search_document:, e5 wants query:/passage:).
        return self._client.embed(texts)


@lru_cache
def get_embedder():
    s = get_settings()
    backend = s.embedding_backend.lower()
    if backend == "provider":
        return ProviderEmbedder()
    raise ValueError(
        f"unknown embedding_backend: {s.embedding_backend!r} — this build only serves "
        "embeddings through the chat provider (expected 'provider'). Set EMBEDDING_MODEL "
        "to a model your provider serves, e.g. bge-m3 on Ollama."
    )
