"""Embeddings: turning text into vectors.

By default we embed through the configured provider. For this team, that is
Ollama with `nomic-embed-text`, which keeps the FastAPI image light. The choice
of model, how you build the text you embed, and how you chunk it are all yours
to tune.

`get_embedder()` returns an object with `embed(texts, is_query=False)`. Two backends:
  - "provider" (default): reuse the chat provider's embed() (Ollama / LM Studio / litellm).
  - "sentence-transformers": optional Python backend; install the package before using it.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings


class SentenceTransformerEmbedder:
    """Optional local embeddings via sentence-transformers."""

    def __init__(self, model_name: str):
        # Imported lazily so the app image stays light when provider embeddings are used.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "EMBEDDING_BACKEND=sentence-transformers requires installing "
                "sentence-transformers. The lightweight Docker image uses "
                "EMBEDDING_BACKEND=provider with Ollama."
            ) from e

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._is_e5 = "e5" in model_name.lower()

    def _prefix(self, text: str, is_query: bool) -> str:
        if not self._is_e5:
            return text
        return f"{'query' if is_query else 'passage'}: {text}"

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        prepared = [self._prefix(t, is_query) for t in texts]
        vecs = self._model.encode(prepared, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


class ProviderEmbedder:
    """Embeddings from the configured provider."""

    def __init__(self):
        from ..llm.factory import get_client

        self._client = get_client()

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        # Most provider embedding APIs do not distinguish query vs passage.
        return self._client.embed(texts)


@lru_cache
def get_embedder():
    s = get_settings()
    backend = s.embedding_backend.lower()
    if backend in ("sentence-transformers", "sentence_transformers", "st"):
        return SentenceTransformerEmbedder(s.embedding_model)
    if backend == "provider":
        return ProviderEmbedder()
    raise ValueError(
        f"unknown embedding_backend: {s.embedding_backend!r} "
        "(expected 'provider' or 'sentence-transformers')"
    )
