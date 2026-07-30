"""Ollama provider — https://ollama.com

Talks to a local Ollama server over its native REST API. Pull models first:

    ollama pull llama3.1
    ollama pull bge-m3
"""

from __future__ import annotations

import httpx

from .base import LLMError, Message, strip_thinking


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embedding_model: str,
        timeout: float,
        thinking: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.thinking = thinking

    def chat(self, messages: list[Message], *, thinking: bool | None = None) -> str:
        payload: dict = {"model": self.chat_model, "messages": messages, "stream": False}
        # Thinking stays ON — qwen3 answers better when it reasons first. We leave the
        # field off so the model's own default applies, and only ask the server to skip
        # thinking when someone turns it off explicitly. Either way strip_thinking()
        # keeps the scratchpad out of the reply we return: newer Ollama puts it in a
        # separate `message.thinking` field, older versions inline it in the content.
        if not (self.thinking if thinking is None else thinking):
            payload["think"] = False
        try:
            resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            if "think" in payload and resp.status_code == 400 and "think" in resp.text.lower():
                # Older Ollama, or a model with no thinking mode: retry without the flag.
                payload.pop("think", None)
                resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return strip_thinking(resp.json()["message"]["content"])
        except Exception as e:
            raise LLMError(f"ollama chat failed: {e}") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Vectors for `texts`, in order.

        Newer Ollama has a batched `/api/embed` that takes the whole list in one request;
        the older `/api/embeddings` takes one prompt per call. Ingest embeds 269 chunks,
        so the difference is one HTTP round trip against 269 of them. We try the batched
        endpoint and fall back on a 404 from an older server, which keeps the image
        working against whatever Ollama the user happens to be running.
        """
        if not texts:
            return []
        try:
            resp = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embedding_model, "input": texts},
                timeout=self.timeout,
            )
            if resp.status_code != 404:
                resp.raise_for_status()
                return resp.json()["embeddings"]
        except httpx.HTTPError as e:
            raise LLMError(f"ollama embed failed: {e}") from e

        return self._embed_one_by_one(texts)

    def _embed_one_by_one(self, texts: list[str]) -> list[list[float]]:
        """The pre-/api/embed path, kept so an older Ollama still works."""
        out: list[list[float]] = []
        try:
            for text in texts:
                resp = httpx.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": text},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                out.append(resp.json()["embedding"])
            return out
        except Exception as e:
            raise LLMError(f"ollama embed failed: {e}") from e
