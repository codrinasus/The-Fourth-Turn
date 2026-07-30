"""LM Studio provider — https://lmstudio.ai

LM Studio serves an OpenAI-compatible API. Start its local server (default port
1234) and load a chat model and an embedding model in the UI. The model names
below must match what LM Studio reports.
"""

from __future__ import annotations

import httpx

from .base import LLMError, Message, strip_thinking


class LMStudioClient:
    def __init__(self, base_url: str, chat_model: str, embedding_model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    def chat(self, messages: list[Message], *, thinking: bool | None = None) -> str:
        # OpenAI-shaped API: no server-side thinking switch, so the flag is accepted and
        # ignored. strip_thinking() below still removes an inline <think> block.
        try:
            resp = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json={"model": self.chat_model, "messages": messages, "temperature": 0.2},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return strip_thinking(resp.json()["choices"][0]["message"]["content"])
        except Exception as e:
            raise LLMError(f"lmstudio chat failed: {e}") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = httpx.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.embedding_model, "input": texts},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            # The API may reorder; sort by index to be safe.
            return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
        except Exception as e:
            raise LLMError(f"lmstudio embed failed: {e}") from e
