"""The contract every provider implements.

A client is anything with `chat()` and `embed()`. Keeping this narrow means the
rest of the app (`app/rag/`) never cares which backend is running.
"""

from __future__ import annotations

import re
from typing import Protocol, TypedDict

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove reasoning-model scratchpads from a reply.

    Thinking models (qwen3, deepseek-r1, …) wrap their reasoning in <think>…</think>.
    That text must never reach `QueryResponse.answer` — it is the graded field — nor any
    JSON we try to parse out of a reply, so providers call this before returning.

    Three shapes to survive: a complete block; a stray closing tag (the reply started
    mid-thought, so the answer is whatever follows the last one); a stray opening tag
    (generation was cut off, so keep the text and just drop the markup).
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if _THINK_CLOSE.search(cleaned):
        cleaned = _THINK_CLOSE.split(cleaned)[-1]
    cleaned = _THINK_OPEN.sub("", cleaned)
    return cleaned.strip()


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatModel(Protocol):
    def chat(self, messages: list[Message]) -> str:
        """Return the assistant's reply to a list of chat messages."""
        ...


class EmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""
        ...


class LLMError(RuntimeError):
    """Raised when a provider call fails. Callers decide whether to degrade or surface it."""
