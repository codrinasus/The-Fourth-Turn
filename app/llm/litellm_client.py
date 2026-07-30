"""litellm provider — one interface to ~100 hosted APIs.

https://docs.litellm.ai — OpenAI, Anthropic, Gemini, Groq, Mistral, Bedrock, ...

Model names carry the provider prefix litellm expects, e.g. "gpt-4o-mini",
"anthropic/claude-3-5-sonnet-latest", "gemini/gemini-1.5-pro". Credentials come
from LITELLM_API_KEY / LITELLM_API_BASE, or the provider's own env var
(OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).
"""

from __future__ import annotations

from .base import LLMError, Message, strip_thinking


class LiteLLMClient:
    def __init__(
        self,
        chat_model: str,
        embedding_model: str,
        api_base: str | None,
        api_key: str | None,
        timeout: float,
    ):
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.api_base = api_base or None
        self.api_key = api_key or None
        self.timeout = timeout

    def chat(self, messages: list[Message], *, thinking: bool | None = None) -> str:
        # Accepted and ignored — litellm fronts many providers with no common switch.
        import litellm  # imported lazily so local-only setups need not install it

        try:
            resp = litellm.completion(
                model=self.chat_model,
                messages=messages,
                api_base=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
                temperature=0.2,
            )
            return strip_thinking(resp["choices"][0]["message"]["content"])
        except Exception as e:
            raise LLMError(f"litellm chat failed: {e}") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        import litellm

        try:
            resp = litellm.embedding(
                model=self.embedding_model,
                input=texts,
                api_base=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            data = resp["data"]
            return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
        except Exception as e:
            raise LLMError(f"litellm embed failed: {e}") from e
