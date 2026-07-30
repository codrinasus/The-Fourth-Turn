"""Resolve a conversational follow-up into a standalone search query.

This is the heart of Level 2, and the reason the hackathon is named after it.

Putting the conversation in the prompt is necessary but **not sufficient**. Grounding
comes from the retrieved context block, and retrieval sees only the question. When the
user asks *"Why does that happen?"*, the embedder is handed six words that contain no
retrievable content: no entity, no topic, nothing the document can match. The dense
search returns noise, BM25 returns whatever documents happen to contain "happen", and the
model is then asked to answer a good question from bad evidence. We reproduced exactly
that on our own stack — see TECHNICAL_NOTE.md — where a level-2 turn whose history
already contained the answer still replied "the context does not state", because history
never reached the retriever.

So the follow-up is resolved **before** embedding, in two stages:

1. `is_context_dependent()` — a cheap linguistic gate. A question carrying a dangling
   pronoun ("that", "it", "they"), opening with a conjunction ("And how …"), or too short
   to stand on its own cannot be searched as written. Anything else already has its own
   content words and is left completely alone, which keeps standalone questions free of
   both the latency and the risk of a rewrite. The gate tests *grammar*, not topic — it
   knows nothing about this document or these nine questions.
2. `standalone_query()` — the chat model rewrites the follow-up against the recent turns
   into one self-contained question, with the antecedents substituted in.

The rewrite is treated as untrusted output: it is validated (single line, bounded length,
non-empty, not a refusal) and on any failure we fall back to *previous question + current
question* concatenated, which is a strictly better retrieval query than the bare follow-up
and needs no model at all.
"""

from __future__ import annotations

import logging
import re

from ..llm.base import LLMError, Message
from ..llm.factory import get_client

log = logging.getLogger(__name__)

# Words that can only refer to something already said. "it"/"they"/"them" are matched as
# whole words so "item" or "therapy" do not trip the gate.
_DEPENDENT = re.compile(
    r"\b(that|those|these|this|it|its|they|them|their|he|she|his|her|"
    r"the former|the latter|the same|such)\b",
    re.IGNORECASE,
)
# A question that opens with a conjunction or a bare wh-word is continuing a thread.
_CONTINUATION = re.compile(
    r"^\s*(and|but|so|also|then|plus|what about|how about|why|why not|how come|"
    r"anything else|which one|ok|okay)\b",
    re.IGNORECASE,
)

_MIN_STANDALONE_WORDS = 6  # below this there is rarely enough content to retrieve on
_MAX_QUERY_CHARS = 400  # a rewrite longer than this is prose, not a query
_HISTORY_TURNS = 3  # recent turns shown to the rewriter
_ANSWER_BUDGET = 600  # chars of each past answer kept as antecedent material

_SYSTEM = (
    "You rewrite the last question of a conversation so that it can be understood on its "
    "own, with no conversation attached.\n"
    "Rules:\n"
    "- Replace every pronoun and reference to earlier turns with the thing it refers to.\n"
    "- Keep the user's intent exactly. Do not answer the question.\n"
    "- Do not add facts that are not in the conversation.\n"
    "- If the question already stands on its own, repeat it unchanged.\n"
    "- Reply with the rewritten question only: one line, no preamble, no quotes."
)


def is_context_dependent(question: str) -> bool:
    """True when the question cannot be retrieved on its own as written.

    Deliberately grammatical: dangling reference, continuation opener, or too few words
    to carry content. It never inspects the topic, so it behaves the same on any
    document and cannot encode knowledge of a specific question.
    """
    stripped = question.strip()
    if not stripped:
        return False
    if _CONTINUATION.match(stripped):
        return True
    if _DEPENDENT.search(stripped):
        return True
    return len(stripped.split()) < _MIN_STANDALONE_WORDS


def _recent(history: list[Message]) -> list[Message]:
    """The last few turns, with long answers clipped.

    A past answer is antecedent material — "that" usually points into it — but the whole
    thing would dominate the rewriter's prompt, so each is capped.
    """
    turns = history[-_HISTORY_TURNS * 2 :]
    return [{"role": m["role"], "content": m["content"][:_ANSWER_BUDGET]} for m in turns]


def _fallback(question: str, history: list[Message]) -> str:
    """Model-free resolution: prepend the last user question.

    Crude, but it puts the topic's content words in front of the embedder, which is the
    whole point. Used when the rewriter is unavailable or returns something unusable, so
    a dead LLM degrades retrieval instead of breaking it.
    """
    previous = [m["content"] for m in history if m["role"] == "user"]
    if not previous:
        return question
    return f"{previous[-1].strip()} {question.strip()}"


def _clean(reply: str) -> str | None:
    """Validate the rewriter's output, or None if it cannot be trusted."""
    text = reply.strip().strip('"').strip("'").strip()
    if not text:
        return None
    # Models sometimes narrate ("Sure! Here is the rewritten question: ..."); keep the
    # last non-empty line, which is where the actual query lands when they do.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    text = lines[-1].strip('"').strip("'").strip()
    if not text or len(text) > _MAX_QUERY_CHARS:
        return None
    return text


def standalone_query(question: str, history: list[Message]) -> str:
    """`question` rewritten so it can be retrieved without the conversation.

    Returns the original question untouched when there is no history or the question is
    already self-contained. Never raises: every failure path degrades to `_fallback`.
    """
    if not history or not is_context_dependent(question):
        return question

    messages: list[Message] = [{"role": "system", "content": _SYSTEM}]
    conversation = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in _recent(history)
    )
    messages.append(
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{conversation}\n\n"
                f"Question to rewrite: {question}\n\n"
                "Rewritten standalone question:"
            ),
        }
    )

    try:
        # No reasoning here: substituting an antecedent is mechanical, and thinking on a
        # local 8B model costs more latency than the rewrite itself.
        rewritten = _clean(get_client().chat(messages, thinking=False))
    except LLMError as e:
        log.warning("query rewrite unavailable (%s) — falling back to history concatenation", e)
        return _fallback(question, history)

    if rewritten is None:
        log.warning("query rewrite returned nothing usable — falling back to concatenation")
        return _fallback(question, history)

    log.info("rewrote %r -> %r", question, rewritten)
    return rewritten
