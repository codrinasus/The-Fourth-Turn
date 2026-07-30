"""Turn the latest turn of a conversation into a search query.

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

So whenever there is a conversation, the model reads it and writes the query. There is no
heuristic deciding *whether* a question needs resolving: an earlier version of this module
gated on grammar (dangling pronoun, continuation opener, question length) and only called
the model when the gate fired. That gate was wrong in both directions — it rewrote
"What is **this** survey about?", which needs nothing, and it would sail past "What about
legal search?" if phrased with enough words. Deciding what a question refers to *is* a
language-understanding problem, so it belongs to the language model, not to a regex. The
model is told to return a self-contained question unchanged, which is the same judgement
the gate was trying to make, made by something equipped to make it.

The rewrite is treated as untrusted output: it is validated (single line, bounded length,
non-empty) and on any failure we fall back to *previous question + current question*
concatenated, which is a strictly better retrieval query than a bare follow-up and needs
no model at all.
"""

from __future__ import annotations

import logging

from ..llm.base import LLMError, Message
from ..llm.factory import get_client

log = logging.getLogger(__name__)

_MAX_QUERY_CHARS = 400  # a rewrite longer than this is prose, not a query
_HISTORY_TURNS = 3  # recent turns shown to the rewriter
_ANSWER_BUDGET = 600  # chars of each past answer kept as antecedent material

_SYSTEM = (
    "You write the search query for the last question in a conversation, so that it can be "
    "used to search a document with no conversation attached.\n"
    "Rules:\n"
    "- Replace every pronoun and every reference to an earlier turn with the thing it "
    "refers to. 'Why does that happen?' must name what 'that' is.\n"
    "- Resolve an ambiguous reference to ONE thing — the main subject of the previous "
    "answer. Never expand it into a list of everything that was mentioned.\n"
    "- If the question already stands on its own, repeat it EXACTLY as written. Do not "
    "append background, qualifiers or facts from earlier answers to it.\n"
    "- Ask about one thing. The result must be a single, short question.\n"
    "- Do not answer the question. Do not add facts that are not in the conversation.\n"
    "- Reply with the question only: one line, no preamble, no quotes, no explanation."
)


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

    The first turn of a conversation has nothing to resolve against and is returned as
    asked. After that the model always gets a say. Never raises: every failure path
    degrades to `_fallback`.
    """
    if not history:
        return question

    conversation = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in _recent(history)
    )
    messages: list[Message] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{conversation}\n\n"
                f"Question to turn into a search query: {question}\n\n"
                "Search query:"
            ),
        },
    ]

    try:
        # No reasoning here: substituting an antecedent is mechanical, and thinking on a
        # local model costs more latency than the rewrite itself.
        rewritten = _clean(get_client().chat(messages, thinking=False))
    except LLMError as e:
        log.warning("query rewrite unavailable (%s) — falling back to history concatenation", e)
        return _fallback(question, history)

    if rewritten is None:
        log.warning("query rewrite returned nothing usable — falling back to concatenation")
        return _fallback(question, history)

    if rewritten == question:
        log.info("query kept as asked: %r", question)
    else:
        log.info("rewrote %r -> %r", question, rewritten)
    return rewritten
