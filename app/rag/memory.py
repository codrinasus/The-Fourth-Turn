"""Conversation memory — where Level 2 lives.

A Level-2 question is a follow-up: *"and how large is its test split?"* only makes
sense given the earlier turn it refers to. To answer it you need the history of the
conversation it belongs to.

The baseline stores history in a process-local dict, which is enough to demonstrate
the idea but forgets everything on restart and does not scale past one worker.
"""

from __future__ import annotations

from ..llm.base import Message

# conversation_id -> ordered list of messages
_STORE: dict[str, list[Message]] = {}


def get_history(conversation_id: str | None) -> list[Message]:
    if not conversation_id:
        return []
    return list(_STORE.get(conversation_id, []))


def append(conversation_id: str | None, user: str, assistant: str) -> None:
    if not conversation_id:
        return
    history = _STORE.setdefault(conversation_id, [])
    history.append({"role": "user", "content": user})
    history.append({"role": "assistant", "content": assistant})


def reset(conversation_id: str) -> None:
    _STORE.pop(conversation_id, None)


# TODO(level-2): history alone is not enough. The retrieval step still embeds the
#   raw follow-up ("and the test split?"), which has no searchable content. The high-
#   leverage fix is to REWRITE the question into a standalone query using this history
#   BEFORE retrieving — see rag/retrieve.py::rewrite_query.
# TODO(level-2): a persistent, shared store (Redis, Postgres, ...) if you run more
#   than one worker or want memory to survive a restart.
