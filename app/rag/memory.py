"""Conversation memory — where Level 2 lives.

A Level-2 question is a follow-up: *"and how large is its test split?"* only makes
sense given the earlier turn it refers to. To answer it you need the history of the
conversation it belongs to.

History is stored in a process-local dict. That is honestly a weakness: it forgets
everything on restart and does not scale past one worker. It is enough for the graded
flow, where a level's three questions arrive in order against one process, and replacing
it with Redis would not change how Level 2 actually works — the interesting part is
`rag/rewrite.py`, which turns this history into a retrievable query.
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


# History alone is not enough, and that gap is now closed: the retriever never sees this
# store, so a raw follow-up ("and the test split?") would be embedded with no searchable
# content in it. `rag/rewrite.py` resolves the question against this history BEFORE
# retrieval — see rag/retrieve.py::rewrite_query.
#
# TODO(level-2): a persistent, shared store (Redis, Postgres, ...). This dict dies with the
#   process and does not work across workers — the main remaining weakness at this level.
