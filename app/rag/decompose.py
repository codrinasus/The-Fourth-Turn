"""Split a whole-document question into the sub-questions it is really made of.

A Level-3 question is several questions wearing a trenchcoat:

    "The introduction warns that over-parameterized models learn shortcuts. Tracing this
     through feature attribution and probing, what token-level behavior exemplifies this,
     and what vulnerability does it create?"

Embedded whole, that is one long vector that is a poor match for every passage and a good
match for none — the topics it mixes (shortcut learning, attention weights, probing,
adversarial attacks) live on pages 1, 8 and 17 and share almost no vocabulary. One query
cannot rank all three highly, and `top_k` is spent on whichever hop happens to dominate.

So the model splits it into standalone sub-questions, each retrieved on its own, and the
result lists are fused. Every hop gets its own shot at the index instead of competing for
one ranking. `retrieve` then reranks the union against the original question, so breadth
comes from the sub-queries while the final ordering still answers what was asked.

Kept honest by construction: the decomposition sees only the question text — never the
document, never an answer — so it cannot smuggle in knowledge of these nine questions. If
it fails or returns nothing usable, retrieval carries on with the single original query.
"""

from __future__ import annotations

import logging
import re

from ..llm.base import LLMError, Message
from ..llm.factory import get_client

log = logging.getLogger(__name__)

_MAX_SUB_QUERIES = 4
_MIN_SUB_QUERY_CHARS = 12
_MAX_SUB_QUERY_CHARS = 200

_SYSTEM = (
    "You break a complex question about a single document into the smaller search "
    "questions needed to answer it.\n"
    "Rules:\n"
    "- Write between 2 and 4 sub-questions, one per line.\n"
    "- Each must stand on its own: no pronouns, no 'the above', no numbering.\n"
    "- Together they must cover every part of the original question.\n"
    "- Do not answer anything. Do not add commentary.\n"
    "- If the question is already simple, reply with the question itself on one line."
)

# Models like to prefix list items even when told not to.
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _clean(reply: str, original: str) -> list[str]:
    out: list[str] = []
    for raw in reply.splitlines():
        line = _BULLET.sub("", raw).strip().strip('"').strip()
        if not (_MIN_SUB_QUERY_CHARS <= len(line) <= _MAX_SUB_QUERY_CHARS):
            continue
        if line.lower() == original.strip().lower():
            continue
        if line not in out:
            out.append(line)
    return out[:_MAX_SUB_QUERIES]


def sub_queries(question: str) -> list[str]:
    """Standalone sub-questions covering `question`, or `[]` to retrieve it as-is."""
    messages: list[Message] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nSub-questions:"},
    ]
    try:
        parts = _clean(get_client().chat(messages, thinking=False), question)
    except LLMError as e:
        log.warning("decomposition unavailable (%s) — retrieving the question as-is", e)
        return []

    # One sub-question is no decomposition; fall back rather than replace a rich original
    # question with a lossy one-line paraphrase of it.
    if len(parts) < 2:
        return []
    log.info("decomposed %r into %s", question, parts)
    return parts
