"""Retrieve, look at what came back, decide whether it is enough, search again.

Level 3 asks questions no single passage answers, and one-shot retrieval has no way of
knowing whether it found all the parts. Decomposition helps — each hop gets its own search
instead of competing for one ranking — but the sub-questions are written *before* anything
has been retrieved, from the question alone. If a hop comes back empty, or if reading the
evidence reveals a gap the question did not spell out, a single pass has no mechanism to
notice or to react.

So this is a small ReAct-style loop:

    thought      what do I still need?         (the model reads the evidence so far)
    action       search for exactly that       (its own follow-up queries)
    observation  the new passages              (fused into the pool)
    ...          until it says "enough", or the step budget runs out

Three things keep it honest and bounded:

- **The model judges evidence, not answers.** It is asked whether the passages *contain*
  what is needed and what is missing — never whether it likes its own answer, which is a
  question models are bad at and which would invite it to keep going until it agreed with
  itself. Its only lever is proposing search queries.
- **It cannot lose evidence.** Every round's candidates are added to a pool that is fused
  and reranked once at the end. A bad follow-up query wastes a round; it cannot displace a
  passage an earlier round found.
- **It always terminates.** A hard step budget, a stop when it proposes nothing new, and
  any LLM failure ends the loop with whatever has been gathered. The pipeline degrades to
  ordinary multi-query retrieval rather than failing.

The trace is reported in `diagnostics.retrieval_steps`, so what the loop did on a given
question is visible in the graded response instead of only in the logs.
"""

from __future__ import annotations

import logging
import re

from ..llm.base import LLMError, Message
from ..llm.factory import get_client

_BOOLEAN = re.compile(r"\b(AND|OR|NOT)\b")
_SYNTAX = re.compile(r"[\"'()\[\]{}~^*]|\bTITLE:|\bTEXT:", re.IGNORECASE)

log = logging.getLogger(__name__)

_MAX_QUERIES_PER_STEP = 3
_MIN_QUERY_CHARS = 12
_MAX_QUERY_CHARS = 200
_EVIDENCE_BUDGET = 700  # chars shown per passage when judging sufficiency
_MAX_EVIDENCE = 12  # passages shown to the judge

_SYSTEM = (
    "You are checking whether a set of passages from one document contains everything "
    "needed to answer a question. You are NOT answering the question.\n"
    "Reply in exactly this format, nothing else:\n"
    "VERDICT: ENOUGH\n"
    "or\n"
    "VERDICT: MISSING\n"
    "MISSING: <what the passages do not cover, one line>\n"
    "SEARCH: <a search query> | <another search query>\n"
    "Rules:\n"
    "- Answer ENOUGH if every part of the question is covered by some passage.\n"
    "- Answer MISSING only if a specific part is genuinely absent, and then give at most "
    "three search queries aimed at exactly that part.\n"
    "- Each search query must stand on its own: no pronouns, no 'the above'.\n"
    "- Write each query as a plain natural-language question, the way a person would ask "
    "it. This is a semantic search over prose, NOT a keyword engine: never use AND, OR, "
    "quotation marks, parentheses or field syntax — they are matched as literal text and "
    "make the query worse.\n"
    "- Do not ask for passages that would merely confirm what is already there."
)


def _plain(query: str) -> str:
    """Strip search-engine syntax out of a proposed query.

    Told to write a "search query", models reach for Lucene — we saw
    `survey authors argue" AND ("fidelity" OR "faithfulness")`. Both of our arms are the
    wrong audience for that: the embedder encodes the operators as words, and BM25 matches
    `AND` and the stray quotes as literal tokens. The prompt asks for prose; this makes
    sure of it, because a prompt rule is a request and this is a guarantee.
    """
    cleaned = _BOOLEAN.sub(" ", query)
    cleaned = _SYNTAX.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _evidence_block(passages: list[str]) -> str:
    return "\n\n".join(
        f"({i}) {text[:_EVIDENCE_BUDGET]}"
        for i, text in enumerate(passages[:_MAX_EVIDENCE], start=1)
    )


def _parse(reply: str) -> tuple[bool, str, list[str]]:
    """`(is_enough, what_is_missing, follow_up_queries)`.

    Anything unparseable counts as ENOUGH: a judge we cannot read is not a reason to keep
    spending retrieval rounds, and the pool already in hand is what the answer is built
    from either way.
    """
    verdict_missing = False
    missing = ""
    queries: list[str] = []

    for raw in reply.splitlines():
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("VERDICT:"):
            verdict_missing = "MISSING" in upper
        elif upper.startswith("MISSING:"):
            missing = line.split(":", 1)[1].strip()
        elif upper.startswith("SEARCH:"):
            for part in line.split(":", 1)[1].split("|"):
                q = _plain(part)
                if _MIN_QUERY_CHARS <= len(q) <= _MAX_QUERY_CHARS and q not in queries:
                    queries.append(q)

    if not verdict_missing or not queries:
        return True, "", []
    return False, missing, queries[:_MAX_QUERIES_PER_STEP]


def next_queries(
    question: str,
    passages: list[str],
    already_searched: list[str],
) -> tuple[bool, str, list[str]]:
    """Judge the evidence and propose follow-up searches.

    Returns `(is_enough, missing, queries)`. Queries already issued are filtered out, so a
    judge that keeps asking for the same thing ends the loop instead of spinning.
    """
    if not passages:
        return True, "", []

    messages: list[Message] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Passages retrieved so far:\n{_evidence_block(passages)}\n\n"
                "Is this enough?"
            ),
        },
    ]

    try:
        # Thinking off: this is a checklist against the passages, and the loop's whole
        # value depends on a round costing seconds rather than the price of an answer.
        enough, missing, queries = _parse(get_client().chat(messages, thinking=False))
    except LLMError as e:
        log.warning("retrieval judge unavailable (%s) — stopping with the evidence in hand", e)
        return True, "", []

    seen = {q.lower() for q in already_searched}
    fresh = [q for q in queries if q.lower() not in seen]
    if enough or not fresh:
        return True, missing, []
    return False, missing, fresh
