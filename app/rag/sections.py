"""A second index, one entry per section — the Level-3 document map.

Chunk retrieval is myopic by construction. A chunk is ~800 characters, the prompt holds
five of them, and a question like *"summarise the contribution of each section"* or
*"what is the most common way explanations are evaluated across the survey"* is not
answerable from any five passages: the evidence is spread over 35 pages and the shape of
the answer is the shape of the document. Retrieving harder does not fix it, because there
is no passage to find.

So at ingest we build a **second, coarser index**: the chunker already tags every chunk
with the section breadcrumb it came from (`3 FEATURE ATTRIBUTION > 3.2 Model-introspective
Feature Attribution`), so grouping by that gives real section boundaries for free — no
heuristics, they are Docling's own `section_header` blocks. Each section is summarised
once by the chat model and indexed under its own embedding.

Two things it buys:

- **Navigation.** A whole-document question can search sections instead of sentences, and
  find *where* in the paper to look before looking.
- **A map in the prompt.** For Level 3 the outline of the document goes into the prompt
  alongside the retrieved passages, so the model can reason about coverage and structure
  rather than guessing from five fragments.

**Integrity boundary — this matters.** Section summaries are *generated text*. They are
never quoted and never become a `Source`. Every `Source.quote` still comes from the
verbatim chunk index, sliced out of page text. The outline is labelled as a generated
navigation aid in the prompt and carries no citation number, so the model cannot cite it.
Summaries help the system decide *where to look*; only the document itself is evidence.
"""

from __future__ import annotations

import logging
import uuid

from qdrant_client import models

from ..config import get_settings
from ..llm.base import LLMError, Message
from ..llm.factory import get_client
from ..vectorstore.qdrant_store import get_store
from .chunking import Chunk
from .embeddings import get_embedder

log = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("6f0d9b1e-3b7a-4c2e-9a1d-000000000001")

# Back matter has no "contribution" to summarise and would waste ~85 LLM calls.
_SKIP_SECTIONS = {"REFERENCES", "ACKNOWLEDGMENTS"}
_SKIP_KINDS = {"reference"}

_MAX_SECTION_CHARS = 6000  # what the summariser sees of a long section
_SUMMARY_SYSTEM = (
    "You summarise one section of a research paper for a table of contents.\n"
    "Write 1-3 sentences saying what this section covers and what it contributes.\n"
    "Be concrete: name the methods, models or categories it introduces.\n"
    "Do not add anything that is not in the text. Reply with the summary only."
)


def collection_name() -> str:
    return f"{get_settings().qdrant_collection}_sections"


def _grouped(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Chunks by section breadcrumb, in first-appearance (reading) order."""
    groups: dict[str, list[Chunk]] = {}
    for c in chunks:
        section = (c.section or "").strip()
        if not section or c.kind in _SKIP_KINDS:
            continue
        if section.split(">")[0].strip().upper() in _SKIP_SECTIONS:
            continue
        groups.setdefault(section, []).append(c)
    return groups


def _summarise(section: str, body: str) -> str:
    """One summary, or a graceful degradation to the section's opening text.

    The fallback is deliberately still *useful*: the first sentences of a section describe
    it reasonably well, so a dead LLM costs summary quality, not the index itself.
    """
    messages: list[Message] = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {"role": "user", "content": f"Section: {section}\n\n{body[:_MAX_SECTION_CHARS]}"},
    ]
    try:
        # Mechanical work — reasoning here would multiply ingest time for no gain.
        summary = get_client().chat(messages, thinking=False).strip()
    except LLMError as e:
        log.warning("section summary failed for %r (%s) — using opening text", section, e)
        return body[:400].strip()
    return summary or body[:400].strip()


def build_index(chunks: list[Chunk], source: str, reset: bool = False) -> int:
    """Summarise every section and index it. Returns how many sections were written."""
    groups = _grouped(chunks)
    if not groups:
        log.warning("no sections found — skipping the section index")
        return 0

    embedder = get_embedder()
    store = get_store(collection_name())

    summaries: list[dict] = []
    for section, members in groups.items():
        body = "\n\n".join(c.text for c in members)
        pages = sorted({c.page for c in members})
        summaries.append(
            {
                "section": section,
                "heading_path": members[0].heading_path,
                "summary": _summarise(section, body),
                "page_start": pages[0],
                "page_end": pages[-1],
                "chunks": len(members),
                "order": members[0].index,  # reading order, for rendering the outline
                "source": source,
            }
        )
    log.info("summarised %d sections", len(summaries))

    # The heading is part of what we embed: a summary that never repeats the section's
    # own title would otherwise be unfindable by that title.
    vectors = embedder.embed([f"{s['section']}\n{s['summary']}" for s in summaries], is_query=False)
    store.ensure_collection(dim=len(vectors[0]), reset=reset or store.exists())

    store.upsert(
        [
            models.PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, f"{source}:{s['section']}")),
                vector=vec,
                payload=s,
            )
            for s, vec in zip(summaries, vectors)
        ]
    )
    return len(summaries)


def outline() -> str:
    """The whole document as `section — summary` lines, in reading order.

    Small enough to always carry for a Level-3 question (this paper: 44 sections, ~10k
    characters) and complete, which matters for "each section" questions where retrieving
    only the most similar sections would silently drop the rest.
    """
    records = get_store(collection_name()).scroll_all()
    if not records:
        return ""
    rows = sorted((r.payload or {} for r in records), key=lambda p: p.get("order", 0))
    return "\n".join(
        f"- {p['section']} (pages {p['page_start']}-{p['page_end']}): {p['summary']}"
        for p in rows
        if p.get("section")
    )


def search(query: str, top_k: int) -> list[dict]:
    """The sections most related to `query`, as payload dicts."""
    store = get_store(collection_name())
    if not store.exists():
        return []
    vector = get_embedder().embed([query], is_query=True)[0]
    return [dict(h.payload or {}, score=h.score) for h in store.search(vector, top_k)]
