"""The RAG pipeline: chunk -> embed -> store -> retrieve -> answer.

Reading order, following a question through the system:

    rewrite.py    a level-2 follow-up becomes a standalone query (before embedding)
    decompose.py  a level-3 question becomes several sub-questions
    agent.py      level 3 reads the evidence and searches again for what is missing
    retrieve.py   dense + BM25 per query, RRF over all arms, dedup, cross-encoder rerank
    sections.py   the second index: one summary per section, the level-3 document map
    citations.py  which passages the answer used, and which span of each is the evidence
    verbatim.py   that span, re-expressed in the PDF's own characters, page verified
    pipeline.py   assembles the prompt and the graded QueryResponse

Two invariants hold across all of it: an evidence quote is only ever *sliced* from indexed
page text, never generated; and the generated section summaries are a navigation aid that
can never become a citation. See TECHNICAL_NOTE.md.
"""
