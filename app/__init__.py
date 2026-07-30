"""The Fourth Turn — a RAG-over-PDF backend. Team KrautWineSarmale, ESSIR 2026.

Answers questions about one committed PDF at three levels: retrieval, conversational
memory, and whole-document reasoning. Everything runs locally.

    app/llm/          provider abstraction (Ollama · LM Studio · litellm)
    app/vectorstore/  Qdrant wrapper — the chunk index and the section index
    app/rag/          the pipeline; start there, its __init__ lists the reading order

Design decisions, ablations and diagnosed failures: TECHNICAL_NOTE.md.
"""

__version__ = "0.1.0"
