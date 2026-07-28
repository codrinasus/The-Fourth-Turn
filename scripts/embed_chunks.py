#!/usr/bin/env python
"""Embed the files in data/chunks/ and store the vectors next to them.

    uv run python scripts/embed_chunks.py
    uv run python scripts/embed_chunks.py --query "how are explanations evaluated?"

Uses the app's own `get_embedder()` (app/rag/embeddings.py), so the vectors are produced
exactly the way `/ingest` produces them — same model, same query/passage prefixes — but
without Qdrant. That makes it a cheap way to sanity-check an embedding model before
committing to a re-ingest.

Writes:
    data/chunks/embeddings.npy    # float32 [n_chunks, dim], row i = chunks[i] in the meta
    data/chunks/embeddings.json   # model, backend, dim, and the chunk files in row order

With --query it also runs the retrieval you would get from those vectors: the query is
embedded with is_query=True and scored by cosine against every chunk. No vector store
involved — just the embeddings, so you see the model's behaviour in isolation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHUNKS_DEFAULT = Path("data/chunks")


def load_chunks(chunks_dir: Path) -> tuple[list[dict], list[str]]:
    index_path = chunks_dir / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"no {index_path} — run scripts/dump_chunks.py first")
    meta = json.loads(index_path.read_text(encoding="utf-8"))["chunks"]
    texts = [(chunks_dir / m["file"]).read_text(encoding="utf-8").rstrip("\n") for m in meta]
    return meta, texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunks", type=Path, default=CHUNKS_DEFAULT)
    ap.add_argument("--batch", type=int, default=8, help="texts per encode call")
    ap.add_argument("--query", default=None, help="probe the vectors with a question")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    meta, texts = load_chunks(args.chunks)

    from app.config import get_settings
    from app.rag.embeddings import get_embedder

    settings = get_settings()
    embedder = get_embedder()
    print(
        f"embedding {len(texts)} chunks with {settings.embedding_model} "
        f"({settings.embedding_backend}) — first run downloads the model"
    )

    vectors: list[list[float]] = []
    for i in range(0, len(texts), args.batch):
        vectors.extend(embedder.embed(texts[i : i + args.batch], is_query=False))
        print(f"  {min(i + args.batch, len(texts))}/{len(texts)}", end="\r", flush=True)

    mat = np.asarray(vectors, dtype=np.float32)
    print(f"\nvectors: {mat.shape[0]} x {mat.shape[1]}")

    norms = np.linalg.norm(mat, axis=1)
    print(f"  L2 norm: min {norms.min():.4f}  max {norms.max():.4f} (1.0 = normalised)")

    np.save(args.chunks / "embeddings.npy", mat)
    (args.chunks / "embeddings.json").write_text(
        json.dumps(
            {
                "model": settings.embedding_model,
                "backend": settings.embedding_backend,
                "dim": int(mat.shape[1]),
                "count": int(mat.shape[0]),
                "chunks": [
                    {"index": m["index"], "page": m["page"], "file": m["file"]} for m in meta
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  -> {args.chunks / 'embeddings.npy'}, {args.chunks / 'embeddings.json'}")

    if args.query:
        # Vectors are already L2-normalised, so a dot product IS cosine similarity.
        qvec = np.asarray(embedder.embed([args.query], is_query=True)[0], dtype=np.float32)
        scores = mat @ qvec
        order = np.argsort(-scores)[: args.top_k]
        print(f'\ntop {args.top_k} for "{args.query}":')
        for rank, i in enumerate(order, start=1):
            preview = " ".join(texts[i].split())[:110]
            print(f"  {rank}. p{meta[i]['page']:>3} score {scores[i]:.4f}  {preview}…")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
