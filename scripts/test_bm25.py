"""Smoke test for the BM25 index over a directory of text chunks.

Usage:
    PYTHONPATH=. python3 scripts/test_bm25.py "query" [dir]

If `dir` is omitted the script will try `data/chunks`, then `data/pages`, then `docs`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from app.rag.bm25_index import BM25Index


def choose_dir(arg_dir: str | None) -> str:
    if arg_dir:
        return arg_dir
    for d in ("data/chunks", "data/pages", "docs"):
        if Path(d).exists():
            return d
    raise SystemExit("no candidate directory found (run /ingest to create data/chunks)")


def main():
    if len(sys.argv) < 2:
        print("usage: PYTHONPATH=. python3 scripts/test_bm25.py \"query\" [dir]")
        raise SystemExit(2)
    query = sys.argv[1]
    dirpath = choose_dir(sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"indexing files in: {dirpath}")
    idx = BM25Index()
    n = idx.build_from_dir(dirpath)
    print(f"indexed {n} sections")
    hits = idx.search(query, top_k=5)
    for h in hits:
        print(f"score={h.score:.4f} source={h.source} page={h.page} section={h.section}\n{h.text[:400]}\n---\n")


if __name__ == "__main__":
    main()
