"""Minimal BM25 index over plain text chunk files.

Usage:
    from app.rag.bm25_index import BM25Index
    idx = BM25Index()
    idx.build_from_dir("data/chunks", glob="**/*.txt")
    hits = idx.search("your query", top_k=5)

This implementation is intentionally small and depends on `rank_bm25` for scoring.
It records source file and attempts to extract a page number from filenames like
`page-001.txt`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


def _tokenize(text: str) -> List[str]:
    tokens = re.split(r"[^0-9a-zA-Z]+", text.lower())
    return [t for t in tokens if len(t) > 1]


@dataclass
class BM25Hit:
    text: str
    section: int
    score: float
    source: str | None = None
    page: int | None = None


class BM25Index:
    def __init__(self):
        self._docs_texts: List[str] = []
        self._docs_meta: List[dict] = []
        self._tokenized: List[List[str]] = []
        self._bm25: BM25Okapi | None = None
        self._doc_freq: dict[str, int] = {}
        self._doc_len: List[int] = []
        self._avg_len: float = 0.0

    def build_from_dir(self, dirpath: str | Path, glob: str = "**/*.txt") -> int:
        p = Path(dirpath)
        if not p.exists():
            raise FileNotFoundError(f"directory not found: {p}")

        texts: List[str] = []
        metas: List[dict] = []
        for f in sorted(p.glob(glob)):
            raw = f.read_text(encoding="utf-8")
            parts = [part.strip() for part in re.split(r"\n#{1,3} ", raw) if part.strip()]
            if not parts:
                parts = [part.strip() for part in raw.split("\n\n") if part.strip()]
            for part in parts:
                texts.append(part)
                meta: dict = {"source": str(f)}
                m = re.search(r"page[-_ ]?(\d{1,4})", f.name, re.IGNORECASE)
                meta["page"] = int(m.group(1)) if m else None
                metas.append(meta)

        if not texts:
            raise ValueError("no text files found to index")

        self._docs_texts = texts
        self._docs_meta = metas

        # Tokenize and build the rank_bm25 model.
        self._tokenized = [_tokenize(t) for t in self._docs_texts]
        if BM25Okapi is not None:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            print("Warning: BM25 library not found; using fallback scoring (slower, less accurate).")
            self._build_fallback()
        return len(self._docs_texts)

    def _build_fallback(self) -> None:
        # Document frequencies and lengths for simple BM25 scoring.
        self._doc_freq = {}
        self._doc_len = [len(toks) for toks in self._tokenized]
        for toks in self._tokenized:
            seen = set()
            for token in toks:
                if token not in seen:
                    self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
                    seen.add(token)
        self._avg_len = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0

    def search(self, query: str, top_k: int = 5) -> List[BM25Hit]:
        q_tokens = _tokenize(query)
        if self._bm25 is not None:
            scores = self._bm25.get_scores(q_tokens)
        else:
            # fallback BM25 scoring
            N = len(self._tokenized)
            k1 = 1.5
            b = 0.75
            import math

            scores = [0.0] * N
            for t in q_tokens:
                df = self._doc_freq.get(t, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                for i, toks in enumerate(self._tokenized):
                    tf = toks.count(t)
                    if tf == 0:
                        continue
                    denom = tf + k1 * (1 - b + b * (len(toks) / (self._avg_len or 1)))
                    score = idf * (tf * (k1 + 1)) / denom
                    scores[i] += score
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        out: List[BM25Hit] = []
        for i, score in ranked:
            out.append(
                BM25Hit(
                    text=self._docs_texts[i],
                    section=i,
                    score=float(score),
                    source=self._docs_meta[i].get("source"),
                    page=self._docs_meta[i].get("page"),
                )
            )
        return out
