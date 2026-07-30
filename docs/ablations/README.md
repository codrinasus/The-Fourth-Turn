# Ablation baselines

Real `POST /query` responses kept from earlier states of the pipeline, so the before/after
claims in `TECHNICAL_NOTE.md` can be checked rather than taken on trust. Nothing here is
hand-edited; each file is a response the API actually returned.

| Directory | What the pipeline looked like | Compare against |
|---|---|---|
| `baseline-no-rewrite/` | `rewrite_query` still returned its input unchanged, so a level-2 follow-up was embedded as typed | `submission/level-2/` |
| `baseline-single-query/` | No decomposition and no section index: one query, one dense + one BM25 pass | `submission/level-3/` |

Both were produced against the same document, the same 269 chunks, the same models
(`qwen3.6` + `bge-m3` + `bge-reranker-v2-m3`) and the same `TOP_K`. Scores are directly
comparable because the cross-encoder is calibrated — unlike the RRF scores it replaced,
which sat in a narrow band whether a passage was relevant or not.

The clearest single number is q5. Asked *"Why does that happen?"* with no rewriting, the
best passage scored **0.054** and the answer hedged across two readings of "that":

> "If referring to the difficulty of disentangling explanations… If referring to why
> ranking scores drop when specific tokens are removed…"

With the follow-up resolved to a standalone query first, the same question retrieves at
**0.50** and answers once, from one source.
