# Ablation baselines

Real `POST /query` responses captured with one component switched off, so the before/after
claims in [`../../TECHNICAL_NOTE.md`](../../TECHNICAL_NOTE.md) §4 can be checked rather than
taken on trust. Nothing here is hand-edited: each file is a response the API actually
returned, and each carries the `diagnostics` that show which configuration produced it.

| Directory | Switch | Compare against |
|---|---|---|
| `dropout-no-rewrite/` | `REWRITE_ENABLED=false` — a level-2 follow-up is embedded exactly as typed | `submission/level-2/` |
| `dropout-no-agent/` | `AGENT_ENABLED=false` — decomposition only, no retrieve-judge-search loop | `submission/level-3/` |

Both were produced against the committed document (`data/in/srivastava14a.pdf`), the same
151 chunks, the same models and the same `TOP_K` as the shipped answers. Reproduce either
with:

```bash
./scripts/set_flag.sh REWRITE_ENABLED false     # verifies the container actually took it
uv run python scripts/run_questions.py 2 --out docs/ablations/scratch
./scripts/set_flag.sh REWRITE_ENABLED true
```

`--out` matters: an ablation is *meant* to produce worse answers, and writing them into
`submission/` is one forgotten restore away from shipping a crippled baseline as the graded
answer.

## How to read them

The clearest single result is q5. Asked *"Why does that happen?"* with rewriting off, the
follow-up carries nothing retrievable, and the answer says so outright before answering a
different question it can support:

> "The provided context does not discuss increased training time or noisy gradients."

With the follow-up resolved into a standalone query first, the same question retrieves page
24 at **1.00** and answers what was asked.

The Level-3 rows are weaker evidence and the note says so: q8's chunk scores sit near zero
in both columns because its answer is built from the section index rather than any passage,
and its best source has swung between 0.01 and 0.51 across repeated runs of an identical
configuration. Single-run Level-3 comparisons carry that much noise; the q5 result does not.

Earlier baselines for a previous document were removed when the document changed — a
baseline whose PDF is no longer in the repository cannot be verified by anyone, and leaving
it here implied a comparison with the current answers that was not valid.
