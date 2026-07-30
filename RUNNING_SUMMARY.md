# Running summary — what I did while you were out

Read this top to bottom when you get back. Newest section is at the bottom, so the story
reads in order. Anything I need **you** to decide is flagged **`→ YOUR CALL`**.

Started: Thursday 2026-07-30, evening. Deadline: **Friday 12:00**, graded from `main`.

---

## 0. Where things stood when you left

Branch `feat/reranker-and-citations`, one commit ahead of `main` (`313acf2`).

| Rubric item | Weight | State |
|---|---|---|
| Level 1 answers | 8 | Pipeline good; answers not committed |
| Level 2 answers | 10 | Pipeline could not do it — `rewrite_query` was `return question` |
| Level 3 answers | 12 | Single query, no whole-document machinery |
| Conversational memory (impl) | 10 | **0 — the no-op** |
| Whole-document reasoning (impl) | 8 | Not started |
| Technical note | 5 | File did not exist |
| Measurement & self-evaluation | 5 | Numbers existed in `TODO.md`, nowhere graded |

**The thing that would have cost us most:** all nine `submission/level-*/q*.json` were `{}`.
That is a straight 0 on *Answer accuracy* — 30% of the code score — no matter how good the
code is. So that got fixed first.

---

## 1. The nine questions — revised

You gave me a free hand here. What changed and why:

| # | Before | After | Why |
|---|---|---|---|
| q1 | "What type of research paper is the document?" | "How many papers does the survey consider, and how many of them receive a more detailed treatment?" | The old q1 **reliably failed** (documented in `TODO.md`): it is a document-level meta question, not the "single passage" Level 1 asks for, and it hits a textbook vocabulary mismatch — the answer is on page 1 where the word "survey" appears 4×, but the question contains "paper"/"type"/"document", none of which occur on page 1 at all. Submitting a known-failing answer costs real marks. The failure itself is too good to throw away, so it stays in `TECHNICAL_NOTE.md` as a diagnosed negative result, which *Rigor* rewards. |
| q2 | unchanged | "What is the main goal of explainable information retrieval according to the survey?" | Already scores 0.99 on page 1. |
| q3 | "What are the defined evaluation of explanations types in IR?" | "How does the survey categorise the evaluation of explanations?" | Same target passage, grammatical English. |
| q4 | unchanged | "What is the main limitation the authors acknowledge?" | Good topic opener for the Level-2 chain. |
| q5 | unchanged | "Why does that happen?" | Textbook pronoun follow-up. |
| q6 | "What is the following section?" | "And how do they propose to address it?" | The old one was vague and barely conversational. The new one is a proper elliptical follow-up — "they" and "it" both need the history — which is exactly what Level 2 grades. |
| q7 | IDCM / Table 6 | "Across the classification tables in this survey … what is the most common way the quality of explanations is evaluated, and what do the authors themselves say about relying on that kind of evidence?" | The old q7 was answerable from **page 23 alone** (Table 6 and the IDCM paragraph sit on the same page), so it would have scored badly on *level-appropriate approach*. The new one forces aggregation over four tables on pages 10, 15, 22 and 23 plus the authors' commentary on pages 9 and 26. |
| q8 | unchanged | "Summarise the contribution of each section." | Genuine synthesis; the per-section summary index is built for it. |
| q9 | unchanged | shortcuts → §3.2.2 → §7.2 | Verified the premise against the PDF: "shortcuts" p1, marker-token attention p8, attention offloaded to punctuation → adversarial susceptibility p17. Real three-hop, three pages apart. |

Both `postman/fourth-turn.postman_collection.json` and the new
`questions/chosen.json` carry the final nine. `questions/chosen.json` is the single source
of truth that `scripts/run_questions.py` reads.

---

## 2. Safety net — the nine answers are no longer empty

New `scripts/run_questions.py` asks all nine through the live API in level order (so the
Level-2 chain actually threads) and writes each raw response into `submission/`. Nothing is
hand-edited: what the endpoint returned is what got filed.

```bash
uv run python scripts/run_questions.py        # all nine
uv run python scripts/run_questions.py 2      # one level
```

Results are in section 5 below.

---

## 3. Level 2 — `rewrite_query` is no longer a no-op

This was worth 10% of the code score on its own and it was returning its input unchanged.

**New file `app/rag/rewrite.py`.** Two stages:

1. **`is_context_dependent(question)`** — a cheap *grammatical* gate. Fires on a dangling
   reference ("that", "it", "they", "the latter"), on a continuation opener ("And …",
   "Why …", "What about …"), or on a question too short to carry content (< 6 words).
   Everything else is left completely alone, so standalone questions pay neither the
   latency nor the risk of a rewrite. The gate never looks at the topic — it would behave
   identically on any PDF, which matters for *Integrity*: it is not keyed to our nine
   questions.
2. **`standalone_query(question, history)`** — the chat model substitutes the antecedents
   in, given the last 3 turns with past answers clipped to 600 chars.

The rewrite is treated as untrusted: validated for single-line, non-empty, bounded length,
and on *any* failure it degrades to `previous question + current question` concatenated —
which still puts real content words in front of the embedder and needs no model at all. A
dead LLM degrades retrieval instead of breaking it.

**Wiring.** `retrieve()` now returns a `Retrieval(contexts, query)` instead of a bare list,
so the resolved query travels back out. Three things use it, and this matters:

- the **embedder** and **BM25** search with it (the point of the exercise);
- the **cross-encoder** reranks with it — scoring a chunk against "why does that happen?"
  tells it nothing either;
- **`citations.evidence_quote`** picks the supporting sentence with it, for the same reason.

**Visible in the response.** `Diagnostics.retrieval_query` now reports what was actually
searched. `QueryResponse`'s graded shape is untouched — `Diagnostics` is explicitly
"self-reported context, not graded for correctness", and the field is optional and
additive. It is the artefact that proves to the jury the resolution happened.

**`chat(messages, thinking=...)`** was added to the provider protocol so the mechanical
calls (rewrite, and the Level-3 decomposition next) can turn qwen3.6's reasoning off. It is
antecedent substitution, not reasoning, and thinking costs more than the rewrite itself.
LM Studio and litellm accept the flag and ignore it.

---

## 4. Level 3 — multi-query retrieval and a second index

**`app/rag/decompose.py`** — the chat model splits a whole-document question into 2–4
standalone sub-questions. Each gets its own dense + BM25 pass and *all* the rankings are
fused together in one RRF, not pairwise: fusing per sub-query and then fusing the results
applies RRF twice and buries a chunk that only one sub-query found, which is precisely the
chunk decomposition exists to rescue. If decomposition fails or yields fewer than two
usable lines, retrieval carries on with the single original query.

**`app/rag/sections.py`** — a second Qdrant collection, `aim_hackathon_sections`, holding
one LLM-written summary per section. The chunker already tags every chunk with its section
breadcrumb, so the boundaries are Docling's own `section_header` blocks, not a heuristic.
For a Level-3 question the whole outline goes into the prompt as a document map, and
`TOP_K` rises from 5 to 8.

**→ The integrity line, worth being able to defend on Friday.** Section summaries are
*generated text*. They are never quoted, never become a `Source`, and go into the prompt
unnumbered and explicitly labelled as a navigation aid — so the model physically cannot
cite one. Every `Source.quote` still comes from the verbatim chunk index. Summaries decide
*where to look*; only the document itself is ever evidence. If a juror pushes on "aren't
you feeding it AI-generated content?", that is the answer.

---

## 5. Level-1 polish — and a real grounding bug I did not expect

Cleared most of the remaining `TODO.md` items:

- **Reference chunks down-weighted** (`_REFERENCE_WEIGHT = 0.3` in RRF). The bibliography is
  85 of 269 chunks — 32% of the index — and was crowding the pool. Down-weighted rather
  than filtered, because "how many papers does the survey review?" is a fair question whose
  evidence lives in exactly those chunks. BM25 learned to read `kind` from the chunk
  filename so both arms can see it, not just the dense side.
- **The whole fused union is reranked.** Fusion used to cut to 20 before the cross-encoder
  — the only component with calibrated scores — ever saw the candidates. Now capped at 60.
- **Near-duplicates dropped** before the prompt. `CHUNK_OVERLAP=150` meant adjacent chunks
  shared sentences and could take two of five prompt slots; q6 did exactly that.
- **Batched embeddings** via Ollama's `/api/embed`, falling back to the per-text endpoint on
  a 404 from an older server. Ingest was 269 sequential HTTP calls.
- **PUA ligatures expanded at parse time.** The PDF's font stores "Qu" at `U+E048`, so
  Docling returned "ery:" where the page reads "Query:" — one had already leaked into a
  saved answer as `ery:can you do yoga from a chair`.
- **Pipeline logs are visible.** uvicorn only configures its own loggers, so every
  `log.info` in `app/rag/` was being swallowed. Now you can watch a follow-up get rewritten
  and a Level-3 question get decomposed, live in `docker compose logs -f app`.

**The bug I did not expect — and it was in the answers we had already committed.**

Our quotes were exact substrings of their chunk (sliced by offset, never generated) and
`TODO.md` recorded that at 100%. That measurement was **circular**: it compared our text
against our own parse. Checking against the PDF with an independent extractor (pypdf) found
two real problems:

1. **Typography.** The PDF contains 135 en-dashes, 53 right single quotes and 32 curly
   double quotes. Docling's output contains **none** of them — all folded to ASCII. So any
   quote with an apostrophe in it was not findable in the document a grader actually opens.
2. **Page numbers.** Docling labels a block with the page it *starts* on. A paragraph
   flowing across a page break is therefore cited one page early — and two of our committed
   q6 sources were, pointing at page 17 for text that is physically on page 18.

**`app/rag/verbatim.py`** fixes both by *locating* rather than rewriting: fold both sides to
a common form, find the quote in the PDF's own page text, and return the PDF's characters
for that span — with the page number corrected to wherever the text actually is. The cited
page is tried first and always wins a tie, so a correct citation is never moved. If the span
cannot be found anywhere, the quote is returned untouched; this can only improve fidelity,
never invent anything.

Matching ignores whitespace entirely, not just runs of it — the two extractors disagree
about *where* spaces go (pypdf reads a citation marker as `[ 30]`, Docling as `[30]`), so
comparing whitespace would measure the extractors against each other rather than measure
our grounding. Punctuation and case are **not** folded, so a passing quote matches the PDF
character for character.

**Measured, on the nine answers already committed:**

| | before | after |
|---|---|---|
| quotes verbatim on their cited page (vs the real PDF) | 16/19 (84%) | **19/19 (100%)** |
| page numbers corrected | — | 2 |

`scripts/audit_quotes.py` is the check, and it exits non-zero on failure so it can gate a
push. Run it any time with `uv run python scripts/audit_quotes.py`.

`pypdf` was added to `pyproject.toml` — it had been available locally only as a transitive
dependency and was **missing from the container**, so this would have silently done nothing
in the graded path.

---

## 6. Verified: Level 2 works now

Re-ran the Level-2 chain on the rebuilt index. The rewrites, straight from
`diagnostics.retrieval_query` in the committed answers:

```
q4 "What is the main limitation the authors acknowledge?"
   → unchanged. The gate correctly declined to rewrite a standalone question.

q5 "Why does that happen?"
   → "Why do simple models cannot faithfully explain all localities of a complex
      model's decision boundary, even when trained with significantly more data?"

q6 "And how do they propose to address it?"
   → "How do they propose to address the limitation that simple models cannot
      faithfully explain all localities of a complex model's decision boundary
      due to conflicting relevance factors?"
```

The q5 rewrite is not grammatical ("Why do simple models cannot") and I left it alone — it
is a retrieval query, not prose, and it carries exactly the content words the six-word
follow-up had none of. The effect:

| | before (no-op) | after |
|---|---|---|
| q5 best passage score | 0.054 | **0.50** |
| q5 answer | hedged: *"If referring to X… If referring to Y…"* | direct, single source |
| q6 sources | p11 0.20, p17 0.22, p17 0.11 — **a duplicate**, drifts onto LiEGe | p7 **0.98**, p25 0.86, p18 0.17, no duplicates, answers the actual follow-up |

Those two hedged/duplicated baseline answers are preserved in
`docs/ablations/baseline-no-rewrite/` so the comparison is reproducible rather than
asserted.

Quote audit after the change: **19/19 (100%)** verbatim on the cited page.

---

## 7. Final run — all nine, on the finished pipeline

| Q | Level | Latency | Sources | Behaviour |
|---|---|---|---|---|
| q1 | 1 | 16.0 s | 1 | p2 **1.00** — correct (68 / 32) |
| q2 | 1 | 24.0 s | 2 | p1 0.99 / 0.95 — correct |
| q3 | 1 | 34.7 s | 1 | p5 0.76 — correct |
| q4 | 2 | 60.9 s | 3 | standalone, **not rewritten** (the gate declining correctly) |
| q5 | 2 | 59.2 s | 1 | **rewritten** → p25 **0.70** |
| q6 | 2 | 68.0 s | 5 | **rewritten** → p7 **0.99** |
| q7 | 3 | 83.3 s | 3 | **2 sub-queries**; now cites 3 tables, not 2 |
| q8 | 3 | 104.6 s | 8 | **4 sub-queries** + outline; per-section summary answer |
| q9 | 3 | 77.2 s | 3 | **2 sub-queries**; all three hops (p1 → p8 → p17) |

Final gate: `scripts/audit_quotes.py` reports **27/27 (100%)** quotes verbatim on their
cited page, zero empty files, `ruff check` and `ruff format --check` both clean.

A late catch worth knowing about, because it was silently corrupting Level-3 answers: the
model writes grouped citations like `(2; 6)` when one claim rests on two passages, and the
marker regex only matched a lone `(1)`. So q7 leaned on four passages, **returned one
source**, and shipped markers pointing at nothing. Now parsed and renumbered as a group, and
the system prompt asks for `(2, 6)` explicitly. You can see it working in q2's answer above.

Also fixed: evidence quotes could be a bare section heading. "2.5 Evaluation of
Explanations" out-scored the prose beneath it for q3, because the heading echoes the
question's words. Headings are now set aside unless a chunk has nothing else, and q3's quote
is real supporting prose.

---

## 8. What I deliberately did NOT do — and why

**Abstention on a weak retrieval pool.** This was top of my list: the cross-encoder is
calibrated, and the q1 we retired had a pool topping out at 0.007, which is the system
correctly saying "not in this document". Then I tabulated the best score of every answer
that is *correct*:

| q1 | q2 | q3 | q4 | q5 | q6 | q7 | q8 | q9 |
|---|---|---|---|---|---|---|---|---|
| 0.998 | 0.994 | 0.763 | **0.119** | 0.503 | 0.979 | 0.262 | **0.028** | 0.178 |

Correct answers run from 0.028 to 0.998. The only gap available is between the failing
0.007 and q8's correct 0.028 — and choosing a constant inside a window that narrow, from
nine observations, is fitting to our own question set, which is what the rules penalise. It
would also have made q8 abstain, and q8 is right. **Shipping this would have made the
submission worse.** The reasoning and the table are in `TECHNICAL_NOTE.md` §6 and `TODO.md`,
which is a better rigor story than a feature that misfires.

---

## 9. Where things stand, and what's left for you

Everything is committed and pushed to `feat/reranker-and-citations`.

**→ YOUR CALL — the one thing I did not do:** grading reads **`main`**, and this work is
still on the feature branch. You asked me to push the branch and leave the merge to you:

```bash
git checkout main && git merge feat/reranker-and-citations && git push origin main
```

Remote `main` was also one commit behind your local `main` before I started, so the push
will carry that along too.

Open items, in the order I would do them, are in `TODO.md`. The honest summary: the
pipeline answers all nine questions correctly, every quote verifies against the PDF, and
the two things I would most want next are persistent memory (`rag/memory.py` is a dict that
dies with the container) and a labelled set of unanswerable questions so abstention can be
built on evidence rather than on a guessed constant.

Read `TECHNICAL_NOTE.md` before Friday — it is what the jury reads, and §5 ("what broke")
is the part worth being able to defend out loud.

---

## 10. Your two changes: no gate, and a ReAct loop for Level 3

**The grammatical gate is gone.** `rag/rewrite.py` now calls the model whenever there is
history, and the model decides — it is told to repeat a self-contained question exactly. In
the final run it does that for six of the nine and rewrites q5, q6 and q9. You were right
about the principle; the gate was making wrong calls in both directions (it would rewrite
"What is *this* survey about?", and miss "What about legal search?").

It was not free, though, and the first run after removing it was **worse** — worth knowing
because it is the argument someone will make against the change:

- q2 got contaminated. "…according to the survey" became "…according to the survey *that
  considers 68 papers, and a subset of 32 of them receive a more detailed treatment*" — it
  glued q1's answer onto an unrelated question.
- q5 over-resolved. q4's answer lists three limitations, so "that" became all three, and q5
  and q6 went straight back to hedging ("If you are referring to…") — the exact failure the
  rewriting exists to cure.

Both were prompt problems, not architecture problems. The rewriter is now told to resolve an
ambiguous reference to **one** thing — the main subject of the previous answer, never a list
— and to repeat a self-contained question *exactly* rather than enriching it. That fixed
both: q5 back to 0.67, q6 to 0.99, no hedging, and q2 left alone.

**`rag/agent.py` is the ReAct loop.** After retrieving, the model sees the best passages and
answers one question: is this enough, and if not, what is missing? Its follow-up queries get
searched and appended to the same pool. Three properties I was careful about:

- it judges **evidence, not its own answer** — asking a model whether it likes its answer
  invites it to loop until it agrees with itself; asking whether passages contain a fact is
  a checklist;
- it **cannot lose evidence** — every round appends to one pool that is fused and reranked
  once at the end, so a bad follow-up wastes a round but cannot displace an earlier hit;
- it **always terminates** — budget of 2, stop when it proposes nothing new, stop on any
  LLM error. The trace is in `diagnostics.retrieval_steps`.

**Measured, and the honest read is one win in three:**

| | decompose only | + loop |
|---|---|---|
| q7 | p10 0.52, p15 0.34, p16 0.27 | 0.52 / 0.33 / 0.27 — **no gain**, 2 steps spent |
| q8 | p21 0.08 + near-zeros | same — **no gain**, structure still comes from the outline |
| q9 | p1 0.16, p17 0.67, p9 0.06 | p17 **0.84**, **p8 0.23** — the only thing that ever retrieves §3.2.2 |
| latency | 83–105 s | 105–134 s |

q9 is the case Level 3 is actually about — the question names §3.2.2 and §7.2, and only the
loop found §3.2.2. q7 and q8 pay ~40 s for nothing. I kept it because the q9 gain is real
and a wasted round is harmless, but it is not a uniform win and the note says so.

**One defect worth seeing**, because it is invisible unless you read the generated queries.
Asked for a "search query", the model wrote Lucene:
`survey authors argue" AND ("fidelity" OR "faithfulness")`. Our embedder encodes the
operators as words and BM25 matches `AND` and the quotes as literal tokens. The prompt now
asks for plain questions *and* `_plain()` strips the syntax in code — a prompt rule is a
request, the strip is a guarantee.

Final state: **25/25 quotes verbatim**, lint and format clean, all nine committed. Still on
`feat/reranker-and-citations` — the merge to `main` is still yours to make.
