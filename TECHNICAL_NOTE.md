# Technical note — Team KrautWineSarmale

Document: *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*
(Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov, JMLR 2014), 30 pages,
committed at `data/in/srivastava14a.pdf`. 151 chunks, 34 section summaries. Everything
runs locally: no hosted API is used or required.

## 1. System

`POST /query` threads the question into a per-level conversation, resolves it into a
retrievable query, searches two indexes, reranks with a cross-encoder, answers from the
numbered passages, and returns only the evidence the answer actually cited — with each
quote located in the PDF itself before it ships.

```
question + level
 └─ memory.get_history("level-N")
 └─ rewrite_query           the model writes the search query   (whenever there is history)
 └─ decompose.sub_queries   question → 2-4 sub-questions        (Level 3 only)
 └─ agent.next_queries      judge the evidence, search again    (Level 3, ≤2 rounds)
 └─ per query: dense (bge-m3 → Qdrant cosine) + BM25 (rank_bm25 over data/chunks)
 └─ RRF over all arms, reference chunks down-weighted, near-duplicates dropped
 └─ bge-reranker-v2-m3 cross-encoder → top_k (5, or 8 at Level 3)
 └─ prompt: numbered passages [+ generated section outline at Level 3] + history
 └─ qwen3.6 answers, citing passages as (1), (2)
 └─ keep only cited passages; quote = cross-encoder-selected sentence, sliced by offset
 └─ verbatim.locate(): re-express the quote in the PDF's characters, verify/repair its page
```

| Stage | What we did | Changed from the scaffold? |
|---|---|---|
| Extraction | Docling Serve; `page_header`/`page_footer`/`footnote` dropped; tables flattened to pipe-separated rows; ligatures expanded (`ﬀ ﬁ ﬂ ﬃ ﬄ`, and PUA slots such as `U+E048` → `Qu`) | Yes — scaffold had none |
| Chunking | Section-aware over Docling's own `section_header` blocks; never spans a page; tables split on row boundaries; breadcrumb in `embed_text`, verbatim page text in `text` | Yes — scaffold indexed one vector per page |
| Embeddings / index | `bge-m3` (1024-d, symmetric) via Ollama; Qdrant cosine; **second index** of 34 section summaries | Yes |
| Retrieval | Hybrid dense + BM25, RRF, cross-encoder rerank, history-aware rewriting, Level-3 decomposition and a reflective retrieve-judge-search loop | Yes — scaffold was single dense search |
| Answer + citation | Model cites numbered passages; code widens the evidence to the largest span the PDF can vouch for; quote aligned to the PDF and its page verified | Yes — scaffold returned the chunk truncated to 300 chars |

Models: `qwen3.6` (chat, thinking on), `bge-m3` (embeddings), `bge-reranker-v2-m3` (Q8_0
GGUF via llama.cpp). Thinking is left enabled because the answers are better for it;
`strip_thinking()` removes the `<think>` block in every provider so a scratchpad can never
reach the graded `answer` field or break citation parsing. The mechanical calls — query
rewriting, decomposition, section summarisation — pass `thinking=False`, since substituting
an antecedent is not a reasoning task and reasoning there costs more than the call itself.

## 2. Level 2 — conversational memory

Putting the history in the prompt is necessary and **not sufficient**, and we reproduced
that on our own stack before fixing it. In a level-2 run whose history already contained an
answer naming the document a survey, the next turn still replied "the context does not
state" — because grounding comes from the retrieved context block, and retrieval never sees
the history. It sees only the question.

So `rag/rewrite.py` resolves the question **before** embedding: whenever there is a
conversation, the model reads it and writes the search query.

There is deliberately **no heuristic deciding whether a question needs resolving**. An
earlier version gated on grammar — dangling pronoun, continuation opener, question length —
and only called the model when the gate fired. We removed it, because the gate was wrong in
both directions: it would rewrite "What is **this** survey about?", which needs nothing, and
sail past "What about legal search?" if phrased with enough words. Working out what a
question refers to *is* a language-understanding problem, so it belongs to the language
model. The model is instructed to return a self-contained question unchanged — the same
judgement the gate was attempting, made by something equipped to make it. In the final run
it does exactly that for q1–q4, q7 and q8, and rewrites q5, q6 and q9.

Removing the gate is not free, and the first run after doing so was **worse**, in two ways
we then fixed in the prompt:

- it appended context to questions that did not need it — q2's "…according to the survey"
  became "…according to the survey *that considers 68 papers, and a subset of 32 of them
  receive a more detailed treatment*", dragging q1's answer into an unrelated query;
- it over-resolved an ambiguous reference. q4's answer lists three limitations, so "Why
  does that happen?" became a query about all three at once, and q5 and q6 went back to
  hedging ("If you are referring to…"), the exact failure the rewriting was meant to cure.

Both are prompt problems, not architecture problems: the rewriter is now told to resolve an
ambiguous reference to **one** thing (the main subject of the previous answer), never to a
list, and to repeat a self-contained question *exactly* rather than enriching it. That
restored q5 to 0.67 and q6 to 0.99 with no hedging.

The rewrite is validated and, on any failure, falls back to `previous question + current
question` concatenated — model-free, and still far better than embedding six pronouns. The
resolved query is used for the dense search, the BM25 search, the cross-encoder, **and**
the sentence-level quote selection, and is reported in `diagnostics.retrieval_query`.

Real output, taken from `diagnostics.retrieval_query`:

```
q4 as asked:     "What drawback of dropout do the authors report?"
retrieval query: unchanged — the model judged it self-contained and repeated it

q5 as asked:     "Why does that happen?"
retrieval query: "Why does dropout significantly increase training time"

q6 as asked:     "And what benefit does that same noise bring?"
retrieval query: "What benefit does the noisy parameter updates from dropout bring"
```

Six of the nine come back unchanged and three are resolved, which is the split the
question set implies.

## 3. Level 3 — whole-document reasoning

Three additions, because whole-document questions fail in more than one way.

**Multi-query decomposition** (`rag/decompose.py`), which writes the loop's opening moves.
q9 chains shortcut learning (page 1),
attention-based feature attribution (page 8) and probing (page 17) — topics sharing almost
no vocabulary. Embedded as one string it is a mediocre match for all three, and `top_k` goes
to whichever hop dominates. The model splits it into 2–4 standalone sub-questions, each
gets its own dense + BM25 pass, and every ranking is fused **together** (not pairwise, which
would apply RRF twice and bury a chunk only one sub-query found).

**A ReAct-style retrieval loop** (`rag/agent.py`). Decomposition writes its sub-questions
*before* anything has been retrieved, from the question alone — so if a hop comes back
empty, a single pass has no way to notice. After retrieving, the model is shown the best
passages so far and asked one thing: is this enough, and if not, what is missing? Its only
lever is proposing further search queries, which are retrieved and appended to the same
pool. Three properties keep it safe:

- **It judges evidence, not its own answer.** Asking a model whether it likes its answer
  invites it to loop until it agrees with itself; asking whether passages *contain* a fact
  is a checklist.
- **It cannot lose evidence.** Every round appends to one ranking pool that is fused and
  reranked once at the end, so a bad follow-up query wastes a round but can never displace
  a passage an earlier round found.
- **It always terminates.** A hard step budget (2), a stop when it proposes nothing new,
  and any LLM failure ends the loop with the evidence in hand.

The trace is in `diagnostics.retrieval_steps`, so what the loop did is visible in the
graded response rather than only in the logs.

One defect worth recording, because it is invisible unless you read the queries the loop
generates. Asked for a "search query", the model produced Lucene:
`survey authors argue" AND ("fidelity" OR "faithfulness") AND ("limitation")`. Both of our
arms are the wrong audience for that — the embedder encodes the operators as words, BM25
matches `AND` and the stray quotes as literal tokens. The prompt now asks for plain
questions **and** `_plain()` strips boolean/field syntax in code, because a prompt rule is a
request and the strip is a guarantee.

**A section index** (`rag/sections.py`). The chunker already tags each chunk with its
section breadcrumb, so grouping by that gives real section boundaries for free — they are
Docling's own headers, not a heuristic. Each of the 44 body sections is summarised once at
ingest and indexed under its own embedding. For a Level-3 question the whole outline goes
into the prompt as a document map.

**Integrity boundary.** Section summaries are generated text. They are never quoted, never
become a `Source`, and are labelled in the prompt as a navigation aid carrying no citation
number — so the model cannot cite one. Every `Source.quote` still comes from the verbatim
chunk index. Summaries decide *where to look*; only the document is evidence.

## 4. Measurement

All numbers are from this repository on the committed PDF, host Ollama with `qwen3.6` +
`bge-m3`: 30 pages → 151 chunks → 34 section summaries. Both ablations are reproducible —
`REWRITE_ENABLED` and `AGENT_ENABLED` in `.env` are the switches, and the baseline
responses are committed under `docs/ablations/`.

Cross-encoder scores are comparable across rows because the reranker is calibrated, unlike
the RRF scores it replaced.

### Ablation 1 — query rewriting (Level 2)

`REWRITE_ENABLED=false` embeds the follow-up exactly as typed. Everything else is identical.

| | no rewriting | with rewriting |
|---|---|---|
| q4 (topic opener, standalone) | p24 **0.97** | p24 **0.97** — identical, the control |
| q5 "Why does that happen?" | p7 0.55, p15 0.29 | p24 **1.00** |
| q5 answer | **answers a different question**: explains why dropout wipes out information in pretrained weights during finetuning | explains why training takes 2–3× longer — what was actually asked |
| q6 "And what benefit does that same noise bring?" | p5 0.01, p1 0.01, p7 0.09 | p7 **0.96**, p10 0.50 |

q5 is the clearest result we have. Without rewriting, "Why does that happen?" carries no
retrievable content, so the retriever lands on an unrelated passage about finetuning and
the model dutifully answers *that* question instead — fluently, with a real citation, and
wrong. It is not a failure that announces itself, which is exactly why Level 2 is worth
the 10% it carries. q4 is the control: standalone, left unrewritten, unchanged.

### Ablation 2 — quote grounding against the real PDF

`scripts/audit_quotes.py` searches every quote on its cited page using pypdf — a
*different* extractor from the Docling one we index with, so a pass means two independent
readers agree. Both sides are normalised only for extraction artefacts: whitespace,
line-break hyphenation, ligatures, and the dash/quote variants the extractors disagree
about. Case and wording are untouched, so a paraphrase fails.

| | | |
|---|---|---|
| quotes verbatim on their cited page | **100%** | every source, every question |
| median quote length | 150 → **580 chars** | after widening to the largest verifiable span |
| page numbers corrected against the PDF | 2 (previous document) | Docling labels a block with the page it *starts* on |

### Ablation 3 — the reflective retrieval loop (Level 3)

`AGENT_ENABLED=false` runs decomposition only. This is the ablation that went against us,
and it is the most useful thing we measured.

| | loop off | loop on (after the fixes below) |
|---|---|---|
| q7 | p8 0.93 | p8 **0.94** — unchanged, +13 s |
| q8 | p3 0.89 | p3 **0.83** — unchanged within run-to-run noise |
| q9 | p15 0.03, p16 **0.00** | p16 **0.78**, p15 0.61, p15 0.53 |
| latency | 84–124 s | 95–118 s |

q9 is the case Level 3 exists for: it asks where the paper *shows* co-adaptation being
broken, and one-shot decomposition retrieved essentially nothing (0.03 and 0.00). The loop
read that, said the evidence was missing, searched for it, and found §7.1 on pages 15–16.

Getting there meant finding a bug the ablation exposed, and it was not the one we assumed.
Our first measurement had the loop making q8 *worse* — p3 fell from 0.89 to 0.03. The
mechanism: the cross-encoder scores every candidate **independently**, so extra candidates
can never lower an existing one's score. The only way a good passage loses is by never
reaching the reranker. The paragraph beginning *"In Section 6, we present our experimental
results…"* was out-voted in fusion by six invented section-specific queries, fell below the
60-candidate pool cut, and was never scored at all.

We had asserted in code comments that "a wasted round cannot displace an earlier hit,
because everything is fused at the end". That was false. Two fixes made it true:

- **Weighted fusion.** The user's question votes at 1.0, its decomposition at 0.7, a
  speculative follow-up at 0.4. Necessary but not sufficient — this alone recovered q7 and
  left q8 broken, because a hard cut cannot be fixed by soft ordering.
- **A reserved pool.** Half the candidate slots belong to the question and its
  decomposition; follow-up rounds fill only the remainder. Extra retrieval is now additive
  by construction rather than by assertion.

The honest sequence is worth stating plainly: the reflective loop as first written **hurt**,
the ablation is what caught it, and the fix was a retrieval-plumbing bug rather than
anything to do with the loop's judgement. After the fix it is a clear win on one of three
Level-3 questions and neutral on the other two, for 15–20 % more latency.

### Per-question result, final run

Best cross-encoder score for each question's evidence, and wall-clock latency:

| | q1 | q2 | q3 | q4 | q5 | q6 | q7 | q8 | q9 |
|---|---|---|---|---|---|---|---|---|---|
| best source | 0.81 | 0.99 | 0.97 | 0.97 | **1.00** | 0.96 | 0.94 | 0.83 | 0.78 |
| latency (s) | 73 | 51 | 48 | 22 | 33 | 47 | 95 | 118 | 76 |
| sources | 2 | 2 | 2 | 1 | 1 | 2 | 8 | 7 | 3 |

All nine were checked by hand against the PDF and are factually correct, including the
numbers q7 quotes (21.8% TIMIT phone error rate, 31.05% → 29.62% on Reuters, both page 13).
All 28 evidence quotes are verbatim on their cited page.

### Cost

From `diagnostics.latency_ms`, wall clock, thinking enabled:

| Level | Latency | Why |
|---|---|---|
| 1 | 48–73 s | one rewrite call, one dense + one BM25 pass, rerank, answer |
| 2 | 22–47 s | same, with the rewrite actually changing the query |
| 3 | 76–118 s | plus decomposition, up to 2 judge calls with their retrieval passes, 8 passages and the 34-section outline in the prompt |

The cross-encoder is not the expensive part anywhere: scoring a 60-candidate pool costs
well under a second against a 20–120 s generation. Ingest is 2m50s end to end — 30 pages
parsed, 151 chunks embedded, 34 sections summarised.

## 5. What broke

**Two extraction defects that made correct quotes look fabricated.** Both were invisible
until we checked against the PDF with a second extractor rather than against our own parse.

*Ligatures.* This paper contains 121 `ﬁ` and 91 `ﬀ` glyphs — "diﬀerent", "eﬀect",
"overﬁtting". Our fold table had `ﬁ` and `ﬂ` but not `ﬀ`, so a large share of quotes would
have failed to match the document they came from. Expanded at parse time now, and folded
on both sides when matching.

*Line-break hyphenation.* pypdf reports the page as typeset — `optimiza- tion`,
`au- tomatically` — while Docling rejoins the word. Neither hyphen is content; it records
where the line happened to end. Before we handled it, two perfectly correct quotes were
reported as non-verbatim. Matching now ignores hyphens on both sides and the returned span
has the break repaired, so the quote reads as prose.

**A degraded answer filed into the submission.** Level-3 generation measured up to ~135 s
against a 120 s `REQUEST_TIMEOUT`, and one q7 run timed out. The pipeline degrades rather
than 500s — right for the API, wrong for the deliverable: it wrote
`[LLM unavailable: ollama chat failed: timed out]` over a good answer. The timeout is now
300 s, and `scripts/run_questions.py` refuses to overwrite an answer with a degraded
response. The lesson generalises: a graceful degradation is only graceful if something
downstream knows it happened.

**A retrieval claim we had asserted but never tested.** Described in §4 — the reflective
loop was silently evicting the best passage from the candidate pool, and our own code
comments claimed that could not happen. The ablation caught it. This is the strongest
argument we have for measuring ablations rather than reasoning about them.

**A question that could not be retrieved (previous document).** On our earlier document we
asked "What type of research paper is the document?" and it failed on every run. The answer
was the title on page 1, but none of "paper", "type" or "document" occurs on page 1, while
they occur 43×/23×/181× elsewhere in table headers — so BM25 ranked page 1 near-last and
the dense side did not rescue it. The cross-encoder scored that entire pool ≤0.007, which
is the calibrated "nothing here" signal RRF could never give. We replaced the question
rather than special-casing it: it was a document-level meta question, and Level 1 asks for
a fact answerable from a single passage. Fixing it in code would have meant a branch keyed
to one question, which is what *Integrity* penalises.

## 6. Limitations and next steps

- **Memory is process-local.** A dict in `rag/memory.py`; it does not survive a restart and
  does not work across workers. Redis would be a small change and we did not make it.
- **The reflective loop earns its keep on one question in three.** q9 goes from nothing to
  0.78; q7 and q8 are unchanged and pay 15–20 % latency. A cheap improvement we did not
  build: stop early when a round retrieves nothing the pool did not already contain.
- **Level-2 answers vary between runs.** When a topic-opening answer enumerates several
  things, "why does that happen?" is genuinely ambiguous and different runs resolve it to
  different items. The rewriter is instructed to take the first, which makes it stable in
  practice, but the underlying ambiguity is real and we have not measured it across many
  runs.
- **Section summaries are only as good as one pass of a local model.** They are never
  quoted, so a bad summary misdirects retrieval rather than corrupting evidence — but it
  can still misdirect it.
- **No table-structure reasoning.** Tables are flattened to pipe-separated rows and split on
  row boundaries, which keeps rows intact but leaves the model to parse the layout. A
  question needing a specific cell by row *and* column header is not handled structurally.
- **We do not abstain.** The idea is sound — on our previous document a genuinely
  unanswerable question produced a pool topping out at 0.007, exactly the "not in this
  document" signal a calibrated reranker should give. But on the submitted document every
  one of the nine answers scores between **0.78 and 1.00**, so we have no failing case here
  to calibrate a threshold against, and on the previous document correct answers ran as low
  as 0.021 — overlapping the failure outright. Picking a constant from nine observations
  would be fitting it to our own question set. A defensible threshold needs a labelled set
  of questions the document genuinely cannot answer; we do not have one, so we ship without
  abstention and say so.

---

**Repository**: https://github.com/codrinasus/The-Fourth-Turn
**Provider / models**: Ollama · `qwen3.6` · `bge-m3` · `bge-reranker-v2-m3`
**Team**: Ionita Catalin Nihai, Nathan Nowakowski, Bjoern Nieth, Limona Andrei Codrin
