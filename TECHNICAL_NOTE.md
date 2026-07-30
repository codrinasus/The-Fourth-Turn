# Technical note — Team KrautWineSarmale

Document: *Explainable Information Retrieval: A Survey* (Anand et al.), 35 pages, committed
at `data/in/document.pdf`. Everything runs locally: no hosted API is used or required.

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
| Extraction | Docling Serve; `page_header`/`page_footer`/`footnote` dropped; tables flattened to pipe-separated rows; PUA ligatures expanded (`U+E048` → `Qu`) | Yes — scaffold had none |
| Chunking | Section-aware over Docling's own `section_header` blocks; never spans a page; tables split on row boundaries; breadcrumb in `embed_text`, verbatim page text in `text` | Yes — scaffold indexed one vector per page |
| Embeddings / index | `bge-m3` (1024-d, symmetric) via Ollama; Qdrant cosine; **second index** of 44 section summaries | Yes |
| Retrieval | Hybrid dense + BM25, RRF, cross-encoder rerank, history-aware rewriting, Level-3 decomposition and a reflective retrieve-judge-search loop | Yes — scaffold was single dense search |
| Answer + citation | Model cites numbered passages; code selects the span; quote aligned to the PDF and its page verified | Yes — scaffold returned the chunk truncated to 300 chars |

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

Real output from the final run, taken from `diagnostics.retrieval_query`:

```
q4 as asked:     "What is the main limitation the authors acknowledge?"
retrieval query: unchanged — the model judged it self-contained and repeated it

q5 as asked:     "Why does that happen?"
retrieval query: "Why can simple models not faithfully explain all localities of a
                  complex model's decision boundary"

q6 as asked:     "And how do they propose to address it?"
retrieval query: "How do the authors propose to address the limitation that simple
                  models cannot faithfully explain all localities of a complex model's
                  decision boundary?"
```

Six of the nine come back unchanged and three are resolved, which is the split the question
set implies. The effect is measured in §4: for q5 the cross-encoder's score for the best
passage goes from 0.054 to 0.67 and the answer stops hedging.

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

All numbers below are from this repository on the committed PDF, host Ollama with
`qwen3.6` + `bge-m3`, 35 pages → 269 chunks → 44 section summaries.

### Ablation 1 — query rewriting (Level 2)

The same three questions, same index, same models; the only change is whether
`rewrite_query` resolves the follow-up before retrieval. Cross-encoder scores are
comparable across rows because the reranker is calibrated, which RRF scores were not.

| | no-op (baseline) | with rewriting |
|---|---|---|
| q4 (topic opener, standalone) | p25 0.011, p7 0.022 | p7 0.021 — left as asked, correctly |
| q5 "Why does that happen?" | p25 **0.054**, p6 0.034 | p25 **0.668** |
| q5 answer | hedges across two readings: *"If referring to the difficulty of disentangling explanations… If referring to why ranking scores drop…"* | one reading, one source, direct |
| q6 "And how do they propose to address it?" | p11 0.20, p17 0.22, p17 0.11 — **a duplicate source**, answer drifts onto LiEGe | p7 **0.987**, p25 0.41 — answers the follow-up |

The baseline answers are committed at `docs/ablations/baseline-no-rewrite/`, so this is
reproducible rather than asserted. q4 is the control: it is standalone, the gate declines
to rewrite it, and it is unchanged apart from retrieval-pool improvements.

### Ablation 2 — quote grounding against the real PDF

Checked with `scripts/audit_quotes.py`, which searches each quote on its cited page of
`data/in/document.pdf` using pypdf — a *different* extractor from the Docling one we index
with, so a pass means two independent readers agree the text is there. Whitespace is
normalised; punctuation and case are not.

| | before `verbatim.locate()` | after |
|---|---|---|
| quotes verbatim on their cited page | 16/19 (84%) | **19/19 (100%)** |
| page numbers corrected | — | 2 |

We had previously measured this at 100% by comparing quotes against our own parse. That
number was circular and hid both defects described in §5.

### Ablation 3 — decomposition and the section index (Level 3)

Same three questions, before and after multi-query decomposition plus the outline. Baseline
responses at `docs/ablations/baseline-single-query/`.

| | single query | decomposed + outline |
|---|---|---|
| q7 (four tables + commentary) | 2 sources, p10 0.52 / p15 0.34; answer covers Tables 1 and 3 only | split into **exactly the four sub-questions matching the four tables**; answer names the dominant evaluation mode *and* the authors' critique of it |
| q8 "summarise each section" | p2 0.004, p14 0.011 — retrieval scores at the floor | outline supplies the structure; chunk scores stay low **and honestly so**, because no passage answers this question |
| q9 (3-hop, pages 1/8/17) | already found all three hops | unchanged — decomposition neither helped nor hurt |

The honest reading: q7 improved clearly, q8 became answerable at all, and **q9 shows no
gain** — its hops share enough vocabulary that one query already ranked all three. We are
reporting that rather than claiming three wins. q8's near-zero scores are worth dwelling on:
they are correct. There is no passage in the paper that summarises every section, so a
retrieval score at the floor is the system accurately reporting that the answer had to come
from structure rather than from a passage.

### Ablation 4 — the reflective retrieval loop (Level 3)

Same questions and index, with `AGENT_ENABLED` off and on. The loop is what searches for
what decomposition alone did not think to ask for.

| | decompose only | + reflective loop |
|---|---|---|
| q7 | p10 0.52, p15 0.34, p16 0.27 | p10 0.52, p15 0.33, p16 0.27 — **no gain**, 2 steps spent |
| q8 | p21 0.08 + 7 near-zero | p21 0.08 + 7 near-zero — **no gain**, structure still comes from the outline |
| q9 | p1 0.16, p17 0.67, p9 0.06 | p17 **0.84**, p8 0.23, p9 0.13, p8 0.12 — **finds §3.2.2 on page 8**, which the single pass missed entirely |
| Level-3 latency | 83–105 s | 105–134 s |

Honestly read: **one of three improved.** q9 is a real win — the question names §3.2.2 and
§7.2, and only the loop retrieved §3.2.2, after judging the first pass incomplete. q7 and
q8 spent both steps and gained nothing, at ~40 s each. The loop's own trace says so: on q9
it reports the intro/shortcut link still missing when the budget runs out, which is
accurate — it never surfaced page 1.

So the loop buys recall on multi-hop questions where one hop is lexically distant, and buys
nothing on questions whose answer is structural (q8) or already well covered (q7). We kept
it because the q9 gain is the case Level 3 is actually about, and because a wasted round
cannot displace evidence — but it costs ~40 % latency for a benefit that is not uniform.

### Cost

From `diagnostics.latency_ms`, wall clock, thinking enabled:

| Level | Latency | Why |
|---|---|---|
| 1 | 34–50 s | one rewrite call, one dense + one BM25 pass, rerank, answer |
| 2 | 48–62 s | same, with the rewrite actually changing the query |
| 3 | 105–134 s | plus decomposition, up to 2 judge calls and their retrieval passes, 8 passages and the outline in the prompt |

The rewrite now runs on every turn that has history rather than only when a gate fired,
which adds ~2 s (thinking off) to Level-1 questions that end up unchanged. That is the
price of not having a heuristic decide; it is small, and the gate was making wrong calls.

Ingest is 3m05s end to end: 35 pages parsed, 269 chunks embedded, 44 sections summarised.
The cross-encoder is not the expensive part anywhere — scoring a 60-candidate pool costs
well under a second against a 30–110 s generation.

## 5. What broke

**A question that could not be retrieved, and why.** Our original q1 was "What type of
research paper is the document?" It failed on every run, including one whose own history
contained the answer. The diagnosis is a textbook vocabulary mismatch, and it is
deterministic — repeat runs returned identical pages and scores, so it is systematic, not
sampling. The answer is the title on page 1, where "survey" appears 4×. But the question
tokenises to `what type of research paper is the document`, and **none of "paper", "type"
or "document" occurs on page 1 at all**, while they occur 43×/23×/181× elsewhere, mostly in
`Paper | Task | …` table headers. BM25 therefore ranks page 1 near-last and the dense side
does not rescue it. After adding the cross-encoder the pool for this question scored ≤0.007
across the board — the reranker correctly reporting that retrieval had found nothing
relevant, which is the calibrated-abstention signal RRF could never give us (RRF scored
everything 0.016–0.03 whether good or garbage).

We replaced the question rather than special-casing it: it is a document-level meta question
and Level 1 asks for a fact answerable from a single passage, so it was the wrong question
for the level. Fixing it in code would have meant a branch keyed to it, which is exactly
what *Integrity* penalises. The diagnosis is kept here because it is the most useful thing
we learned about our own retrieval.

**Citations that were verbatim against the wrong thing.** Our quotes were exact substrings
of the chunk they came from — sliced by offset, never generated — and we had measured that
at 100%. The measurement was circular: it compared our text against our own parse. Checking
against the PDF with an independent extractor showed the file contains 135 en-dashes, 53
right single quotes and 32 curly double quotes that Docling folds to ASCII, so any quote
containing an apostrophe was not findable in the document a grader actually opens. The same
check found Docling labels a block with the page it *starts* on, so a paragraph crossing a
page break is cited one page early — real, in our submitted answers, twice.

## 6. Limitations and next steps

- **Memory is process-local.** A dict in `rag/memory.py`; it does not survive a restart and
  does not work across workers. Redis would be a small change and we did not make it.
- **The rewrite gate is heuristic.** It fires on grammar, so a context-dependent question
  phrased without a pronoun ("What about legal search?") is caught by the length rule rather
  than by understanding, and a standalone question containing "this" is rewritten
  unnecessarily. The rewriter is told to pass such questions through unchanged, which
  contains the cost but does not remove it.
- **Section summaries are only as good as one pass of an 8B model.** They are never quoted,
  so a bad summary misdirects retrieval rather than corrupting evidence — but it can still
  misdirect it.
- **No table-structure reasoning.** Tables are flattened to pipe-separated rows and split on
  row boundaries, which keeps rows intact but leaves the model to parse the layout. A
  question needing a specific cell by row *and* column header is not something we handle
  structurally.
- **We do not abstain, and we checked before deciding not to.** The cross-encoder's scores
  are calibrated, so an abstention threshold looks obvious: the pool for the q1 we retired
  topped out at **0.007**, which is the system correctly saying "not in this document".
  Then we tabulated the best score of every answer we believe is *correct*:

  | | q1 | q2 | q3 | q4 | q5 | q6 | q7 | q8 | q9 |
  |---|---|---|---|---|---|---|---|---|---|
  | best source score | 0.998 | 0.994 | 0.763 | **0.021** | 0.668 | 0.987 | 0.519 | **0.076** | 0.844 |

  Correct answers run from **0.021 (q4) to 0.998 (q1)**. q4 is the finding: a correct,
  well-grounded answer scoring 0.021 sits essentially on top of the 0.007 that marked the
  genuinely unanswerable question. The two distributions overlap, so no threshold both
  keeps q4 and rejects the failure — and any constant we picked anyway would be fitted to
  these nine questions, which is what the rules penalise.

  q8 is the other instructive case: 0.076 *and right*, because no passage summarises every
  section — its answer legitimately comes from the section index rather than from a quote,
  so a low passage score is the correct reading of the evidence, not a warning.

  A defensible threshold needs a labelled set of questions the document genuinely cannot
  answer. We do not have one, so we ship without abstention and say so.

---

**Repository**: https://github.com/codrinasus/The-Fourth-Turn
**Provider / models**: Ollama · `qwen3.6` · `bge-m3` · `bge-reranker-v2-m3`
**Team**: Ionita Catalin Nihai, Nathan Nowakowski, Bjoern Nieth, Limona Andrei Codrin
