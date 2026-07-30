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
 └─ rewrite_query           follow-up → standalone query        (Level 2)
 └─ decompose.sub_queries   question → 2-4 sub-questions        (Level 3 only)
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
| Retrieval | Hybrid dense + BM25, RRF, cross-encoder rerank, history-aware rewriting, Level-3 decomposition | Yes — scaffold was single dense search |
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

So `rag/rewrite.py` resolves the follow-up **before** embedding, in two stages:

1. a grammatical gate — dangling reference ("that", "it", "they"), continuation opener
   ("And …", "Why …"), or fewer than six words. It tests *form*, not topic, so it behaves
   identically on any document and encodes nothing about our nine questions;
2. the chat model substitutes the antecedents, given the last three turns.

The rewrite is validated and, on any failure, falls back to `previous question + current
question` concatenated — model-free, and still far better than embedding six pronouns. The
resolved query is used for the dense search, the BM25 search, the cross-encoder, **and**
the sentence-level quote selection, and is reported in `diagnostics.retrieval_query`.

Real output from the final run, taken from `diagnostics.retrieval_query`:

```
q4 as asked:     "What is the main limitation the authors acknowledge?"
retrieval query: unchanged — the gate correctly declined to rewrite a standalone question

q5 as asked:     "Why does that happen?"
retrieval query: "Why do simple models cannot faithfully explain all localities of a
                  complex model's decision boundary, even when trained with
                  significantly more data?"

q6 as asked:     "And how do they propose to address it?"
retrieval query: "How do they propose to address the limitation that simple models
                  cannot faithfully explain all localities of a complex model's
                  decision boundary due to conflicting relevance factors?"
```

The q5 rewrite is not grammatical — "Why do simple models cannot" — and we are leaving it
that way, because it is the retrieval query, not the answer. What matters is that it
carries the content words the bare follow-up had none of, and the effect is measured in
§4: the cross-encoder's score for the best passage goes from 0.054 to 0.50, and the
answer stops hedging.

## 3. Level 3 — whole-document reasoning

Two additions, because whole-document questions fail in two different ways.

**Multi-query decomposition** (`rag/decompose.py`). q9 chains shortcut learning (page 1),
attention-based feature attribution (page 8) and probing (page 17) — topics sharing almost
no vocabulary. Embedded as one string it is a mediocre match for all three, and `top_k` goes
to whichever hop dominates. The model splits it into 2–4 standalone sub-questions, each
gets its own dense + BM25 pass, and every ranking is fused **together** (not pairwise, which
would apply RRF twice and bury a chunk only one sub-query found).

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

<!-- FILL: final table -->

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
- **We do not abstain.** The cross-encoder's calibrated scores would support it — a pool
  topping out at 0.007 means "not in this document" — but we currently answer from the best
  available context regardless. This is the single change we would make with another day.

---

**Repository**: https://github.com/codrinasus/The-Fourth-Turn
**Provider / models**: Ollama · `qwen3.6` · `bge-m3` · `bge-reranker-v2-m3`
**Team**: Ionita Catalin Nihai, Nathan Nowakowski, Bjoern Nieth, Limona Andrei Codrin
