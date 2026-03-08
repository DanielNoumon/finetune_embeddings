# Fine-tuning Embeddings for Dutch Legal RAG

A step-by-step guide to fine-tuning an embedding model for retrieval-augmented generation on Dutch legal documents (EU AI Act).

## Step 1 — Chunking the Source Document

### The document

The EU AI Act (`eu_ai_act_NL.pdf`) is a 144-page Dutch legal regulation. It has three distinct zones:

| Section | Share | Content |
|---|---|---|
| Recitals (Overwegingen) | ~42% | 180 numbered recitals providing legislative intent |
| Articles (Artikelen) | ~49% | 113 articles across 13 chapters — the binding provisions |
| Annexes (Bijlagen) | ~8% | 13 annexes with reference lists and technical requirements |

### Why structural chunking over naive splitting?

Fixed-size chunking (e.g. 512 tokens) would split articles mid-sentence and lose legal context. Legal documents have an inherently well-defined hierarchy — articles, paragraphs (leden), and sub-items are atomic semantic units designed by legislators. A structural chunker respects these boundaries.

We chose **not** to use an LLM-based chunker because:
- The document's hierarchy IS the optimal boundary set — an LLM can't improve on it.
- Structural parsing is deterministic and reproducible.
- It scales to larger document sets without API costs.

### Chunking strategy

1. **Clean extraction** — strip headers/footers, fix column-break word splits.
2. **Parse by structure** — recitals by number, articles by paragraph (`lid`), definitions individually, annexes by item.
3. **Size guardrails** — target 50–1000 tokens. Oversized chunks split at sentence boundaries with overlap; tiny chunks merged with neighbours.
4. **Rich metadata** — each chunk carries `section_type`, `chapter`, `article_number`, `paragraph_number`, `hierarchy_path`.

### Two output versions

We produce two JSONL files:
- **`chunks_without_context.jsonl`** — raw chunk text, used for training data.
- **`chunks_with_context.jsonl`** — each chunk prefixed with a contextual header (e.g. `EU AI Act (NL) > Hoofdstuk III > Artikel 9 > Lid 2:`), used to test whether context-enriched indexing improves retrieval.

This lets us later compare retrieval quality between the two representations.

### Result

573 chunks: 223 overwegingen, 329 artikelen, 21 bijlagen. Token range 50–1015, average 283.

---

## Step 2 — Synthetic Query Generation

### Why synthetic data?

We need `(query, relevant_chunk)` pairs to train the embedding model with Multiple Negatives Ranking Loss (MNRL). Real user queries don't exist yet, so we generate them with an LLM.

### Query diversity

For each chunk, we generate 3–5 diverse Dutch queries across different types:
- **Factual** — asking for specific facts stated in the chunk.
- **Definitional** — asking what a term or concept means.
- **Procedural** — asking about processes or requirements.
- **Scenario-based** — describing a situation and asking which rules apply.

Diversity ensures the embedding model learns to match varied phrasings to the same content, not just keyword overlap.

### Training format

We generate `(anchor, positive)` pairs only — no hard negatives at this stage. MNRL uses in-batch negatives: with batch size 64, each sample gets 63 negatives for free (every other sample in the batch serves as a negative). 

**Why batch size 64?** Larger batches provide more negatives per sample, which strengthens the training signal. Batch size 64 is chosen as a balance between:
- **GPU memory** — fits on a 16GB GPU with ~110M parameter models (e.g., multilingual-e5-base)
- **Training quality** — 63 negatives per sample is sufficient; beyond 128 the returns diminish
- **Dataset size** — with ~2,300 pairs, batch size 64 gives ~36 batches per epoch (enough for stable gradients)

For smaller GPUs (8GB), use batch size 32. For larger GPUs (24GB+), batch size 128 can improve quality slightly but isn't necessary.

Hard negatives are mined later from the fine-tuned model checkpoint (Stage 2 training), because mining from an already-adapted model produces more informative negatives than mining from the base model.

### Positive chunks use `chunks_without_context.jsonl`

We train on raw chunks (no context header) so the model doesn't become dependent on seeing the prefix at inference time. The context-prefixed version is only used for retrieval comparison experiments.
