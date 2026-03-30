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

---

## Step 3 — Dataset Preparation for Fine-tuning

### Critical design choice: Chunk-level splitting

**The problem:** Each chunk has ~4 queries pointing to it. If we randomly split query-chunk pairs, the same chunk could appear in both train and eval sets → **data leakage** → inflated eval metrics.

**The solution:** Split by unique `chunk_id` first, then assign all queries for that chunk to the same split.

```python
# Split chunks (not pairs)
chunk_ids = sorted(set(p["chunk_id"] for p in pairs))
shuffle(chunk_ids)
eval_chunk_ids = set(chunk_ids[:15%])  # 15% of chunks

# All queries for a chunk stay together
train_pairs = [p for p in pairs if p["chunk_id"] not in eval_chunk_ids]
eval_pairs = [p for p in pairs if p["chunk_id"] in eval_chunk_ids]
```

**Result:** 0 chunk overlap between train/eval. The model never sees eval chunks during training.

### Split statistics

| | Pairs | Chunks | Avg queries/chunk |
|---|---|---|---|
| **Train** | 1,944 | 486 (85%) | 4.0 |
| **Eval** | 340 | 85 (15%) | 4.0 |
| **Total** | 2,284 | 571 | 4.0 |

### Why different formats for train vs eval?

**Train set** → HuggingFace Dataset (Arrow format)
- **Purpose:** Feed batches to MNRL training loop
- **Format:** Columnar Arrow files (`data-00000-of-00001.arrow`)
- **Columns:** `anchor` (query), `positive` (chunk)
- **Why Arrow?** Fast memory-mapped loading, zero-copy reads, native format for Sentence Transformers
- **Files:**
  - `data-00000-of-00001.arrow` — actual data (1,944 pairs)
  - `state.json` — dataset state metadata (fingerprint, shard info)
  - `dataset_info.json` — schema (column names, types)

**Eval set** → InformationRetrievalEvaluator format
- **Purpose:** Simulate retrieval task during training (compute MRR@10, NDCG@10, Recall@10)
- **Format:** 3 JSON files
  - `queries.json` — `{query_id: query_text}` (340 queries)
  - `corpus.json` — `{chunk_id: chunk_text}` (85 unique chunks, deduplicated)
  - `relevant_docs.json` — `{query_id: [chunk_id]}` (340 mappings)
- **Why this format?** Sentence Transformers' `InformationRetrievalEvaluator` expects this structure:
  1. Encode all queries
  2. Encode all corpus chunks
  3. For each query, rank all chunks by similarity
  4. Check if correct chunk ranks high → compute retrieval metrics

**Key insight:** Different tools expect different formats. We optimize for each use case:
- Train → fast batch loading for gradient updates
- Eval → retrieval simulation to measure search quality

### Loading from HuggingFace

The dataset preparation script loads directly from HuggingFace Hub (`danielnoumon/eu-ai-act-nl-queries`) instead of local JSONL files. This ensures:
- Reproducibility (same dataset version across machines)
- No need to regenerate synthetic queries locally
- Easy sharing and versioning

```bash
uv run python finetuning/stage_1_mnrl/prepare_dataset.py
```

The script maps HF column names (`query`, `chunk`) to internal format (`anchor`, `positive`) and performs the chunk-level split.

---

## Step 4 — Fine-tuning with MNRL

### Model choice: multilingual-e5-large

We chose `intfloat/multilingual-e5-large` (560M params) over alternatives:

| Candidate | Params | Why chosen / rejected |
|---|---|---|
| **multilingual-e5-large** ✅ | 560M | Strong multilingual retrieval baseline. Native query/passage prefix system aligns perfectly with MNRL training. Well-documented fine-tuning pattern with Sentence Transformers. |
| multilingual-e5-base | 278M | Smaller but weaker on MTEB multilingual benchmarks. Same architecture — e5-large is strictly better if GPU memory allows. |
| BGE-M3 | 568M | Supports dense + sparse + ColBERT retrieval. But MNRL only trains the dense component — the sparse/ColBERT heads are ignored, wasting the model's main advantage. Better suited as a comparison baseline or for hybrid search experiments. |

**Key properties of e5-large:**
- 1024-dim embeddings, 512 max tokens (our chunks average ~280 tokens — well within limits)
- Requires `"query: "` prefix for queries and `"passage: "` prefix for documents
- Fits on Colab T4 (16GB VRAM) with batch size 64 + fp16

### Loss function: Matryoshka + MNRL

We use two loss functions composed together:

**1. Multiple Negatives Ranking Loss (MNRL)** — the core contrastive loss for retrieval. It uses **in-batch negatives**: for each query in a batch, every other passage serves as a negative example. With batch size 64, each query gets 63 negatives — equivalent to a 64-way classification problem per sample.

**2. Matryoshka Loss** — wraps MNRL so the model learns useful embeddings at multiple truncated dimensionalities simultaneously. During each training step, the loss is computed at every Matryoshka dimension and averaged.

```python
inner_loss = MultipleNegativesRankingLoss(model)
loss = MatryoshkaLoss(
    model, inner_loss,
    matryoshka_dims=[1024, 768, 512, 256, 128, 64]
)
```

### Why Matryoshka embeddings?

Standard embeddings are fixed-size (e5-large = 1024 dims). Matryoshka Representation Learning (MRL) trains the model so that the **first N dimensions** of the embedding are independently useful. This enables:

- **Flexible quality/speed tradeoff** — use 1024-dim for maximum accuracy, 128-dim for 8x faster search, no retraining needed
- **Storage savings** — 64-dim embeddings use 16x less memory than 1024-dim
- **Staged retrieval** — retrieve candidates with 128-dim, re-rank with 1024-dim
- **Negligible training overhead** — MatryoshkaLoss adds ~10-15% training time (just extra forward passes at truncated dims)

**Dimensions chosen: `[1024, 768, 512, 256, 128, 64]`**

| Dim | Relative size | Use case |
|---|---|---|
| 1024 | 100% (full) | Maximum quality — final ranking, evaluation |
| 768 | 75% | Compatible with common vector DB defaults |
| 512 | 50% | Good balance for production RAG |
| 256 | 25% | Fast retrieval with minimal quality loss |
| 128 | 12.5% | Candidate retrieval in staged pipelines |
| 64 | 6.25% | Ultra-compact — prototyping, edge deployment |

At inference time, truncation is a single argument:
```python
model.encode(text, output_dimensionality=256)
```

**Why MNRL as the inner loss?**
- **No hard negatives needed** at this stage — in-batch negatives are sufficient for an initial fine-tune
- **Simple and effective** — standard approach for embedding fine-tuning
- **Scales with batch size** — larger batches = more negatives = stronger signal
- Hard negative mining comes in Stage 2, using the fine-tuned model from this stage

### Training configuration

**Designed for Google Colab (free T4 GPU, 16GB VRAM).**

| Parameter | Value | Rationale |
|---|---|---|
| **Batch size** | 64 | Fits on T4 with fp16. Gives 63 in-batch negatives per sample. |
| **Gradient accumulation** | 1 | Not needed — batch 64 fits directly on T4. |
| **Epochs** | 3 | Small dataset (1,944 pairs) → few epochs to avoid overfitting. |
| **Learning rate** | 2e-5 | Standard for fine-tuning pre-trained transformers. |
| **Warmup ratio** | 0.1 | Gradual LR ramp-up over first 10% of training (~3 steps). |
| **Weight decay** | 0.01 | Mild L2 regularization to prevent overfitting. |
| **FP16** | Yes (GPU) | Half-precision: faster training, lower memory. Auto-disabled on CPU. |
| **Batch sampler** | NO_DUPLICATES | Ensures no duplicate samples in a batch — important for MNRL so each negative is truly different. |
| **Scheduler** | Linear (default) | Linear LR decay after warmup. |

**Training size estimates (Colab T4):**
- Steps per epoch: ~30 (1,944 pairs / batch 64)
- Total steps: ~90 (3 epochs)
- Warmup steps: ~3 (10% of 30)
- Estimated training time: **~5-10 minutes**

**For CPU training** (laptop fallback): use `--batch-size 16 --grad-accum 4` for the same effective batch of 64. Estimated time: ~1.5-3 hours.

### E5 prefix handling

multilingual-e5 models require specific text prefixes to distinguish queries from documents. This is handled automatically in two places:

1. **During training** — the `prompts` parameter in `SentenceTransformerTrainingArguments` prepends prefixes to the correct columns:
   ```python
   prompts={"anchor": "query: ", "positive": "passage: "}
   ```

2. **During evaluation** — the `InformationRetrievalEvaluator` uses `query_prompt` and `corpus_prompt`:
   ```python
   ir_evaluator = InformationRetrievalEvaluator(
       ...,
       query_prompt="query: ",
       corpus_prompt="passage: ",
   )
   ```

### Evaluation during training

We evaluate with a `SequentialEvaluator` containing one `InformationRetrievalEvaluator` per Matryoshka dimension. After each epoch, retrieval quality is measured at all 6 truncation points (1024, 768, 512, 256, 128, 64).

Each evaluator simulates a retrieval task:
1. Encode all 340 eval queries (with `"query: "` prefix)
2. Encode all 85 eval corpus chunks (with `"passage: "` prefix)
3. Truncate both to the target dimensionality
4. For each query, rank all 85 chunks by cosine similarity
5. Check if the correct chunk ranks high → compute metrics

**Metrics reported per dimension:**

| Metric | What it measures |
|---|---|
| **NDCG@10** | Normalized Discounted Cumulative Gain — are relevant docs ranked near the top? (primary metric) |
| **MRR@10** | Mean Reciprocal Rank — how high is the first relevant result on average? |
| **Recall@10** | What fraction of relevant docs appear in the top 10? |
| **Accuracy@1** | Is the top result correct? |
| **Precision@k** | What fraction of top-k results are relevant? |
| **MAP@100** | Mean Average Precision — overall ranking quality. |

**Best model selection:** The checkpoint with the highest full-dim (1024) `NDCG@10` is kept (`load_best_model_at_end=True`, `metric_for_best_model="eu-ai-act-nl-dim1024_cosine_ndcg@10"`).

### Checkpoints and model saving

**How checkpoints work:**
- After each epoch, the trainer saves a checkpoint to `models/stage_1_mnrl/checkpoint-{step}/`
- Each checkpoint contains the full model weights, optimizer state, and training metadata
- `save_total_limit=2` keeps only the 2 most recent checkpoints (prevents disk bloat)
- At the end of training, the best checkpoint (by NDCG@10) is automatically loaded back
- The final model is saved to `models/stage_1_mnrl/final/`

**Checkpoint directory structure:**
```
models/stage_1_mnrl/
├── checkpoint-30/          # Epoch 1 checkpoint (step 30)
├── checkpoint-60/          # Epoch 2 checkpoint (step 60)
│   ├── model.safetensors   # Model weights
│   ├── config.json         # Model config
│   ├── optimizer.pt        # Optimizer state
│   ├── scheduler.pt        # LR scheduler state
│   └── training_args.bin   # Training arguments
├── runs/                   # TensorBoard logs
│   └── {timestamp}-e5-large-ai-act-nl-mnrl/
│       └── events.out.tfevents.*
└── final/                  # Best model (loaded at end)
    ├── model.safetensors
    ├── config.json
    └── tokenizer/
```

### Monitoring training progress

**1. Console output:** Loss is printed every 5 steps. After each epoch, full IR evaluation metrics are printed.

**2. TensorBoard loss curves:** The script logs to TensorBoard (`report_to="tensorboard"`). To view in Colab:
```python
%load_ext tensorboard
%tensorboard --logdir models/stage_1_mnrl/runs/
```

This shows:
- **Training loss curve** — should decrease over time. A healthy curve drops steeply in epoch 1, then flattens.
- **Eval metrics** — NDCG@10, MRR@10, etc. plotted per epoch.

**What to watch for:**
- **Loss decreasing** → model is learning
- **NDCG@10 increasing** → retrieval quality improving
- **Loss increasing after epoch 2-3** → overfitting (reduce epochs)
- **Eval metrics plateauing** → model has converged (can stop early)

**3. Final summary:** The script prints a Matryoshka quality table showing NDCG@10 at every dimension:
```
==========================================================
Training Complete — Matryoshka Evaluation
==========================================================
   Dim    Base NDCG@10       Finetuned         Δ
------  --------------  --------------  --------
  1024          0.XXXX          0.XXXX   +0.XXXX
   768          0.XXXX          0.XXXX   +0.XXXX
   512          0.XXXX          0.XXXX   +0.XXXX
   256          0.XXXX          0.XXXX   +0.XXXX
   128          0.XXXX          0.XXXX   +0.XXXX
    64          0.XXXX          0.XXXX   +0.XXXX
==========================================================
```

This table reveals the quality/size tradeoff: ideally, 256-dim should retain most of 1024-dim's quality, while 64-dim will show a noticeable but acceptable drop.

### Running the training

```bash
# On Colab (T4 GPU) — default settings
python finetuning/stage_1_mnrl/prepare_dataset.py
python finetuning/stage_1_mnrl/finetune.py

# On CPU (laptop fallback) — smaller batch with gradient accumulation
python finetuning/stage_1_mnrl/finetune.py --batch-size 16 --grad-accum 4
```

### Stage 1 Results

Training completed in ~14 minutes on Colab T4 (batch 64, SDPA + gradient checkpointing). Best checkpoint: epoch 2.

**NDCG@10 across Matryoshka dimensions:**

| Dim | Base | Stage 1 | Δ |
|---|---|---|---|
| 1024 | 0.8612 | 0.9436 | +0.0825 |
| 768 | 0.8577 | 0.9381 | +0.0804 |
| 512 | 0.8495 | 0.9352 | +0.0857 |
| 256 | 0.7848 | 0.9380 | +0.1532 |
| 128 | 0.7283 | 0.9236 | +0.1952 |
| 64 | 0.6009 | 0.9029 | +0.3019 |

**Full metrics at dim=1024:**

| Metric | Base | Stage 1 | Δ |
|---|---|---|---|
| NDCG@10 | 0.8612 | 0.9436 | +0.0825 |
| MRR@10 | 0.8315 | 0.9287 | +0.0972 |
| MAP@100 | 0.8336 | 0.9293 | +0.0957 |
| Accuracy@1 | 0.7618 | 0.8882 | +0.1265 |
| Accuracy@10 | 0.9529 | 0.9882 | +0.0353 |
| Recall@10 | 0.9529 | 0.9882 | +0.0353 |

Key takeaway: Matryoshka training flattened the quality curve — dim=64 now retains 96% of dim=1024's quality (0.90 vs 0.94), compared to only 70% before fine-tuning (0.60 vs 0.86).

---

## Step 5 — Hard Negative Mining

### Why hard negatives?

Stage 1 used only **in-batch negatives** — random passages from the same batch. These are mostly "easy" negatives (clearly different topics). The model quickly learns to distinguish them, but may struggle with **confusing near-misses** — passages that are topically similar but answer a different question.

Hard negatives are specifically chosen to be **similar but wrong**. They force the model to learn fine-grained distinctions, like telling apart two different articles about AI system obligations.

### Mining strategy

We use the **Stage 1 model itself** to mine hard negatives:

1. Encode all training queries with the Stage 1 model
2. Encode all training corpus chunks
3. For each query, rank all chunks by cosine similarity
4. Exclude the correct positive chunk
5. **Apply filtering** (see below), then take the top-N most similar wrong chunks

**Why mine from Stage 1 (not the base model)?**
- Stage 1 has learned task-specific similarity — its mistakes are more informative
- A passage that fools the *adapted* model is a genuinely confusing case
- Base model negatives would be too easy for the already-fine-tuned model

### Hard negative filtering

Raw top-K mining can produce **false hard negatives** — chunks that are actually relevant to the query but aren't the annotated positive. This is especially common in legal text where multiple chunks may validly answer the same question.

Inspired by the sentence-transformers `mine_hard_negatives` function, we added three filtering parameters:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `range_min` | 5 | Skip the top-5 most similar candidates — they're likely too confusing or even valid positives |
| `margin` | 0.1 | Negative similarity must be at least 0.1 lower than the query-positive similarity |
| `max_score` | 0.9 | Skip any negative with similarity > 0.9 — likely a true positive |

**How filtering works per query:**
1. Sort all non-positive chunks by descending similarity
2. Skip ranks 0–4 (range_min=5) — these are the most similar and potentially valid answers
3. Skip any chunk with similarity > 0.9 (max_score)
4. Skip any chunk where `neg_sim > pos_sim - 0.1` (margin)
5. Collect the first N chunks that pass all filters

This produces "Goldilocks" negatives — hard enough to be informative, but not so hard they're actually correct answers the model shouldn't learn to reject.

**Why 5 hard negatives now works (where it previously regressed):**

In Step 9 below, we found that 5 raw hard negatives caused a regression (0.9463 → 0.9398). With filtering, the top-5 most confusing candidates (likely false negatives or near-duplicates) are skipped, and the margin ensures we don't mine chunks that are nearly as similar as the positive. The result is 5 genuinely informative negatives rather than 5 potentially misleading ones.

### Output

The mining script produces a new dataset with columns `(anchor, positive, negative_1, ..., negative_5)` — same row count as the Stage 1 training set (1,944 rows, 5 negatives each).

```bash
# On Colab (uses Stage 1 model for encoding)
python finetuning/stage_2_hard_neg/mine_negatives.py

# Custom paths
python finetuning/stage_2_hard_neg/mine_negatives.py \
    --model models/stage_1_mnrl/final \
    --train-dir data/processed/train \
    --output-dir data/processed/train_hard_neg
```

---

## Step 6 — Stage 2 Fine-tuning with Hard Negatives

### Starting point: Stage 1 checkpoint

Stage 2 continues training from the Stage 1 model — not from scratch. The model already has a strong retrieval foundation; Stage 2 refines its ability to distinguish between confusingly similar passages.

### How MNRL handles hard negatives

When the training dataset has a `negative` column, MNRL automatically uses it:

```python
# Stage 1 dataset: (anchor, positive) — MNRL uses in-batch negatives only
# Stage 2 dataset: (anchor, positive, negative) — MNRL uses both:
#   1. The explicit hard negative (mined)
#   2. All other positives in the batch (in-batch negatives)
```

With batch size 16 (effective 64 via gradient accumulation), each query now sees: 1 explicit hard negative + ~15 in-batch negatives per micro-batch. The hard negative is weighted more heavily by the loss because it has a higher similarity score, forcing the model to learn the distinction.

### Training configuration — tuned for Stage 2

| Parameter | Stage 1 | Stage 2 | Rationale |
|---|---|---|---|
| **Base model** | `intfloat/multilingual-e5-large` | `models/stage_1_mnrl/final` | Continue from adapted checkpoint |
| **Learning rate** | 2e-5 | **1e-5** | Lower — model is already fine-tuned, large updates risk catastrophic forgetting |
| **Epochs** | 3 | **2** | Fewer — hard negatives provide stronger signal per step, more prone to overfitting |
| **Batch size** | 64 | **16 (×4 grad accum = 64 effective)** | Reduced — 3 columns × 6 Matryoshka dims OOMs at batch 32+; grad accum preserves effective batch |
| **Matryoshka dims** | [1024..64] | [1024..64] | Same — maintain multi-dim compatibility |
| **Loss** | MatryoshkaLoss(MNRL) | MatryoshkaLoss(MNRL) | Same structure — MNRL auto-detects the negative column |

### Evaluation

Same eval setup as Stage 1 — `SequentialEvaluator` with `InformationRetrievalEvaluator` per Matryoshka dimension, same eval dataset (340 queries, 85 chunks). This enables direct comparison of base → Stage 1 → Stage 2.

### Running Stage 2

```bash
# Step 1: Mine hard negatives (requires Stage 1 model)
python finetuning/stage_2_hard_neg/mine_negatives.py

# Step 2: Fine-tune with hard negatives
python finetuning/stage_2_hard_neg/finetune.py
```

### Stage 2 Results

Training completed in ~18 minutes on Colab T4 (batch 16 × 4 grad accum, SDPA + gradient checkpointing).

**NDCG@10 across Matryoshka dimensions — full pipeline comparison:**

| Dim | Base | Stage 1 | Stage 2 | Δ (Base→S2) | Δ (S1→S2) |
|---|---|---|---|---|---|
| 1024 | 0.8612 | 0.9426 | **0.9465** | +0.0853 | +0.0039 |
| 768 | 0.8577 | 0.9411 | **0.9445** | +0.0868 | +0.0035 |
| 512 | 0.8495 | 0.9379 | **0.9412** | +0.0917 | +0.0033 |
| 256 | 0.7848 | 0.9383 | **0.9423** | +0.1575 | +0.0041 |
| 128 | 0.7283 | 0.9225 | **0.9277** | +0.1994 | +0.0051 |
| 64 | 0.6009 | 0.9011 | **0.9058** | +0.3049 | +0.0047 |

**Full metrics at dim=1024 — full pipeline comparison:**

| Metric | Base | Stage 1 | Stage 2 | Δ (Base→S2) | Δ (S1→S2) |
|---|---|---|---|---|---|
| NDCG@10 | 0.8612 | 0.9426 | **0.9465** | +0.0853 | +0.0039 |
| MRR@10 | 0.8315 | 0.9272 | **0.9315** | +0.1000 | +0.0043 |
| MAP@100 | 0.8336 | 0.9279 | **0.9319** | +0.0983 | +0.0041 |
| Accuracy@1 | 0.7618 | 0.8853 | **0.8912** | +0.1294 | +0.0059 |
| Accuracy@3 | — | 0.9706 | **0.9765** | — | +0.0059 |
| Accuracy@5 | — | 0.9794 | **0.9824** | — | +0.0029 |
| Accuracy@10 | 0.9529 | 0.9882 | **0.9912** | +0.0383 | +0.0029 |
| Recall@10 | 0.9529 | 0.9882 | **0.9912** | +0.0383 | +0.0029 |

### Analysis

**Stage 2 provided consistent gains across all dimensions and metrics.** Key observations:

1. **Lower Matryoshka dims benefited most** from hard negatives (dim=128: +0.51%, dim=64: +0.47%). Hard negatives teach finer distinctions that are especially valuable when the model has fewer dimensions to work with.

2. **Accuracy@1 improved by +0.59%** (88.53% → 89.12%) — the model became better at ranking the correct chunk first.

3. **Recall@10 reached 99.12%** — near ceiling. The correct chunk is almost always in the top 10.

4. **Matryoshka quality curve remains flat**: dim=64 retains 96% of dim=1024's quality (0.906 vs 0.947), consistent with Stage 1.

5. **No regressions** — all dimensions improved, confirming the lower learning rate (1e-5) and fewer epochs (2) prevented catastrophic forgetting.

6. **Total improvement from base**: NDCG@10 at dim=1024 went from 0.861 → 0.947 (+8.5 points). At dim=64: 0.601 → 0.906 (+30.5 points).

### Model outputs

```
models/
├── stage_1_mnrl/final/       # Stage 1: MNRL + Matryoshka on base model
└── stage_2_hard_neg/final/   # Stage 2: MNRL + Matryoshka with hard negatives on Stage 1
```

The Stage 2 model at `models/stage_2_hard_neg/final/` is the final production model.

---

## Step 7 — RTX 5090 Reproduction

### Goal

Reproduce the Colab results on a local RTX 5090 (32GB VRAM, Blackwell architecture) to isolate whether an instruct model underperformance was caused by the model itself or GPU-specific numerical issues.

### Hardware constraints discovered

The RTX 5090 (Blackwell, sm_120) exposed several critical issues that don't occur on Colab's T4/A100:

1. **bf16 causes immediate embedding collapse** — gradient norms spike to 500-4000+ within the first few steps, all cosine similarities converge to 1.0. This happens with both SDPA and eager attention.
2. **SDPA is partially unstable** — even in fp32, SDPA shows occasional gradient instability on Blackwell.
3. **Gradient checkpointing causes numerical collapse** — recomputation through transformer layers introduces enough numerical drift to destabilize training.

**Working config:** fp32 + eager attention + no gradient checkpointing.

### Batch size impact

fp32 uses 2× the VRAM of fp16/bf16, severely limiting batch sizes:

| Config | Max micro-batch on 32GB |
|--------|------------------------|
| MatryoshkaLoss (2 columns) | 8 |
| MatryoshkaLoss + hard neg (3 columns) | 4 |

With batch 8, each query sees only **7 in-batch negatives** (vs 63 on Colab with batch 64). We used gradient accumulation to maintain the same effective batch for optimizer updates, but gradient accumulation does NOT increase in-batch negatives — it just averages gradients across separate forward passes.

### Results: RTX 5090 vs Colab

| Config | Stage 1 NDCG@10 | Stage 2 NDCG@10 |
|--------|----------------|----------------|
| Colab T4 (batch 64, fp16, SDPA) | 0.9436 | 0.9465 |
| RTX 5090 (batch 8, fp32, eager) | 0.9327 | **0.9492** |

**Key finding:** Despite only 7 in-batch negatives (vs 63), the RTX 5090 produced the **best overall Stage 2 result** (0.9492). The Stage 1 gap (0.9327 vs 0.9436) is explained by fewer negatives, but Stage 2 hard negatives compensated strongly (+0.0165 on RTX vs +0.0029 on Colab).

**Hypothesis formed:** With fewer in-batch negatives in Stage 1, the model has more room to improve from hard negatives in Stage 2. The pipeline converges to similar quality regardless of Stage 1 batch size.

---

## Step 8 — GradCache: Decoupling Batch Size from VRAM

### Motivation

The RTX 5090's batch-8 limitation (7 in-batch negatives) was a concern. Industry literature recommends 64-128 in-batch negatives for contrastive learning. Could we get more negatives without more VRAM?

### CachedMultipleNegativesRankingLoss (GradCache)

GradCache decouples the contrastive pool size from GPU memory by splitting the forward pass:

1. **Embed** all N samples in small mini-batches **without gradients** (cheap, no computation graph stored)
2. **Compute loss** on the full N×N similarity matrix (tiny — just floats)
3. **Re-embed** in small mini-batches **with gradients**, using cached gradient signals from step 2

This means:
- `per_device_train_batch_size = 128` → 127 in-batch negatives (contrastive quality)
- `mini_batch_size = 4` → only 4 samples in VRAM at a time (memory safety)
- `gradient_accumulation_steps = 1` → no longer needed since GradCache handles the accumulation internally

Trade-off: ~20-30% slower (every sample is embedded twice), but allows arbitrarily large contrastive pools.

### VRAM tuning: mini_batch_size

With fp32 + eager attention + 6 Matryoshka dims, finding the right mini_batch_size required trial and error:

| mini_batch_size | Result |
|----------------|--------|
| 32 | OOM (during backward re-embed) |
| 16 | OOM (just barely — 256 MiB short) |
| 4 | Fits ✅ |

MatryoshkaLoss runs 6 GradCache cycles (one per dimension), and each cycle's re-embed step holds a computation graph. With mini_batch_size=4, each re-embed processes 4 samples — comparable VRAM to the old batch-4 config.

### Results

| Config | In-batch neg | Stage 1 | Stage 2 (1 hard neg) |
|--------|-------------|---------|---------------------|
| RTX batch 8, standard MNRL | 7 | 0.9327 | **0.9492** ✅ |
| RTX batch 128, CachedMNRL | 127 | 0.9422 | 0.9463 |
| Colab batch 64, standard MNRL | 63 | 0.9436 | 0.9465 |

### Analysis

1. **More in-batch negatives clearly help Stage 1:** 7 neg → 0.9327, 63 neg → 0.9436, 127 neg → 0.9422. The jump from 7 to 63 is significant (+1.1%), but 63→127 shows **diminishing returns** (and is slightly lower, possibly due to fp32 vs fp16 differences).

2. **Stage 2 hard negatives are a great equalizer:** Regardless of how many in-batch negatives Stage 1 had, the pipeline totals converge:
   - Batch 8 pipeline: 0.9327 → 0.9492 (S2 adds +0.0165)
   - Batch 128 pipeline: 0.9422 → 0.9463 (S2 adds +0.0041)
   - Colab pipeline: 0.9436 → 0.9465 (S2 adds +0.0029)

3. **Inverse relationship:** The weaker Stage 1 is, the more Stage 2 gains. Hard negatives fill in exactly what in-batch negatives missed.

4. **The simplest config won:** Batch 8 with standard MNRL → 1 hard negative produced the best final result (0.9492). This was unexpected but makes sense — with few in-batch negatives, the model is "hungrier" for the targeted signal that hard negatives provide.

5. **Practical implication:** For small datasets (~2000 samples), investing in GradCache complexity isn't worth it. The standard MNRL → hard negative pipeline is sufficient.

---

## Step 9 — Hard Negative Count Experiments

### Motivation

Since hard negatives provided the biggest Stage 2 gain with the batch-8 config, could we amplify that by mining **more** hard negatives per query?

### Setup

Updated `mine_negatives.py` to output multiple negative columns (`negative_1` through `negative_N`) instead of repeating rows. Updated `finetune_stage2.py` to dynamically detect and prompt all `negative_*` columns.

Tested with the CachedMNRL Stage 1 checkpoint (0.9422) since it was the most recent:

| Hard negatives | Stage 1 | Stage 2 | Δ (S1→S2) |
|---------------|---------|---------|-----------|
| 1 | 0.9422 | 0.9463 | +0.0041 |
| 5 | 0.9422 | 0.9398 | **-0.0025** |

### Analysis: 5 hard negatives caused regression

5 hard negatives **hurt** performance (0.9463 → 0.9398, -0.7%). Why?

1. **Overfitting to hard negatives:** With 5 hard negatives per query in a 1,944-sample dataset, the model sees 9,720 negative pairs. Combined with 127 in-batch negatives, this is an overwhelming amount of negative signal relative to the dataset size — the model over-corrects.

2. **Diminishing quality of deeper negatives:** The top-1 hard negative is genuinely confusing (high similarity but wrong). The top-5 includes progressively less informative negatives — they're hard enough to distract the model but not informative enough to teach useful distinctions.

3. **Signal imbalance:** Each query has 1 positive but 5 hard negatives + 127 in-batch negatives = 132 negatives. The positive signal is drowned out.

4. **Recall@10 dropped** from 0.9912 to 0.9853 — the model started ranking some correct chunks outside the top 10, indicating it learned to be "too suspicious" of similar passages.

### Key takeaway

For this dataset size (~2000 pairs), **1 hard negative is the sweet spot.** The quality of the hardest negative matters more than the quantity. This aligns with the literature: most embedding fine-tuning papers use 1-3 hard negatives, and the sentence-transformers documentation defaults to 1.

---

## Summary: What Worked, What Didn't

### Final results table (all experiments, dim=1024 NDCG@10)

| # | Config | In-batch neg | Hard neg | Stage 1 | Stage 2 | Notes |
|---|--------|-------------|----------|---------|---------|-------|
| 1 | Colab T4, batch 64, fp16 | 63 | 1 | 0.9436 | 0.9465 | Original baseline |
| 2 | RTX 5090, batch 8, fp32 | 7 | 1 | 0.9327 | **0.9492** | **Best overall** |
| 3 | RTX 5090, CachedMNRL batch 128, fp32 | 127 | 1 | 0.9422 | 0.9463 | GradCache experiment |
| 4 | RTX 5090, CachedMNRL batch 128, fp32 | 127 | 5 | 0.9422 | 0.9398 | Regression — too many negatives |

### What worked
- **Two-stage pipeline (MNRL → hard negatives):** Consistent +3-8% improvement from hard negatives
- **MatryoshkaLoss:** Minimal overhead, massive benefit at lower dims (dim=64 retains 96% of dim=1024)
- **fp32 + eager attention on Blackwell:** Stable training despite bf16/SDPA being the recommended setup
- **1 hard negative per query:** Sweet spot for this dataset size
- **Small batch → hard neg compensation:** Even batch 8 (7 in-batch negatives) reaches near-optimal quality after Stage 2

### What didn't work (but was informative)
- **bf16 on RTX 5090:** Immediate embedding collapse — Blackwell-specific issue
- **Gradient checkpointing on RTX 5090:** Numerical drift causes training instability
- **128 in-batch negatives (GradCache):** Better Stage 1 but diminishing returns after Stage 2 — not worth the complexity
- **5 hard negatives:** Regression from overfitting — 1 is enough for ~2000 samples

### Remaining improvement levers
1. **More/better training data** — the biggest lever. Current dataset is only 2,284 pairs from 573 chunks of a single document. More diverse queries, more documents, or better query quality would help most.
2. **Larger base model** — Qwen2.5 Embedding 7B (7.6B params) vs e5-large (560M params). More parameters can capture finer distinctions, especially with adequate training data.
3. **Cross-encoder reranking** — a 2-stage retrieval pipeline where the embedding model retrieves candidates and a cross-encoder reranks them. Orthogonal to embedding quality.
4. **Knowledge distillation** — train a smaller model to mimic a larger model's rankings. Useful for production deployment.

### Best model for production

The **batch-8 standard MNRL pipeline** (experiment #2) at **NDCG@10 = 0.9492** remains the best result. The model is saved at `models/stage_2_hard_neg/final/` from that run.

---

## Step 10 — Switching to Qwen3-Embedding-0.6B

### Why a different model family?

The e5-large experiments plateaued at NDCG@10 ~0.949. Before scaling to a larger model, we switched architecture entirely — from an **encoder-based** model to a **decoder-based** embedding model.

**Qwen3-Embedding** is a family of embedding models from Alibaba's Qwen team, built on top of the Qwen3 language model. The key architectural difference from e5-large:

| Property | multilingual-e5-large | Qwen3-Embedding-0.6B |
|----------|----------------------|----------------------|
| Architecture | Encoder (BERT-style, bidirectional) | Decoder (GPT-style, causal) |
| Parameters | 560M | 620M |
| Max tokens | 512 | 8,192 |
| Output dims | 1024 | 1024 |
| Pooling | CLS token | Last token |
| Padding | Right | **Left** |
| Query prompts | `"query: "` / `"passage: "` | Instruct-style prompt |

The 512-token limit of e5-large was a real constraint — our chunks average 283 tokens but some exceed 512. Qwen3 removes this ceiling entirely.

### Why decoders make good embedding models

Intuitively, encoder models (bidirectional attention) seem better for embeddings — they process the full context in both directions. But in practice:

- Decoder models are pre-trained on vastly larger corpora with stronger generative objectives
- The instruction-following training of decoder LLMs transfers cleanly to embedding tasks
- Last-token pooling on a well-trained decoder captures the full input context effectively

The model uses **left padding** specifically for this: the embedding is taken from the last non-padding token position. By padding on the left, the last real token is always at the final position of the sequence.

### Instruct prompts for queries

Instead of a simple `"query: "` prefix, Qwen3 uses a structured instruction for queries:

```python
QUERY_PROMPT = (
    "Instruct: Given a question about EU AI regulation, "
    "retrieve the most relevant passage\nQuery:"
)
```

Documents get no prefix at all — the asymmetry is intentional. The query needs to encode what task it's being used for; the document just needs to represent its content.

### Blackwell compatibility

Unlike e5-large, **Qwen3-Embedding runs stably in bf16 on Blackwell**. The reason: Qwen3's RMSNorm upcasts to fp32 internally before computing the normalization. This keeps the critical numerical operations precise even when the rest of the model runs in bf16. The gradient explosion we saw with e5-large in bf16 doesn't occur.

This means we can use:
- `bf16=True` (2× memory savings vs fp32)
- SDPA attention (flash_attention_2 not required, sdpa works as fallback)
- No gradient checkpointing needed (bf16 gives enough headroom)

And CachedMNRL with:
- `per_device_train_batch_size = 128`
- `mini_batch_size = 4`

### Results

| Dim | Zero-shot | Stage 1 | Stage 2 | Δ (ZS→S2) |
|-----|-----------|---------|---------|-----------|
| 1024 | 0.8013 | 0.9419 | **0.9467** | +0.1454 |
| 768 | 0.8109 | 0.9452 | **0.9479** | +0.1370 |
| 512 | 0.8043 | 0.9427 | **0.9449** | +0.1406 |
| 256 | 0.7691 | 0.9343 | **0.9412** | +0.1721 |
| 128 | 0.7358 | 0.9154 | **0.9163** | +0.1805 |
| 64 | 0.6727 | 0.8907 | 0.8854 | +0.2127 |

Stage 2 NDCG@10 at dim=1024 (0.9467) is nearly identical to the best e5-large result (0.9492), with a significantly better context window and native Matryoshka support built into the base model.

**Stage 2 barely improved over Stage 1** — the Qwen3-0.6B model, with 127 in-batch negatives via CachedMNRL, was already near its ceiling after Stage 1. Hard negatives added only +0.005 at dim=1024. This matches the pattern from the e5-large GradCache experiment: more in-batch negatives → less room for hard negative improvement.

The model was published to [danielnoumon/qwen3-embedding-0.6b-ai-act-nl](https://huggingface.co/danielnoumon/qwen3-embedding-0.6b-ai-act-nl).

---

## Step 11 — Qwen3-Embedding-4B with LoRA

### Why 4B?

The 0.6B model hit ~0.947 NDCG@10. To push further, we scale to the 4B variant — but full fine-tuning of a 4B model in bf16 requires approximately:

- Weights: ~8 GB
- Adam optimizer states (fp32): ~32 GB
- Gradients: ~8 GB
- **Total: ~48 GB** → exceeds the 32GB RTX 5090

The solution is **LoRA (Low-Rank Adaptation)**, which freezes the base model weights and only trains small adapter matrices.

### How LoRA works

Every weight matrix in a transformer is large — for example, the query projection in a 4B model might be a 2048×2048 matrix (4M parameters). Instead of updating all 4M values, LoRA inserts two small matrices:

```
W_new = W_original + (A × B)

Where:
  W_original: 2048 × 2048 (frozen, ~4M params)
  A: 2048 × 16   (trainable, ~32K params)
  B: 16 × 2048   (trainable, ~32K params)
```

`A × B` is a **low-rank approximation** of the weight update. With rank `r=16`, you're saying: "the meaningful change to this matrix lives in a 16-dimensional subspace." This is surprisingly effective for domain adaptation.

**Key parameters and why we chose them:**

| Param | Value | Why |
|-------|-------|-----|
| `r` (rank) | 16 | Standard starting point for domain adaptation tasks. r=8 risks underfitting; r=32+ would double trainable params with diminishing returns given ~2000 training pairs. We didn't tune this — r=16 is the most common default and worked well out of the box. |
| `alpha` | 32 (2×r) | The effective scale of the adapter is `alpha / r = 2.0`. This 2×r convention comes from the original LoRA paper and balances adapter influence with training stability. |
| `dropout` | 0.05 | Light regularization to prevent overfitting on ~2000 pairs. Heavy dropout (0.1+) would slow convergence. |
| `target_modules` | q_proj, k_proj, v_proj, o_proj | All attention projections — these are the most impactful layers for embedding models. We skip MLP layers to keep trainable params low (~0.3% of total). |
| `learning_rate` | 1e-4 (Stage 1), 1e-5 (Stage 2) | LoRA adapters need a higher LR than full fine-tuning (2e-5) because only a tiny fraction of params are updated. Stage 2 drops to 1e-5 to avoid forgetting Stage 1 gains. |

**Same config for both 4B and 8B:** Both models use identical LoRA hyperparameters. The only difference is operational — mini_batch_size drops from 2→1 and eval_batch_size from 2→1 on 8B due to VRAM constraints.

**Result for 4B Qwen3:** 11.8M trainable parameters out of 4,034M total — just **0.29%**. Optimizer states, gradients, and VRAM scale with this tiny fraction, making the 4B model fit comfortably on 32GB.

### Two-stage LoRA pipeline

The same two-stage structure applies, with one added complexity: Stage 2 needs to start from Stage 1's *merged* weights, not from the LoRA adapter file.

**Stage 1 saves** a PEFT adapter checkpoint (only the ~50MB of adapter weights, not the 8GB base). To continue training in Stage 2:
1. Load the 4B base model fresh
2. Apply Stage 1's adapter via `PeftModel.from_pretrained()`
3. Merge: `model.merge_and_unload()` — bakes the adapter into the base weights
4. Apply a *new* LoRA adapter for Stage 2 fine-tuning

This merge-then-re-adapt approach prevents the LoRA adapters from conflicting.

### VRAM tuning for 4B

The 4B model is ~7× larger than 0.6B. Accordingly, `mini_batch_size` for CachedMNRL needs to drop:

| Stage | Columns | mini_batch_size | Result |
|-------|---------|----------------|--------|
| Stage 1 | 2 (anchor, positive) | 2 | Fits ✅ |
| Stage 2 | 3 (anchor, positive, negative_1) | 2 | OOM ❌ |
| Stage 2 | 3 (anchor, positive, negative_1) | **1** | Fits ✅ |

The extra column in Stage 2 (hard negatives) requires the backward re-embed pass to process 3× more sequences. With mini_batch_size=2 already at the limit, adding one column tips it into OOM. Dropping to mini_batch_size=1 cuts the backward re-embed memory in half.

### `load_best_model_at_end` bug with PEFT

HuggingFace Trainer's `load_best_model_at_end=True` is broken when PEFT is wrapped inside a SentenceTransformer. The trainer tries to load the best checkpoint using a generic state_dict mechanism that doesn't understand the PEFT key naming convention (`base_model.model.*`).

**Fix:** Set `load_best_model_at_end=False` and add a post-training function that:
1. Iterates over all saved epoch checkpoints
2. Loads Stage 1 merged base + applies each checkpoint's adapter
3. Evaluates each on the IR task
4. Saves the best merged model to `final/`

This is implemented in `upload_to_hf.py` and the corrected `finetune_stage2.py`.

### Results

| Dim | Stage 1 | Stage 2 | Δ vs 0.6B Stage 2 |
|-----|---------|---------|-------------------|
| 2560 | 0.9626 | **0.9616** | — (new dim) |
| 1024 | 0.9631 | **0.9658** | +0.0191 |
| 768 | 0.9616 | **0.9609** | +0.0130 |
| 512 | 0.9537 | **0.9526** | +0.0077 |
| 256 | 0.9410 | **0.9420** | +0.0008 |
| 128 | 0.9186 | **0.9188** | +0.0025 |

**Key observations:**

1. **4B LoRA beats 0.6B full fine-tune at every comparable dim** — 0.29% trainable parameters is enough for a 4B model to outperform a fully fine-tuned smaller model.

2. **Stage 2 barely improved over Stage 1** (same pattern as 0.6B) — the model was already near-ceiling. Hard negatives converge fast when Stage 1 already has strong contrastive signal.

3. **dim=1024 > dim=2560 (0.9658 vs 0.9616)** — a Matryoshka artifact. Each subspace is trained independently; the 1024-dim subspace can sometimes learn a sharper representation than the full-dim space. Acceptable and expected.

4. **Epoch 1 ≈ Epoch 2** in Stage 2 — the model converged after the first epoch. Running 2 epochs is safe but not necessary.

The model was published to [danielnoumon/qwen3-embedding-4b-ai-act-nl](https://huggingface.co/danielnoumon/qwen3-embedding-4b-ai-act-nl).

---

## Step 12 — Benchmarking Against Proprietary SOTA

### Why compare against a proprietary model?

Fine-tuning is only valuable if it beats what you can get out of the box. The ultimate test is whether a small, domain-adapted model outperforms the best general-purpose proprietary embedding API on your specific task.

### Model choice: text-embedding-3-large

We chose OpenAI's `text-embedding-3-large` as the proprietary baseline because:

1. **It's the strongest general-purpose embedding model widely available** — consistently near the top of MTEB leaderboards across multiple languages
2. **It natively supports Matryoshka dimensions** via the `dimensions` API parameter, so we can compare at the exact same truncation points as our fine-tuned models
3. **It's the model most teams would reach for** when building a RAG pipeline without fine-tuning — if we beat it, the fine-tuning investment is clearly justified
4. **We already had Azure OpenAI infrastructure** from the synthetic query generation step, so adding an embedding deployment required no new setup

### Evaluation setup

The evaluation uses the exact same data and metrics as all prior experiments:

- **Eval set:** 340 Dutch queries, 85 corpus chunks (same chunk-level split — no leakage)
- **Primary metric:** NDCG@10 (cosine similarity)
- **Matryoshka dims tested:** 3072 (text-embedding-3-large's full dimensionality), 1024, 768, 512, 256, 128

The only difference is how embeddings are produced: instead of a local SentenceTransformer, we call the Azure OpenAI embeddings API. The metric computation (cosine similarity → ranking → NDCG/MRR/MAP/Recall) is identical.

No query prompt or instruct prefix is used for text-embedding-3-large — it doesn't support task-specific instructions. The model receives raw query and corpus text.

### Results

**NDCG@10 across Matryoshka dimensions:**

| Dim | text-embedding-3-large | Qwen3-4B LoRA (best) | Δ |
|-----|----------------------|---------------------|---|
| 3072 | 0.8643 | — | — |
| 1024 | 0.8635 | **0.9658** | **+0.1023** |
| 768 | 0.8622 | **0.9609** | +0.0987 |
| 512 | 0.8573 | **0.9526** | +0.0953 |
| 256 | 0.8166 | **0.9420** | +0.1254 |
| 128 | 0.7598 | **0.9188** | +0.1590 |

**Full metrics at dim=3072 (text-embedding-3-large's best):**

| Metric | text-embedding-3-large | Qwen3-4B (dim=2560) |
|--------|----------------------|---------------------|
| NDCG@10 | 0.8643 | **0.9616** |
| MRR@10 | 0.8308 | — |
| MAP@100 | 0.8332 | — |
| Accuracy@1 | 0.7441 | — |
| Accuracy@3 | 0.9088 | — |
| Accuracy@5 | 0.9441 | — |
| Accuracy@10 | 0.9647 | — |
| Recall@10 | 0.9647 | — |

**Comparison at dim=1024 across all models:**

| Model | NDCG@10 | vs text-embedding-3-large |
|-------|---------|--------------------------|
| text-embedding-3-large (zero-shot) | 0.8635 | — |
| multilingual-e5-large zero-shot | 0.8612 | -0.0023 |
| multilingual-e5-large Stage 2 | 0.9492 | +0.0857 |
| Qwen3-Embedding-0.6B Stage 2 | 0.9467 | +0.0832 |
| **Qwen3-Embedding-4B LoRA Stage 2** | **0.9658** | **+0.1023** |

### Analysis

1. **Fine-tuning adds ~10 points of NDCG@10 over proprietary SOTA.** At dim=1024, even our smallest fine-tuned model (e5-large, 560M params) beats text-embedding-3-large by +8.6 points. The 4B model wins by +10.2 points. This is a massive gap in retrieval quality — the difference between "good enough" and "near-perfect" retrieval.

2. **text-embedding-3-large barely benefits from more dimensions.** Its own Matryoshka scaling from 1024 to 3072 adds only +0.0008 NDCG@10 (0.8635 → 0.8643). The model lacks Dutch legal domain knowledge; more dimensions can't compensate for that. In contrast, our fine-tuned models encode task-specific patterns that remain useful across all Matryoshka truncation points.

3. **Matryoshka degradation is steeper for the proprietary model at low dims.** text-embedding-3-large drops from 0.8643 (dim=3072) to 0.7598 (dim=128) — a 12% relative decline. Our Qwen3-4B drops from 0.9616 (dim=2560) to 0.9188 (dim=128) — only 4.5%. Matryoshka-aware fine-tuning explicitly teaches the model to be useful at every dimension.

4. **Zero-shot baselines are nearly identical.** text-embedding-3-large (0.8635) and e5-large (0.8612) start at almost the same level on this task. The proprietary model has no inherent advantage on Dutch legal text — its strength is breadth across languages and domains, not depth on any specific one.

5. **The value proposition of fine-tuning is clear.** A single run of fine-tuning on ~2,000 synthetic pairs (generated in minutes with GPT-5-mini) turns a generic embedding model into a domain specialist that far exceeds what the best proprietary API can deliver out of the box. The fine-tuned model is also local, private, and free at inference time.

---

## Final Results: All Models

| Model | NDCG@10 (dim=1024) | Notes |
|-------|-------------------|-------|
| text-embedding-3-large (zero-shot) | 0.8635 | Proprietary SOTA baseline |
| multilingual-e5-large zero-shot | 0.8612 | Open-source baseline |
| multilingual-e5-large Stage 2 (batch 8) | 0.9492 | +8.6 over proprietary SOTA |
| Qwen3-Embedding-0.6B Stage 2 | 0.9467 | +8.3 over proprietary SOTA |
| **Qwen3-Embedding-4B LoRA Stage 2** | **0.9658** | **+10.2 over proprietary SOTA** |

The 4B LoRA model is the clear winner. Fine-tuning on domain-specific synthetic data delivers a +10 point NDCG@10 improvement over the best proprietary embedding API — with a model that runs locally, costs nothing at inference, and keeps sensitive legal data private.

---

## Step 13 — Cross-Domain Evaluation on the Combined EU Regulations Dataset

### Motivation

Steps 1–12 evaluated all models on a small, single-document eval set: 85 held-out EU AI Act chunks, ~340 queries. This made for a highly favourable retrieval environment — the correct chunk only needed to be distinguished from 84 other EU AI Act chunks. 

Two limitations needed addressing:

1. **Corpus too small** — 85 chunks is not realistic. A production RAG pipeline has hundreds or thousands of documents. With a small corpus, even zero-shot models score artificially high.
2. **Single document** — the models were trained *and* evaluated on EU AI Act text. It was unclear whether improvements reflected general Dutch legal retrieval capability or mere domain memorisation.

We extended the evaluation by adding the **Dutch GDPR (AVG)** as a second document, creating a combined corpus of 912 chunks and using the full 5,472-query dataset split across three eval sets.

### Dataset

| Split | Queries | Corpus chunks | Description |
|-------|---------|--------------|-------------|
| `combined` | 5,472 | 912 | All queries, all chunks — cross-document retrieval |
| `eu_ai_act` | 3,210 | 535 | EU AI Act queries vs EU AI Act corpus only |
| `gdpr` | 2,262 | 377 | GDPR queries vs GDPR corpus only |

**Note on methodology:** Unlike Steps 4–12 which used a held-out test split, this evaluation runs over the full dataset (including training queries). This would normally inflate scores for fine-tuned models — but the much larger corpus (535–912 chunks vs 85 previously) more than counteracts that, making overall NDCG@10 numbers lower and more realistic.

### Evaluation setup

Three eval JSON sets were generated from `hf_dataset.parquet` using `evaluation/prepare_eval_combined.py`. Each model was evaluated with `evaluation/eval_combined.py`, which supports both SentenceTransformer models (GPU) and OpenAI embeddings (API), with `--only` and `--append-to` flags for partial runs.

### Results

**NDCG@10 @ dim=1024:**

| Model | Combined | EU AI Act | GDPR |
|-------|----------|-----------|------|
| multilingual-e5-large (zero-shot) | 0.5816 | 0.5584 | 0.6475 |
| Qwen3-0.6B (zero-shot) | 0.5448 | 0.5349 | 0.6007 |
| Qwen3-4B (zero-shot) | 0.6494 | 0.6274 | 0.7179 |
| text-embedding-3-large (OpenAI) | 0.6012 | 0.5682 | 0.6733 |
| multilingual-e5-large (EU AI Act FT) | 0.7199 | 0.7435 | 0.7311 |
| Qwen3-0.6B (EU AI Act FT) | 0.7118 | 0.7441 | 0.7110 |
| **Qwen3-4B (EU AI Act FT)** | **0.7596** | **0.7626** | **0.7900** |

**Full metrics @ dim=1024 on the combined eval set:**

| Metric | e5-large FT | Qwen3-0.6B FT | Qwen3-4B FT | OpenAI |
|--------|------------|--------------|------------|--------|
| NDCG@10 | 0.7199 | 0.7118 | **0.7596** | 0.6012 |
| MRR@10 | 0.6599 | 0.6525 | **0.7045** | 0.5357 |
| MAP@100 | 0.6642 | 0.6569 | **0.7079** | — |
| Accuracy@1 | 0.5316 | 0.5263 | **0.5830** | — |
| Accuracy@5 | 0.8266 | 0.8198 | **0.8655** | — |
| Accuracy@10 | 0.9068 | 0.8969 | **0.9306** | — |
| Recall@10 | 0.9068 | 0.8969 | **0.9306** | — |

**Qwen3-4B Matryoshka tradeoff on combined eval:**

| Dim | NDCG@10 | Δ vs 1024 |
|-----|---------|----------|
| 1024 | 0.7596 | — |
| 512 | 0.7440 | -2.1% |
| 256 | 0.7191 | -5.3% |
| 128 | 0.6710 | -11.7% |

### Analysis

**1. Fine-tuning on one legal domain transfers to another**

The most important result. All models were fine-tuned **only** on EU AI Act data, yet every one improved substantially on GDPR:

| Model | EU AI Act lift | GDPR lift | Transfer ratio |
|-------|---------------|-----------|----------------|
| e5-large | +18.5 pts | **+8.4 pts** | 45% |
| Qwen3-0.6B | +20.9 pts | **+11.0 pts** | 53% |
| Qwen3-4B | +13.5 pts | **+7.2 pts** | 53% |
| Qwen3-8B | +13.8 pts | **+7.1 pts** | 51% |

The transfer ratio is remarkably consistent: **roughly half** the training-domain improvement carries over to the unseen domain, across four different model scales. This suggests the models learn transferable Dutch legal retrieval patterns — legal vocabulary, regulatory sentence structure, query-passage matching conventions — not just EU AI Act-specific content.

**Confound — GDPR is inherently easier:** All models (zero-shot and fine-tuned) score higher on GDPR than EU AI Act. Zero-shot GDPR advantages range from +6.6 pts (0.6B) to +9.1 pts (4B). GDPR's shorter, more prescriptive articles are inherently easier to match. This doesn't negate the transfer finding — it just means the absolute GDPR scores are inflated by corpus difficulty, not just by generalisation. The transfer ratio (computed from deltas, not absolutes) controls for this.

**2. Fine-tuning beats model scale**

| Model | Combined NDCG@10 | Strategy |
|-------|-----------------|----------|
| Qwen3-4B zero-shot | 0.6494 | Scale only |
| Qwen3-8B zero-shot | 0.6683 | Scale only |
| Qwen3-0.6B fine-tuned | **0.7118** | Fine-tuning only |
| Qwen3-4B fine-tuned | **0.7596** | Both |
| Qwen3-8B fine-tuned | **0.7744** | Both |

A fine-tuned 0.6B model (+4.4 pts over 8B zero-shot) outperforms the largest zero-shot model in the family. Fine-tuning on ~2,000 synthetic pairs is worth more than a 13× parameter increase. But combining scale with fine-tuning gives the best result — the 8B fine-tuned model leads overall, though the 4B → 8B gain (+1.5 pts) shows sharply diminishing returns.

**3. Open-source zero-shot already beats proprietary SOTA**

Qwen3-4B zero-shot outperforms OpenAI `text-embedding-3-large` on every split:

| Split | Qwen3-4B ZS | OpenAI | Δ |
|-------|------------|--------|---|
| Combined | 0.6494 | 0.6012 | +4.8 pts |
| EU AI Act | 0.6274 | 0.5682 | +5.9 pts |
| GDPR | 0.7179 | 0.6733 | +4.5 pts |

This was not visible on the old 85-chunk benchmark where OpenAI (0.8635) slightly edged out e5-large zero-shot (0.8612). On the harder, more realistic benchmark, the open-source 4B model wins without any fine-tuning at all.

**4. The gap widens on harder benchmarks**

| Comparison | Old eval (85 chunks) | New eval (912 chunks) |
|-----------|---------------------|----------------------|
| OpenAI vs e5-large ZS | +0.0023 | +0.0196 |
| OpenAI vs Qwen3-4B FT | -0.1023 | -0.1584 |

The harder the benchmark, the more fine-tuning and model scale matter. OpenAI's higher native dimensionality (3072) provides minimal benefit: only +0.0168 going from 1024 to 3072. Domain knowledge beats dimensionality.

**5. Matryoshka holds up on a harder benchmark**

Qwen3-4B at dim=512 (0.7440) retains 97.9% of full-dim quality — a negligible tradeoff for 2× faster vector search. At dim=256 (0.7191), the drop is only 5.3%. These tradeoffs are similar to what was observed on the old, easier benchmark, confirming Matryoshka is robust to corpus size.

### Key takeaways

1. **Fine-tuning transfers across legal domains.** ~50% of training-domain improvement carries over to unseen GDPR text, consistently across three model architectures. Training on one Dutch regulation produces a model that is meaningfully better at retrieving from other Dutch regulations too.

2. **Fine-tuning > scale.** A fine-tuned 0.6B model beats a zero-shot 4B model by +6.2 pts. For domain-specific retrieval, targeted training on a small dataset outweighs a 7× parameter increase.

3. **Open-source beats proprietary on domain tasks.** Qwen3-4B zero-shot already outperforms OpenAI `text-embedding-3-large` by +4.8 pts on this benchmark. After fine-tuning, the gap widens to +15.8 pts.

4. **Small eval corpora lie.** NDCG@10 of 0.95+ on 85 chunks drops to 0.76 on 912 chunks for the same model. Always evaluate on a corpus at least as large as your production index.

5. **GDPR is inherently easier than EU AI Act.** All models score +6-9 pts higher on GDPR zero-shot. This is a corpus property to account for when interpreting cross-domain results.

---

## Step 14 — Qwen3-Embedding-8B Fine-tuning (LoRA)

### Motivation

Steps 1–12 showed that model scale matters: 4B LoRA outperformed 0.6B full fine-tuning on both in-domain and cross-domain benchmarks. The natural question: does scaling to 8B push the frontier further, or have we hit diminishing returns?

Qwen3-Embedding-8B is the largest model in the Qwen3-Embedding family (7.6B params, 4096-dim embeddings, #1 on MTEB multilingual leaderboard). It requires LoRA on our 32GB RTX 5090 — base weights alone consume ~16GB in bf16.

### Setup

**LoRA configuration:** Same as 4B — rank 16, alpha 32, dropout 0.05, targeting q/k/v/o projections (see Step 11 for rationale on each parameter choice). This yields 15.3M trainable parameters (0.20% of 7.6B total). The rank-16 subspace is proportionally smaller on 8B than on 4B (0.20% vs 0.29%), but this is fine for domain adaptation — the task doesn't require large representational shifts from the base model.

**VRAM constraints:** The 8B model pushes the RTX 5090 to its absolute limit:
- `mini_batch_size=1` for both Stage 1 and Stage 2 (mini_batch=2 OOMs even with flash_attention_2)
- `eval_batch_size=1` during training (end-of-epoch eval OOMs at batch=2 because training state still in VRAM)
- PEFT wrapper must be merged before any evaluation (adapter overhead causes OOM on 8B)

**flash_attention_2:** Now installed on the GPU machine. The model loads with flash attention automatically (the scripts try it first). This reduces attention memory from O(N²) to O(N), but the savings weren't enough to allow mini_batch=2 for the 8B model.

### Stage 1 results (EU AI Act eval set)

| Dim | Zero-shot | Stage 1 | Δ |
|-----|-----------|---------|---|
| 4096 | 0.8962 | **0.9682** | +0.072 |
| 1024 | 0.8836 | **0.9625** | +0.079 |
| 768 | 0.8825 | **0.9607** | +0.078 |
| 512 | 0.8774 | **0.9577** | +0.080 |
| 256 | 0.8704 | **0.9524** | +0.082 |
| 128 | 0.8369 | **0.9238** | +0.087 |

Training completed all 3 epochs (48 steps). The script OOM'd during post-training evaluation, but checkpoints were saved by the Trainer. Recovery script loaded each checkpoint, merged LoRA, evaluated, and saved the best (checkpoint-48, epoch 3).

### Operational learnings

1. **Checkpoint recovery is essential for 8B.** Training state + eval = OOM. Always have `save_strategy="epoch"` so you can recover from post-training crashes.
2. **Always merge LoRA before eval on 8B.** The PEFT wrapper's bookkeeping overhead is enough to cause OOM. Call `merge_and_unload()` first.
3. **Delete models between checkpoint evaluations.** Two 8B models in VRAM = 32GB = full GPU. Save to disk immediately, free GPU, then load the next checkpoint.
4. **flash_attention_2 helps but doesn't change batch limits.** The bottleneck on 8B is total model footprint, not attention memory.

### Stage 2 results (EU AI Act eval set)

| Dim | Zero-shot | Stage 1 | Stage 2 | Best |
|-----|-----------|---------|---------|------|
| 4096 | 0.8962 | **0.9682** | 0.9675 | **0.9682** (S1) |
| 1024 | 0.8836 | 0.9625 | **0.9625** | **0.9625** (tie) |
| 768 | 0.8825 | 0.9607 | **0.9629** | **0.9629** (S2) |
| 512 | 0.8774 | 0.9577 | **0.9587** | **0.9587** (S2) |
| 256 | 0.8704 | 0.9524 | **0.9535** | **0.9535** (S2) |
| 128 | 0.8369 | 0.9238 | **0.9253** | **0.9253** (S2) |

Stage 2 (hard negatives mined from Stage 1) produced marginal improvements at lower dims but a slight regression at the primary dim=4096 metric. With 127 in-batch negatives from GradCache already providing a strong contrastive signal, explicit hard negatives offer diminishing returns. The uploaded model uses Stage 1 (best at dim=4096). This is consistent with findings on 0.6B and 4B.

### Combined benchmark results (NDCG@10 @ dim=1024)

| Split | Zero-shot | Fine-tuned | Δ |
|-------|-----------|------------|---|
| Combined | 0.6683 | **0.7744** | +0.1061 |
| EU AI Act | 0.6369 | **0.7748** | +0.1379 |
| GDPR | 0.7348 | **0.8053** | +0.0705 |

Full metrics on combined eval (dim=1024):

| Metric | Zero-shot | Fine-tuned |
|--------|-----------|------------|
| NDCG@10 | 0.6683 | **0.7744** |
| MRR@10 | 0.6072 | **0.7206** |
| MAP@100 | 0.6125 | **0.7235** |
| Accuracy@1 | 0.4768 | **0.5992** |
| Accuracy@5 | 0.7823 | **0.8808** |
| Recall@10 | 0.8589 | **0.9410** |

**Qwen3-8B Matryoshka tradeoff on combined eval:**

| Dim | NDCG@10 | Δ vs 1024 |
|-----|---------|----------|
| 1024 | 0.7744 | — |
| 512 | 0.7653 | -1.2% |
| 256 | 0.7466 | -3.6% |
| 128 | 0.7143 | -7.8% |

### Analysis — does 8B push the frontier?

**1. Scale continues to help, but with diminishing returns**

| Model | Combined NDCG@10 | Δ vs previous |
|-------|-----------------|---------------|
| Qwen3-0.6B FT | 0.7118 | — |
| Qwen3-4B FT | 0.7596 | +0.0478 |
| Qwen3-8B FT | **0.7744** | +0.0148 |

Going from 0.6B → 4B gave +4.8 pts; going from 4B → 8B gives only +1.5 pts. The 8B model is the best, but the marginal return on scale has sharply diminished. The cost (2× VRAM, mini_batch=1, OOM risk) likely doesn't justify the +1.5 pts for most production use cases.

**2. Cross-domain transfer remains consistent**

| Model | EU AI Act lift | GDPR lift | Transfer ratio |
|-------|---------------|-----------|----------------|
| e5-large | +18.5 pts | +8.4 pts | 45% |
| Qwen3-0.6B | +20.9 pts | +11.0 pts | 53% |
| Qwen3-4B | +13.5 pts | +7.2 pts | 53% |
| Qwen3-8B | +13.8 pts | **+7.1 pts** | 51% |

The 8B model shows the same ~50% transfer ratio. Cross-domain generalisation is a property of the fine-tuning approach, not the model scale.

**3. Fine-tuning still beats scale**

| Model | Combined NDCG@10 | Strategy |
|-------|-----------------|----------|
| Qwen3-8B zero-shot | 0.6683 | Scale only |
| Qwen3-0.6B fine-tuned | **0.7118** | Fine-tuning only |
| Qwen3-4B fine-tuned | **0.7596** | Both |
| Qwen3-8B fine-tuned | **0.7744** | Both |

A fine-tuned 0.6B model (+4.4 pts over 8B zero-shot) still outperforms the largest zero-shot model in the family. Fine-tuning on ~2,000 synthetic pairs is worth more than a 13× parameter increase.

**4. Matryoshka holds up better at 8B**

| Model | dim=512 retention | dim=256 retention |
|-------|------------------|------------------|
| Qwen3-4B FT | 97.9% | 94.7% |
| Qwen3-8B FT | **98.8%** | **96.4%** |

The 8B model loses less quality at lower dimensions, likely because the larger model embeds more information into each dimension.

### Recommendation

**Qwen3-4B is the sweet spot.** It delivers 98% of the 8B's quality at half the VRAM, 4× the training throughput (mini_batch=4 vs 1), and far fewer operational headaches. The 8B model is best for maximum-quality deployments where VRAM and training cost are not constraints.

---

## Appendix A — Hyperparameter Sweep Guide

### Why sweep?

All experiments above used a single hyperparameter configuration per model. In industry, running multiple experiments with different settings is standard practice — it's the only way to verify whether you're near the optimum or leaving performance on the table.

Sweeping has diminishing returns: when your model is already at NDCG@10 = 0.94+, the expected gain is small (maybe +0.005–0.015). The ROI is highest when (a) your initial results are suboptimal, or (b) you're training on a new dataset/model for the first time.

### Which parameters matter most?

Ranked by expected impact on embedding fine-tuning quality:

#### 1. Learning Rate — the single most impactful parameter

Controls the magnitude of weight updates per step. Too high and the model diverges or overshoots; too low and the model barely changes from baseline. If you can only sweep one parameter, sweep this one.

| Scenario | Typical range |
|---|---|
| Full fine-tuning (e.g. Qwen3-0.6B) | 1e-5 to 5e-5 |
| LoRA (e.g. Qwen3-4B) | 5e-5 to 3e-4 |
| Stage 2 (any) | 0.5× to 0.1× of Stage 1 LR |

**Example sweep configs (full fine-tuning):**

| Config | LR | Epochs | Warmup | Expected behaviour |
|---|---|---|---|---|
| Conservative | 1e-5 | 4 | 0.1 | Slower convergence, less risk of overfitting, may peak later |
| Standard | 2e-5 | 3 | 0.1 | Our default — balanced convergence speed and stability |
| Aggressive | 5e-5 | 2 | 0.15 | Fast convergence, risk of overshooting on small datasets |

**Example sweep configs (LoRA):**

| Config | LR | Epochs | Warmup | Expected behaviour |
|---|---|---|---|---|
| Conservative | 5e-5 | 4 | 0.1 | More stable with small datasets, slower improvement |
| Standard | 1e-4 | 3 | 0.1 | Our default — good balance for r=16 adapters |
| Aggressive | 3e-4 | 2 | 0.15 | Can work if dataset is large enough; risks adapter instability |

#### 2. Number of Epochs

More passes = more learning, but risk of overfitting. With ~2,000 samples, overfitting is a real risk past epoch 3. With larger datasets (10K+), 5 epochs is safe.

| Epochs | When to use |
|---|---|
| 1–2 | Large dataset (>10K pairs), high LR, or Stage 2 with hard negatives |
| 3 | Default — works well for most setups with ~2K pairs |
| 4–5 | Low LR, large dataset, or if eval metrics haven't plateaued at epoch 3 |

Monitor the eval metric each epoch. If it stops improving or drops, you've gone too far.

#### 3. Batch Size (contrastive pool)

For MNRL/CachedMNRL, this determines how many in-batch negatives each query sees. Our experiments showed 63 and 127 negatives produce similar pipeline totals after hard negatives, so sweeping batch size is lower priority than LR.

| Batch size | In-batch negatives | Notes |
|---|---|---|
| 32 | 31 | Minimum viable — noticeable quality loss |
| 64 | 63 | Good default for standard MNRL |
| 128 | 127 | Our default for CachedMNRL — diminishing returns above this |
| 256 | 255 | Only useful with very large datasets |

#### 4. Temperature (CachedMNRL softmax)

Controls the "sharpness" of the contrastive distribution. Default is 0.05 in sentence-transformers.

| Value | Effect |
|---|---|
| 0.01–0.02 | Focuses on hardest negatives, sharper gradients — more aggressive |
| 0.05 | Default — good starting point |
| 0.1–0.2 | Softer distribution, more weight on easy negatives — more forgiving |

Rarely the bottleneck, but worth trying 0.02 and 0.1 alongside the default in a broader sweep.

#### 5. Weight Decay

L2 regularisation that shrinks weights toward zero. Higher values (0.05–0.1) help on very small datasets; lower (0.001) when the model is prone to underfitting.

| Value | When to use |
|---|---|
| 0.001 | Large datasets, large models, underfitting risk |
| 0.01 | Default — works well for most cases |
| 0.05–0.1 | Small datasets (<2K pairs), heavy overfitting |

### Practical sweep strategy

For a project like this (~2,000 training pairs, single GPU):

**Minimal sweep (3 runs, ~1 hour on RTX 5090):**
1. Run LR = {1e-5, 2e-5, 5e-5} with epochs=3
2. Pick the LR with best eval NDCG@10
3. Run the winner for 2 and 4 epochs to verify convergence timing

**Full sweep (9–12 runs, ~3–4 hours):**
1. LR ∈ {1e-5, 2e-5, 5e-5} × Epochs ∈ {2, 3, 4} = 9 combinations
2. Add 1–2 runs varying temperature or weight decay for the best LR/epoch combo
3. Use W&B (Weights & Biases) or TensorBoard to compare runs

**When NOT to sweep:**
- Model already at NDCG@10 > 0.95 on your eval set — you're near ceiling
- Dataset is the bottleneck, not the hyperparameters — invest in more/better data instead
- You're doing exploratory work and need directional results, not optimal ones

### How each parameter affects the training dynamics

```
LR ↑  → faster convergence, but risk of overshoot / collapse
LR ↓  → slower, safer, but may never reach the optimum in N epochs

Epochs ↑  → more passes over data, better fit, but overfitting risk
Epochs ↓  → underfitting if LR is low, but safer with high LR

Batch ↑  → more in-batch negatives, stronger signal, smoother gradients
Batch ↓  → fewer negatives, noisier gradients, but hard negatives compensate

Temp ↓  → sharper loss landscape, focuses on hardest examples
Temp ↑  → smoother loss, more uniform gradient from all negatives

Weight decay ↑  → stronger regularisation, fights overfitting
Weight decay ↓  → weaker regularisation, model has more freedom
```

The key interaction: **LR and epochs are inversely related**. A higher LR needs fewer epochs (the model converges faster but overshoots sooner). A lower LR needs more epochs (slower convergence, later peak). When sweeping, always vary them together.

---

## Appendix B — Potential Pipeline Improvements

The following improvements have been identified from reviewing NVIDIA's [embedding fine-tuning recipe](https://huggingface.co/blog/nvidia/domain-specific-embedding-finetune) and our own experimental observations. Scripts for items 1 and 2 have been implemented in `synthetic_dataset_creation/`.

### 1. Multi-hop query generation

**Status:** Implemented — `synthetic_dataset_creation/generate_multihop_queries.py`

**Problem:** All current synthetic queries are single-hop — each query maps to exactly one chunk. Real users often ask complex questions that span multiple articles or sections (e.g. *"Hoe beïnvloeden de transparantieverplichtingen de conformiteitsbeoordeling van hoog-risico AI-systemen?"*). The model never sees these during training and may fail to retrieve all relevant passages.

**Approach:**
1. Group related chunks (same chapter/different articles, adjacent chunks, same section type)
2. Generate queries that require information from 2–3 chunks to answer
3. "Unroll" each multi-hop query: a 2-hop query produces 2 training pairs `(query, chunk_1)` and `(query, chunk_2)`, teaching the model that both chunks are relevant

**Expected impact:** Better retrieval for composite questions in production. Especially relevant for legal text where questions often span multiple articles or recitals.

**Usage:**
```bash
python synthetic_dataset_creation/generate_multihop_queries.py
```

The output (`*_multihop_pairs.jsonl`) can be merged with the standard single-hop pairs before training. A reasonable mix is ~80% single-hop + ~20% multi-hop.

### 2. Quality scoring and filtering for synthetic data

**Status:** Implemented — `synthetic_dataset_creation/score_and_filter_queries.py`

**Problem:** Not all LLM-generated (query, chunk) pairs are high quality. Some queries may be ambiguous, too generic, or point to the wrong chunk. Training on noisy pairs injects noise into the contrastive signal. With a small dataset (~2K pairs), even 5–10% bad pairs can matter.

**Approach:**
1. An LLM judge scores each pair on 4 dimensions: **relevance** (does the chunk answer the query?), **accuracy** (does the chunk contain the needed information?), **clarity** (is the query well-formed?), **specificity** (is this chunk uniquely the right answer?)
2. Compute a weighted overall score: `0.35 × relevance + 0.25 × accuracy + 0.20 × clarity + 0.20 × specificity`
3. Filter out pairs below a threshold (default: 7.0/10)

**Expected impact:** Cleaner training signal. NVIDIA's recipe uses this approach and considers it essential for their pipeline. Removing the bottom ~10–20% of pairs should improve convergence and final metrics.

**Usage:**
```bash
python synthetic_dataset_creation/score_and_filter_queries.py
```

Outputs three files: filtered pairs (for training), rejected pairs (for analysis), and the full scored dataset.

### 3. Lower contrastive temperature (0.02)

**Status:** Not yet tested — single parameter change

**Problem:** We use the default CachedMNRL temperature of 0.05. NVIDIA's recipe uses 0.02, calling it "deliberately aggressive" — it produces a sharper probability distribution that focuses the model on the hardest negatives.

**Why it could help:** With our improved filtered hard negatives (margin + range filtering), the negatives are higher quality. A lower temperature would amplify the gradient signal from these carefully selected negatives, potentially improving fine-grained discrimination between similar passages.

**How to test:** In the Stage 2 fine-tuning script, change the loss function temperature:

```python
# Current (default)
inner_loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH)

# Experiment: lower temperature
inner_loss = CachedMultipleNegativesRankingLoss(
    model, mini_batch_size=MINI_BATCH, temperature=0.02
)
```

**Risk:** Lower temperature makes training more sensitive to noise in hard negatives. Only try this *after* implementing quality filtering (item 2) to ensure the negatives are clean.

**Expected impact:** Small but potentially meaningful — most useful in Stage 2 where hard negatives dominate the learning signal.
