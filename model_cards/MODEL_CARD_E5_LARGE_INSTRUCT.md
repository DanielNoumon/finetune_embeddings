# multilingual-e5-large-instruct — EU AI Act NL Fine-tuning

## Model Description

Fine-tuned version of [`intfloat/multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct) for Dutch-language retrieval on the EU AI Act. The model is trained using Multiple Negatives Ranking Loss (MNRL) in a two-stage pipeline: Stage 1 with in-batch negatives, Stage 2 with mined hard negatives.

- **Base model**: `intfloat/multilingual-e5-large-instruct` (24 layers, 1024-dim embeddings)
- **Language**: Dutch (NL)
- **Domain**: EU AI Act regulation
- **Training hardware**: NVIDIA RTX 5090 (32GB VRAM, Blackwell architecture)

## Training Pipeline

### Stage 1: In-batch Negatives (MNRL)

Fine-tune the base instruct model using MNRL with in-batch negatives. Each query is paired with its correct passage; all other passages in the batch serve as negatives.

### Stage 2: Hard Negatives (MNRL)

_(Pending — results will be added after Stage 2 completes)_

Mine hard negatives from the Stage 1 model, then fine-tune further with explicit hard negatives + in-batch negatives.

## Dataset

**Training data:** [danielnoumon/eu-ai-act-nl-queries](https://huggingface.co/datasets/danielnoumon/eu-ai-act-nl-queries)

- 2,284 synthetic Dutch query-chunk pairs (1,944 train / 340 eval)
- Source: EU AI Act (Dutch translation)
- Queries generated via Azure OpenAI GPT-5-mini
- Split at the chunk level to prevent data leakage between train and eval

---

## Experiment Log

### 1. Blackwell GPU Numerical Stability

The RTX 5090 (Blackwell, sm_120) has critical compatibility issues with standard training configurations.

| Configuration | Result |
|--------------|--------|
| bf16 + SDPA | **Catastrophic collapse** — all cosine similarities → 1.0, gradient norms spike to 500-4000+ |
| bf16 + eager attention | **Catastrophic collapse** — same as above |
| fp32 + SDPA | Partially stable, some instability |
| **fp32 + eager attention** | **Stable** ✅ |

**Root cause**: bf16 precision causes immediate gradient explosion on Blackwell GPUs, regardless of attention implementation. This does not occur on Ampere/Ada GPUs (A100, T4, RTX 4090).

**Fix**: Force fp32 precision with eager attention (`attn_implementation="eager"`).

### 2. Gradient Checkpointing

| Setting | Result |
|---------|--------|
| `gradient_checkpointing=True` + fp32 + eager | **Collapse** — embedding space degenerates |
| `gradient_checkpointing=False` + fp32 + eager | **Stable** ✅ |

Gradient checkpointing recomputes activations during the backward pass, which introduces additional numerical error. On Blackwell with fp32, this is enough to destabilize training.

**Fix**: Disable gradient checkpointing. This limits batch size but preserves training stability.

### 3. MatryoshkaLoss vs Plain MNRL

| Loss function | Result (1 epoch) |
|--------------|-------------------|
| MatryoshkaLoss wrapping MNRL | **Degradation** — multi-dim gradient signals amplify instability |
| **Plain MNRL** | **Stable improvement** ✅ |

MatryoshkaLoss computes loss at multiple embedding dimensions (1024, 768, 512, 256, 128, 64) and sums the gradients. This amplifies any numerical instability present in the fp32+eager setup.

**Fix**: Use plain MNRL for training. Matryoshka-style truncation still works at inference time since the model naturally learns useful sub-dimensions.

### 4. Learning Rate and Gradient Clipping

The instruct model is already heavily fine-tuned (base → contrastive pre-training → instruction tuning). It requires extremely conservative hyperparameters.

| LR | max_grad_norm | Epochs | NDCG@10 |
|----|--------------|--------|---------|
| 5e-6 | 1.0 (default) | 1 | Collapse |
| 1e-6 | 0.3 | 1 | 0.8442 (+0.033) |
| **1e-6** | **0.3** | **3** | **0.8956 (+0.084)** ✅ |
| 1e-6 | 0.3 | 5 | 0.8537 (+0.042)* |

*5-epoch result degraded because `save_total_limit=2` deleted the best checkpoint (epoch 3). The actual peak was at epoch 3.

### 5. Batch Size and VRAM Constraints

fp32 without gradient checkpointing severely limits batch size on 32GB VRAM:

| Batch size | Effective batch | VRAM | NDCG@10 | Notes |
|-----------|----------------|------|---------|-------|
| 128 | 128 | OOM | — | fp32 too large |
| 64 | 128 (×2 accum) | OOM | — | Still too large without checkpointing |
| **16** | **128 (×8 accum)** | **~31GB** | **0.8956** | **Best — 15 in-batch negatives** ✅ |
| 8 | 128 (×16 accum) | ~22GB | 0.8296 | Too few in-batch negatives (7) |

**Key insight**: For MNRL, in-batch negatives come from the micro-batch, not the effective batch. Gradient accumulation does NOT increase the number of negatives. Batch 16 is the minimum for decent MNRL quality with fp32 on 32GB.

Required environment variable for reliable batch 16: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 6. Instruction Prefix Optimization

The instruct model requires a task instruction prepended to each query. We tested 8 different phrasings on the base model (no fine-tuning):

| Instruction | Base NDCG@10 | Post-finetune NDCG@10 |
|------------|-------------|----------------------|
| `"Retrieve the most relevant passage for this query"` | **0.8498** | 0.8711 |
| `"Given a web search query, retrieve relevant passages that answer the query"` | 0.8446 | — |
| `"Given a question, retrieve relevant passages that answer the question"` | 0.8421 | — |
| **`"Given a question about EU AI regulation, retrieve the most relevant passage"`** | 0.8116 | **0.8956** ✅ |
| `"query: "` (non-instruct baseline) | 0.7951 | — |
| `"Given a question in Dutch about EU AI regulation, retrieve the most relevant passage in Dutch"` | 0.7928 | — |
| `"Given a question about AI legislation, retrieve relevant legal passages that answer the question"` | 0.7203 | — |
| No prefix | 0.6675 | — |

**Key finding**: Generic instructions score higher zero-shot, but **domain-specific instructions produce better fine-tuned models**. The domain-specific instruction gives the model a stronger learning signal during training by explicitly describing the target domain.

**Best instruction for fine-tuning**: `"Given a question about EU AI regulation, retrieve the most relevant passage"`

---

## Best Stage 1 Configuration

```python
MODEL       = "intfloat/multilingual-e5-large-instruct"
PRECISION   = fp32
ATTENTION   = eager
GRAD_CKPT   = False
LOSS        = MultipleNegativesRankingLoss (plain, no Matryoshka)
BATCH_SIZE  = 16  (effective 128 via grad_accum=8)
EPOCHS      = 3
LR          = 1e-6
WARMUP      = 3 steps
GRAD_CLIP   = 0.3
WEIGHT_DECAY = 0.01
INSTRUCTION = "Instruct: Given a question about EU AI regulation, retrieve the most relevant passage\nQuery: "
CORPUS_PROMPT = ""  (no prefix for documents)
```

**Environment**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Best Stage 1 Results

| Dim | Base | Finetuned | Δ |
|-----|------|-----------|---|
| 1024 | 0.8116 | **0.8956** | **+0.0840** |
| 768 | 0.8072 | 0.8934 | +0.0863 |
| 512 | 0.8131 | 0.8830 | +0.0699 |
| 256 | 0.7547 | 0.8693 | +0.1146 |
| 128 | 0.7070 | 0.8320 | +0.1250 |
| 64 | 0.6337 | 0.7296 | +0.0959 |

### Full Metrics at dim=1024

| Metric | Base | Finetuned | Δ |
|--------|------|-----------|---|
| NDCG@10 | 0.8116 | 0.8956 | +0.0840 |
| MRR@10 | 0.7780 | 0.8682 | +0.0902 |
| MAP@100 | 0.7826 | 0.8695 | +0.0870 |
| Accuracy@1 | 0.7000 | 0.8000 | +0.1000 |
| Accuracy@3 | 0.8500 | 0.9353 | +0.0853 |
| Accuracy@5 | 0.8882 | 0.9500 | +0.0618 |
| Accuracy@10 | 0.9147 | 0.9794 | +0.0647 |
| Recall@10 | 0.9147 | 0.9794 | +0.0647 |

## Stage 2 Results

**Not pursued.** The non-instruct model significantly outperformed the instruct variant at Stage 1 (0.9327-0.9436 vs 0.8956). Stage 2 hard negative training was only run on the non-instruct pipeline.

## Comparison with Non-Instruct Model

| Stage | Non-Instruct (best) | Instruct (RTX 5090) |
|-------|---------------------|---------------------|
| Base | 0.8612 | 0.8116 |
| Stage 1 | 0.9327–0.9436 | 0.8956 |
| Stage 2 | **0.9492** | _(not pursued)_ |

The non-instruct model outperforms due to:
1. **MatryoshkaLoss works** on non-instruct — it amplifies instability on the instruct model, limiting it to plain MNRL
2. **Higher base quality** (0.8612 vs 0.8116) — e5-large non-instruct has a stronger starting point for retrieval-style fine-tuning
3. **More in-batch negatives possible** — non-instruct fits batch 8-128 depending on config, vs max batch 16 for instruct

### Non-instruct experiment history (for reference)

| # | Config | In-batch neg | Hard neg | Stage 1 | Stage 2 |
|---|--------|-------------|----------|---------|----------|
| 1 | Colab T4, batch 64, fp16 | 63 | 1 | 0.9436 | 0.9465 |
| 2 | RTX 5090, batch 8, fp32 | 7 | 1 | 0.9327 | **0.9492** |
| 3 | RTX 5090, CachedMNRL batch 128, fp32 | 127 | 1 | 0.9422 | 0.9463 |
| 4 | RTX 5090, CachedMNRL batch 128, fp32 | 127 | 5 | 0.9422 | 0.9398 |

## Lessons Learned

1. **Blackwell GPUs (RTX 5090) have critical bf16 instability** for fine-tuning transformer models. Use fp32 + eager attention until driver/PyTorch maturity improves.
2. **Gradient checkpointing is unsafe on Blackwell** — it amplifies numerical errors.
3. **MNRL quality depends on micro-batch size**, not effective batch size. Gradient accumulation doesn't help with in-batch negatives.
4. **Instruct models need extremely conservative fine-tuning** — they're already heavily trained. LR=1e-6 with aggressive gradient clipping (0.3) is necessary.
5. **Domain-specific instructions outperform generic ones after fine-tuning**, even though generic instructions score higher zero-shot.
6. **3 epochs is optimal for Stage 1** — performance degrades after epoch 3.
7. **Non-instruct > instruct for domain-specific retrieval fine-tuning** — the instruct model's existing instruction tuning conflicts with domain adaptation, and MatryoshkaLoss instability further limits it.
