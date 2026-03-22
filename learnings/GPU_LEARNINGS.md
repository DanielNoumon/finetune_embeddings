# GPU Learnings: RTX 5090 (Blackwell) for Embedding Fine-tuning

Personal reference document for lessons learned while fine-tuning `intfloat/multilingual-e5-large` (and its instruct variant) on an NVIDIA RTX 5090 (32GB, Blackwell architecture, sm_120).

---

# Part 1: Key Concepts

Before diving into the GPU-specific findings, here's an explanation of every technical concept referenced in this document.

---

## Concept: GPU Architectures (Blackwell, Ampere, Ada)

NVIDIA releases new GPU architectures every ~2 years. Each has a codename and introduces hardware changes:

| Architecture | Codename | Example GPUs | Released |
|-------------|----------|-------------|---------|
| Ampere | sm_80/86 | A100, RTX 3090 | 2020 |
| Ada Lovelace | sm_89 | RTX 4090 | 2022 |
| **Blackwell** | **sm_120** | **RTX 5090** | 2025 |

Each architecture has different hardware units for floating-point math. **Blackwell is the newest**, and its floating-point units behave slightly differently than older architectures — which is why some precision formats that work on Ampere/Ada break on Blackwell.

**"sm_120"** is the CUDA compute capability identifier. Software (PyTorch, CUDA kernels) must be compiled to support this identifier, or it won't run on the GPU at all. This is why we needed `torch>=2.6.0` with CUDA 12.8.

---

## Concept: Number Precision (fp32, fp16, bf16)

Computers store decimal numbers (like 0.00347 or 1024.5) as **floating-point numbers**. The more bits you use, the more precisely you can represent a number — but the more memory it costs.

### How floating-point works

A floating-point number has three parts (like scientific notation):

```
(-1)^sign × 1.mantissa × 2^exponent

Example: 0.15625 = (-1)^0 × 1.01 × 2^(-3)    (in binary)
```

- **Sign bit** (1 bit): positive or negative
- **Exponent** bits: the "scale" — how big or small the number is (like the ×10^6 in scientific notation)
- **Mantissa** bits: the "precision" — how many significant digits you have

### The three formats we care about

| Format | Total bits | Exponent | Mantissa | Memory per number |
|--------|-----------|----------|----------|-------------------|
| **fp32** (float32) | 32 | 8 bits | 23 bits | 4 bytes |
| **fp16** (float16) | 16 | 5 bits | 10 bits | 2 bytes |
| **bf16** (bfloat16) | 16 | 8 bits | 7 bits | 2 bytes |

**fp32** — Full precision. 23 mantissa bits = ~7 decimal digits of precision. Very accurate but uses 4 bytes per number. A 560M-parameter model stores 560M numbers → 2.2 GB just for weights.

**fp16** — Half precision. 10 mantissa bits = ~3 decimal digits. Uses only 2 bytes per number (half of fp32). The model fits in 1.1 GB. But the small exponent range (5 bits) means it can't represent very large or very small numbers — gradients during training can "overflow" (become infinity) or "underflow" (become zero).

**bf16** (Brain Float 16) — Google's format. Same exponent range as fp32 (8 bits) so it handles large/small numbers well, but only 7 mantissa bits = ~2 decimal digits of precision. It was designed specifically for deep learning: most operations don't need high precision, they just need wide range.

### Why precision matters for training

During training, the model computes **gradients** — tiny adjustments to each of its 560M parameters. These gradients are often very small numbers (like 0.0000034). With low precision:

- **bf16** (7 mantissa bits): a gradient of 0.0000034 might be rounded to 0.000003 or 0.000004 — losing 15% of the signal
- **fp32** (23 mantissa bits): the same gradient is stored accurately

When you multiply millions of these slightly-wrong gradients through 24 transformer layers (the backward pass), the errors compound. On Blackwell GPUs specifically, this compounding causes **gradient explosion** — the errors amplify until the model's weights become garbage.

### Analogy

Imagine you're navigating with a compass:
- **fp32** = compass accurate to 0.001° — you arrive at your destination
- **fp16** = compass accurate to 0.1° — you drift a bit but GPS corrections (loss scaling) keep you on track
- **bf16** = compass accurate to 1° — on most roads (older GPUs) you're fine, but on a winding mountain road (Blackwell) you drive off a cliff

---

## Concept: VRAM (Video RAM)

VRAM is the GPU's dedicated memory. It's like your computer's RAM, but exclusively for the GPU. Everything the GPU works with must fit in VRAM:

1. **Model weights** — the 560M learned parameters
2. **Optimizer states** — Adam optimizer stores 2 extra copies of each parameter (momentum + variance)
3. **Gradients** — the computed parameter updates (same size as weights)
4. **Activations** — intermediate results saved during the forward pass, needed for the backward pass

Items 1-3 are **fixed** — they don't change with batch size. Item 4 (activations) **scales with batch size** — double the batch = double the activations.

Our RTX 5090 has **32 GB** of VRAM. With fp32:
- Fixed overhead: ~18 GB (weights + optimizer + gradients)
- Remaining for activations: ~14 GB
- This limits us to batch 8-16 depending on the loss function

---

## Concept: Attention Mechanisms (SDPA vs Eager)

Transformers (the architecture behind our embedding model) use **self-attention** to let each word "look at" every other word in the input. The math is:

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

Where:
- **Q** (Query), **K** (Key), **V** (Value) are matrices derived from the input
- **Q × K^T** computes a similarity score between every pair of words
- **softmax** normalizes these scores to sum to 1
- The result × V produces the output, weighted by how much each word "attends" to others

There are two ways to compute this:

**SDPA (Scaled Dot-Product Attention):** PyTorch fuses the entire computation into a single optimized GPU kernel. Faster, uses less memory (doesn't need to store the full attention matrix). This is the default on modern PyTorch.

**Eager attention:** Computes each step separately as individual operations (`matmul → scale → softmax → matmul`). Slower and uses more memory, but each step is computed with full numerical precision — no shortcutting from kernel fusion.

On most GPUs, SDPA and eager produce identical results. On Blackwell with fp32, SDPA introduces tiny numerical differences that accumulate and cause instability. Eager attention is more predictable.

---

## Concept: Gradient Norms

During training, the model computes **gradients** — a vector of partial derivatives that tells the optimizer which direction to adjust each parameter and by how much.

The **gradient norm** is the magnitude (length) of this gradient vector:

```
grad_norm = √(sum of all gradient values squared)
```

Think of it like a compass needle: the direction tells you where to go, the norm tells you how urgently.

**Healthy gradient norms:** Typically 0.1–10 for a stable training run. The exact range depends on model size and loss function, but it should stay relatively stable.

**Gradient explosion:** When norms spike to 100, 1000, or higher. This means the model is making enormous parameter updates — like turning the steering wheel 180° instead of 2°. The model's weights become garbage in one step. This is what happened with bf16 on Blackwell: gradient norms spiked to 500–4000+.

**Gradient clipping** (`max_grad_norm`): A safety mechanism. If the gradient norm exceeds the threshold, all gradients are scaled down proportionally:

```python
if grad_norm > max_grad_norm:
    gradients *= (max_grad_norm / grad_norm)
```

With `max_grad_norm=0.3`: if the model computes a gradient norm of 3.0, all gradients are multiplied by 0.3/3.0 = 0.1, reducing the update by 10×. This prevents catastrophic weight updates at the cost of slower learning.

---

## Concept: Gradient Checkpointing

During the forward pass, the model saves **activations** (intermediate outputs of each layer) because they're needed during the backward pass to compute gradients. For a 24-layer transformer with batch 64:

```
24 layers × activations per layer × batch size = a LOT of VRAM
```

**Gradient checkpointing** trades compute for memory:
- **Forward pass:** Don't save activations — just throw them away
- **Backward pass:** When you need layer 12's activations to compute its gradients, re-run the forward pass from layer 11 to recompute them

This roughly **halves activation memory** at the cost of ~30% slower training (because you compute the forward pass twice).

**The problem on Blackwell:** Recomputing activations introduces a second round of floating-point operations. Due to the order of operations differing slightly between the original forward pass and the recomputed one, the results aren't bit-for-bit identical. This tiny discrepancy, accumulated through 24 layers, is enough to destabilize training on Blackwell.

---

## Concept: Batch Size, Micro-batch, Effective Batch, and Gradient Accumulation

### Batch size

Neural networks don't train on one sample at a time — they process **batches** of samples simultaneously. This is faster (GPU parallelism) and produces smoother gradient estimates (averaging over multiple samples reduces noise).

### Micro-batch vs effective batch

When a batch of 16 samples is processed in a single GPU forward+backward pass, that's the **micro-batch** (also called per-device batch size).

**Gradient accumulation** lets you simulate a larger batch without needing the VRAM for it:

```
Step 1: Forward+backward on 16 samples → store gradients (don't update weights yet)
Step 2: Forward+backward on 16 more samples → add gradients to stored ones
Step 3: Forward+backward on 16 more samples → add again
Step 4: Forward+backward on 16 more samples → add again
→ Now update weights using the accumulated gradients from all 64 samples
```

With `batch_size=16, grad_accum=4`: the **effective batch** is 64, but only 16 samples are in VRAM at any time.

### Why this matters for MNRL

**This is the single most important concept for understanding our results.**

MNRL (Multiple Negatives Ranking Loss) works by treating every other passage in the **micro-batch** as a negative example. With micro-batch 16:

```
Sample 1: Query₁ matches Passage₁ (positive). Passages 2-16 = negatives (15 negatives)
Sample 2: Query₂ matches Passage₂ (positive). Passages 1,3-16 = negatives (15 negatives)
...
```

Gradient accumulation does NOT combine the micro-batches for negative sampling. The model never "sees" all 64 samples at once — it processes them in groups of 16. So:

- `batch_size=64, grad_accum=1` → **63 negatives per query** ← stronger training signal
- `batch_size=16, grad_accum=4` → **15 negatives per query** ← weaker training signal
- `batch_size=8, grad_accum=8` → **7 negatives per query** ← much weaker

Both have effective batch 64 (same gradient quality for most losses), but MNRL specifically needs large micro-batches because its negatives come from within the micro-batch.

---

## Concept: Learning Rate

The learning rate controls **how big each weight update is**:

```
new_weight = old_weight - learning_rate × gradient
```

- **Too high** (e.g., 1e-3): Large steps → overshoot the optimum → model diverges (loss goes to infinity or embeddings collapse)
- **Too low** (e.g., 1e-8): Tiny steps → model barely changes → waste of compute
- **Just right** (e.g., 2e-5 for base models, 1e-6 for pre-fine-tuned models): Meaningful progress without overshooting

### Why pre-fine-tuned models need lower learning rates

A base model (freshly pre-trained) has weights that are "general purpose" — you can push them significantly toward your task without losing much. But an instruct model has already been fine-tuned once (or twice). Its weights are carefully arranged. A large learning rate tears apart these arrangements — like renovating a house by demolishing walls instead of repainting.

### Warmup

Training typically starts with a **warmup** period: the learning rate starts at 0 and linearly increases to the target over the first few steps. This prevents a sudden large update when the model hasn't "seen" enough data to know which direction to go.

```
warmup_ratio=0.1  →  first 10% of training steps use reduced learning rate
warmup_steps=3    →  first 3 steps use reduced learning rate
```

### Weight decay

A regularization technique that gently pulls all weights toward zero each step:

```
new_weight = old_weight × (1 - weight_decay) - learning_rate × gradient
```

With `weight_decay=0.01`, each weight shrinks by 1% per step. This prevents the model from relying too heavily on any single parameter, reducing overfitting.

---

## Concept: Epochs

One **epoch** = one complete pass through the entire training dataset.

With 1,944 training samples and batch size 64:
- 1 epoch = 1944/64 ≈ 30 steps
- 3 epochs = ~90 steps

**More epochs** = the model sees each sample more times. This can help the model learn more nuanced patterns, but too many epochs leads to **overfitting** — the model memorizes the training data instead of learning generalizable patterns.

For our dataset (only ~2000 samples), overfitting is a real risk. We found that 3 epochs is optimal for Stage 1 — performance degrades after that.

---

## Concept: Loss Functions (MNRL, MatryoshkaLoss)

A **loss function** measures "how wrong" the model is. Training minimizes this number. Lower loss = better model.

### Multiple Negatives Ranking Loss (MNRL)

The core loss for retrieval fine-tuning. For each query in a batch:

1. Compute similarity between the query and ALL passages in the batch
2. The correct passage should have the highest similarity
3. Loss = how much the model "got wrong" in this ranking

It's essentially an N-way classification problem: "which of these N passages is the correct one?" With batch size 64, it's a 64-way classification per query. Harder task = stronger training signal.

### MatryoshkaLoss

Named after Russian nesting dolls. Wraps MNRL so the model learns useful embeddings at multiple sizes:

```python
loss = MatryoshkaLoss(model, MNRL, dims=[1024, 768, 512, 256, 128, 64])
```

During each training step, it:
1. Truncates all embeddings to 1024 dims → computes MNRL loss
2. Truncates to 768 dims → computes MNRL loss
3. Truncates to 512 dims → computes MNRL loss
4. ... and so on for each dimension
5. Averages all 6 losses

This means **6 forward passes** through the loss function per step (one per dimension), which uses ~6× more activation memory than plain MNRL. That's why MatryoshkaLoss reduces our max batch size from 16 to 8.

The benefit: at inference time, you can truncate embeddings to any of these sizes for faster search with minimal quality loss.

---

## Concept: NDCG@10

**Normalized Discounted Cumulative Gain at rank 10** — our primary evaluation metric.

It measures: "when the model retrieves the top 10 passages for a query, how well does it rank the correct passages?"

- **Score of 1.0** = perfect ranking — the correct passage is always #1
- **Score of 0.5** = mediocre — correct passages are scattered around the results
- **Score of 0.0** = completely wrong — correct passages never appear in top 10

The "discounted" part means higher-ranked results count more. Being #1 matters much more than being #8.

---

## Concept: Cosine Similarity and Embedding Collapse

**Cosine similarity** measures the angle between two embedding vectors, ignoring their magnitude:

```
cos(A, B) = (A · B) / (|A| × |B|)
```

- **cos = 1.0**: vectors point in the same direction (identical meaning)
- **cos = 0.0**: vectors are perpendicular (unrelated)
- **cos = -1.0**: vectors point in opposite directions

**Embedding collapse** is when the model maps ALL inputs to nearly the same point in embedding space. Every pair of texts gets cosine similarity ~1.0 — the model has lost the ability to distinguish anything. This is catastrophic and irreversible (the model must be reloaded from scratch).

In our experiments, collapse happened when:
- bf16 precision caused gradient explosion → weights blown to meaningless values
- Gradient checkpointing introduced numerical errors → same result
- MatryoshkaLoss amplified small instabilities → cumulative collapse

---

## Concept: CUDA Memory Fragmentation

When PyTorch allocates and frees GPU memory during training, it can leave small "holes" of unused memory scattered across VRAM — like a parking lot where cars are spread out with empty spaces between them. Even if total free memory is 2 GB, no single contiguous block might be large enough for a 500 MB tensor allocation → OOM error.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` tells PyTorch to use expandable memory segments that can grow as needed, avoiding fragmentation. Always use this when VRAM is tight.

---

# Part 2: Findings and Experiments

Everything below documents what we discovered when fine-tuning on the RTX 5090.

---

## 1. bf16 Causes Collapse on Blackwell

**What happens:** Training with `bf16=True` causes immediate catastrophic embedding collapse. All cosine similarities converge to ~1.0 within the first few training steps. Gradient norms spike to 500–4000+ (healthy is 0.1–10).

**Why:** bf16's 7-bit mantissa can't represent gradients precisely enough. During the backward pass through 24 transformer layers, rounding errors compound until the gradient signal is pure noise. The optimizer then makes random weight updates, destroying the embedding space.

**Tested configurations:**

| Precision | Attention | Result |
|-----------|-----------|--------|
| bf16 | SDPA | Collapse |
| bf16 | Eager | Collapse |
| fp32 | SDPA | Partially stable |
| **fp32** | **Eager** | **Stable** ✅ |

**Does NOT happen on:** Colab T4, A100, RTX 4090. Their floating-point units handle bf16 without the same error accumulation.

**Fix:** `fp32=True, bf16=False, fp16=False`. Costs 2× VRAM but is fully stable.

---

## 2. SDPA Partially Unstable on Blackwell

Even with fp32, SDPA shows occasional instability. The fused kernel computes attention in a different order than the mathematical definition, which produces slightly different rounding. On Blackwell, this is enough to cause intermittent training issues.

**Fix:** `model_kwargs={"attn_implementation": "eager"}` when loading the model.

**Cost:** ~10-15% slower training. Marginally more VRAM.

---

## 3. Gradient Checkpointing Causes Collapse

Gradient checkpointing recomputes activations during the backward pass. The recomputed values aren't bit-for-bit identical to the original (different operation ordering → different floating-point rounding). Through 24 layers, this discrepancy is enough to destabilize training on Blackwell.

**Fix:** `gradient_checkpointing=False`.

**Cost:** Cannot trade compute for memory. Batch size is hard-limited by VRAM.

---

## 4. Batch Size Constraints and MNRL Quality

The three constraints above (fp32 + eager + no checkpointing) severely limit batch size:

**Practical batch limits on RTX 5090 (32GB, fp32):**

| Configuration | Max micro-batch | Forward passes per step |
|--------------|----------------|------------------------|
| Plain MNRL (2 columns) | 16 | 2 |
| MatryoshkaLoss + MNRL (2 columns) | 8 | 2 × 6 = 12 |
| Plain MNRL + hard neg (3 columns) | 8 | 3 |
| MatryoshkaLoss + hard neg (3 columns) | 4 | 3 × 6 = 18 |

**Impact on MNRL training quality:**

| Setup | Micro-batch | In-batch negatives | NDCG@10 |
|-------|-------------|-------------------|---------|
| Colab T4 (fp16, checkpointing) | 64 | 63 | 0.9436 |
| RTX 5090 (fp32, no checkpointing) | 8 | 7 | 0.9327 |
| **Difference** | | | **-0.0109** |

The batch size constraint costs ~1.1% NDCG. Meaningful but not catastrophic — the model still learns well with only 7 negatives.

---

## 5. VRAM Budget Breakdown

| Component | fp32 | bf16 |
|-----------|------|------|
| Model weights (560M params) | 2.2 GB | 1.1 GB |
| Optimizer (Adam, 2 states) | 4.4 GB | 4.4 GB* |
| Gradients | 2.2 GB | 1.1 GB |
| **Fixed overhead** | **~8.8 GB** | **~6.6 GB** |
| **Remaining for activations (32GB GPU)** | **~23 GB** | **~25 GB** |

*Adam always stores its momentum and variance states in fp32, regardless of training precision.

Activations are also 2× larger in fp32, so the same batch uses 2× the activation memory. Net effect: **~3-4× smaller maximum batch size** in fp32 vs bf16.

---

## 6. Instruct vs Non-instruct Model Comparison

We fine-tuned both `multilingual-e5-large` and `multilingual-e5-large-instruct` on the same data to isolate the causes of the performance gap.

### Base model performance (zero-shot, no fine-tuning)

| Model | NDCG@10 | Notes |
|-------|---------|-------|
| `multilingual-e5-large` | 0.8612 | "query: " / "passage: " prefixes |
| `multilingual-e5-large-instruct` | 0.8116 | Domain-specific instruction |
| `multilingual-e5-large-instruct` | 0.8498 | Generic instruction |

The instruct model underperforms zero-shot despite being "more advanced". Likely because:
- The instruction adds overhead without adding value for this specific task/dataset
- Cross-lingual friction (English instruction + Dutch queries/documents)

### After Stage 1 fine-tuning

| Setup | Base | Stage 1 | Δ |
|-------|------|---------|---|
| Non-instruct, Colab (batch 64, fp16) | 0.8612 | **0.9436** | +0.0825 |
| Non-instruct, RTX 5090 (batch 8, fp32) | 0.8612 | **0.9327** | +0.0715 |
| Instruct, RTX 5090 (batch 16, fp32, plain MNRL) | 0.8116 | **0.8956** | +0.0840 |

**Gap decomposition (instruct RTX vs non-instruct Colab):**

| Factor | Impact | % of total gap |
|--------|--------|---------------|
| Hardware constraints (fp32, small batch, no checkpointing) | ~1.1% | ~23% |
| Model difference (instruct vs non-instruct) | ~3.7% | ~77% |
| **Total gap** | **~4.8%** | 100% |

**Conclusion:** The instruct model is the main bottleneck, not the GPU.

---

## 7. Instruction Prefix Optimization

Tested 8 instruction phrasings on the base instruct model:

| Instruction | Base NDCG@10 | After fine-tuning |
|------------|-------------|-------------------|
| Generic: "Retrieve the most relevant passage for this query" | **0.8498** | 0.8711 |
| Domain-specific: "Given a question about EU AI regulation, retrieve the most relevant passage" | 0.8116 | **0.8956** |

**Counterintuitive finding:** Generic instructions score higher zero-shot but produce worse fine-tuned models. Domain-specific instructions give the model a stronger learning signal during training — the instruction explicitly tells the model what domain to optimize for.

**Rule of thumb:** For fine-tuning, use domain-specific instructions. For zero-shot/inference, use generic ones.

---

## 8. Conservative Hyperparameters for Pre-fine-tuned Models

The instruct model is already heavily fine-tuned (base → contrastive pre-training → instruction tuning). It needs much gentler updates.

| Parameter | Base model | Instruct model | Why different |
|-----------|-----------|----------------|---------------|
| Learning rate | 2e-5 | **1e-6** (20× smaller) | Large updates destroy carefully learned representations |
| Max grad norm | 1.0 | **0.3** | Aggressive clipping prevents gradient spikes |
| Epochs | 3 | **3** (peaks at epoch 3) | Similar, but instruct degrades faster after peak |
| Loss | MatryoshkaLoss(MNRL) | **Plain MNRL** | Matryoshka's 6 loss signals amplify instability |

**What happened with aggressive settings on the instruct model:**
- LR 5e-6 → Collapse
- LR 1e-6 + MatryoshkaLoss → Collapse
- LR 1e-6 + plain MNRL + grad_norm 0.3 → **Stable** ✅

---

## GradCache (CachedMultipleNegativesRankingLoss)

### The Problem
Standard MNRL uses in-batch negatives: the number of negatives each query sees equals `micro_batch_size - 1`. On RTX 5090 with fp32 + MatryoshkaLoss, max micro-batch is only 8 → just 7 in-batch negatives. Industry recommends 64-128.

### The Solution
GradCache decouples the contrastive pool size from VRAM usage:
- `per_device_train_batch_size = 128` → 127 in-batch negatives (contrastive quality)
- `mini_batch_size = 4` → only 4 samples in VRAM at a time (memory control)

It works in 3 steps:
1. Embed all 128 samples in mini-batches of 4, **without gradients** (cheap)
2. Compute the full 128×128 similarity matrix and loss (tiny — just floats)
3. Re-embed in mini-batches of 4 **with gradients**, chain cached gradients into backward

Trade-off: ~20% slower (every sample embedded twice), but huge gain in contrastive quality.

### Results

| Config | In-batch neg | Stage 1 | Stage 2 | Best |
|--------|-------------|---------|---------|------|
| RTX batch 8, standard MNRL | 7 | 0.9327 | **0.9492** | **0.9492** |
| RTX batch 128, CachedMNRL | 127 | 0.9422 | 0.9463 | 0.9463 |
| Colab batch 64, standard MNRL | 63 | 0.9436 | 0.9465 | 0.9465 |

### Key Findings
- **More in-batch negatives → better Stage 1** (0.9327 → 0.9422, +0.95%)
- **But diminishing Stage 2 gains** — hard negatives added only +0.004 with 127 negatives vs +0.0165 with 7 negatives
- When Stage 1 already has strong contrastive signal from many negatives, Stage 2 hard negatives provide less marginal value
- With fewer in-batch negatives, Stage 2 hard negatives compensate more aggressively
- **Net effect: pipeline total is similar regardless of in-batch negative count** — hard negatives are a great equalizer
- Diminishing returns above 64 negatives — 127 didn't beat Colab's 63 for the same reason

### VRAM: mini_batch_size selection
With fp32 + eager attention + MatryoshkaLoss (6 dims):
- `mini_batch_size = 32` → OOM
- `mini_batch_size = 16` → OOM
- `mini_batch_size = 4` → fits ✅

The re-embed step (step 3) holds computation graphs, and MatryoshkaLoss runs 6 GradCache cycles.

---

## Hard Negative Count: Quality vs Quantity

### What are hard negatives?

After Stage 1 (in-batch negatives only), we mine "hard negatives" — passages the fine-tuned model thinks are similar to a query but are actually wrong. These are the model's most confusing mistakes, and training on them forces it to learn finer distinctions.

### How many hard negatives per query?

The `n_negatives` parameter in `mine_negatives.py` controls how many hard negatives each query gets. With `n_negatives=5`, the dataset has columns `(anchor, positive, negative_1, negative_2, negative_3, negative_4, negative_5)`. MNRL treats all columns after `positive` as explicit negatives.

### Experimental results

| Hard neg/query | In-batch neg | Stage 2 NDCG@10 | Δ from Stage 1 |
|---------------|-------------|-----------------|----------------|
| 1 | 127 (CachedMNRL) | 0.9463 | +0.0041 |
| 5 | 127 (CachedMNRL) | 0.9398 | -0.0025 |
| 1 | 7 (standard MNRL) | **0.9492** | +0.0165 |

### Why more hard negatives hurt

1. **Signal imbalance:** 1 positive vs 5 hard + 127 in-batch = 132 negatives. Positive signal is drowned out.
2. **Diminishing negative quality:** Top-1 negative is genuinely confusing; top-5 includes less informative cases that add noise.
3. **Overfitting:** 5 × 1,944 queries = 9,720 negative pairs. With a small dataset, the model memorizes specific negative patterns instead of learning general distinctions.

### Rule of thumb

For datasets < 10K pairs: **1 hard negative per query.** Quality of the hardest negative matters more than quantity. This aligns with sentence-transformers defaults and most embedding fine-tuning literature.

---

## Quick Reference: RTX 5090 Training Config

```bash
# Always use this environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

```python
# Forced by Blackwell GPU constraints
fp32 = True          # bf16 causes gradient explosion and collapse
bf16 = False
fp16 = False
attn_implementation = "eager"    # SDPA partially unstable on Blackwell
gradient_checkpointing = False   # recomputation causes numerical collapse

# Max batch sizes (32GB VRAM, fp32, no checkpointing)
# Plain MNRL (2 columns):              batch 16
# MatryoshkaLoss (2 columns):          batch 8
# Plain MNRL + hard neg (3 columns):   batch 8
# MatryoshkaLoss + hard neg (3 cols):  batch 4
# Use gradient accumulation to reach desired effective batch size

# CachedMNRL (GradCache) — decouples contrastive pool from VRAM
# per_device_train_batch_size = 128   # contrastive pool (127 in-batch negatives)
# mini_batch_size = 4                 # actual VRAM usage per forward pass
# gradient_accumulation_steps = 1     # no longer needed
```

---

*Last updated: March 2026. Includes GradCache experiments and Stage 2 results.*
