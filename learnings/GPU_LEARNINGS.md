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

## Concept: LoRA (Low-Rank Adaptation)

LoRA is a technique for fine-tuning large models by only training small "adapter" matrices instead of updating the full weights. The core idea:

Every weight matrix in the model has shape `[in, out]`. Instead of updating all `in × out` parameters, LoRA inserts two smaller matrices:

```
W_update = A × B

Where:
  A: [in × r]   — trainable
  B: [r × out]  — trainable
  r: rank (e.g. 16) << min(in, out)
```

During training, only `A` and `B` are updated. The base weights `W_original` are frozen. At inference (or before Stage 2), the adapters are **merged**: `W_final = W_original + (A × B)`.

**Why this is memory-efficient:**

For a 4B model with ~4034M parameters, the Adam optimizer would normally store:
- Gradients: ~4034M × 2 bytes = ~8 GB
- Optimizer states (momentum + variance, fp32): ~4034M × 8 bytes = ~32 GB
- **Total: ~40 GB extra on top of the 8 GB model**

With LoRA (r=16, targeting q/k/v/o projections, 36 layers):
- Trainable params: ~11.8M (0.29%)
- Optimizer states: ~11.8M × 8 bytes = ~95 MB
- **Total extra: ~95 MB** — negligible

**Key parameters:**

| Param | What it does | Typical value |
|-------|-------------|---------------|
| `r` (rank) | Subspace dimensionality — higher = more capacity | 8–64 |
| `alpha` | Scaling factor. Effective scale = `alpha / r`. Usually set to `2r` | 2× rank |
| `dropout` | Regularization on adapter weights | 0.0–0.1 |
| `target_modules` | Which weight matrices to adapt | `q_proj, k_proj, v_proj, o_proj` |

**Why these specific values for our project:**

| Param | Value | Rationale |
|-------|-------|-----------|
| `r=16` | 16 | Standard starting point for domain adaptation. r=8 risks underfitting (too few dimensions to capture the Dutch legal domain shift), r=32/64 doubles/quadruples trainable params with diminishing returns on ~2000 training pairs. r=16 is the most common default in LoRA literature and worked well without tuning. |
| `alpha=32` | 2×r | Controls the effective learning rate of the adapter: `scale = alpha / r = 2.0`. Setting alpha=2r is the standard convention from the original LoRA paper — it means the adapter contribution is scaled by 2× relative to r=1. Higher alpha would make the adapter updates too aggressive; lower would require more epochs. |
| `dropout=0.05` | 0.05 | Light regularization. With only ~2000 training pairs and 11–15M trainable params, some regularization helps prevent overfitting, but heavy dropout (0.1+) would slow convergence on our small dataset. |
| `target_modules` | q,k,v,o_proj | Targets all four attention projection matrices. This is standard for embedding models where the attention mechanism is the primary driver of semantic understanding. Skipping MLP layers (gate_proj, up_proj, down_proj) keeps trainable params low while covering the most impactful weights. |

**Same LoRA config for 4B and 8B:** Both models use identical LoRA hyperparameters (r=16, alpha=32, dropout=0.05, same targets). The only differences are operational — mini_batch_size and eval_batch_size are smaller on 8B due to VRAM constraints. The LoRA config didn't need per-model tuning because:
1. Both are Qwen3-Embedding family (same architecture, just different layer counts)
2. The rank-16 subspace is proportionally *smaller* on the 8B model (0.20% vs 0.29%), which is fine — domain adaptation doesn't need large updates
3. Results validated this: 8B matched or beat 4B at every dimension without config changes

**Trade-off vs full fine-tuning:** LoRA constrains the weight update to a low-rank subspace. For domain adaptation (our task), this is sufficient. For tasks requiring large representational shifts from the base model, full fine-tuning is stronger. On ~2000 training pairs, LoRA prevents overfitting by limiting the parameter space.

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

## Qwen3-Embedding on Blackwell: bf16 Works

Everything above (bf16 collapse, gradient checkpointing instability, SDPA issues) was specific to `multilingual-e5-large`. When we switched to **Qwen3-Embedding**, the picture changed entirely.

**Qwen3 uses bf16 stably on RTX 5090.** The reason: Qwen3's `RMSNorm` layers upcast inputs to fp32 internally before computing normalization, then cast back to bf16. This keeps the numerically sensitive normalization operations precise even when the rest of the model runs in bf16. The gradient explosion that destroyed e5-large training doesn't occur.

**flash_attention_2 is not available** (not installed), so it falls back to SDPA. Unlike e5-large, SDPA + bf16 is stable with Qwen3.

**Effective config for Qwen3-0.6B (full fine-tuning, 32GB):**

```python
bf16 = True
fp32 = False
attn_implementation = "sdpa"     # flash_attention_2 fallback
gradient_checkpointing = False   # not needed — bf16 gives enough headroom

# CachedMNRL
per_device_train_batch_size = 128
mini_batch_size = 4              # 0.6B + MatryoshkaLoss (6 dims)
```

---

## Qwen3-Embedding-4B with LoRA on Blackwell

The 4B model is too large for full fine-tuning on 32GB. With LoRA (r=16, targeting q/k/v/o_proj):
- Base weights: ~8 GB (frozen, no gradients)
- Trainable LoRA weights: 11.8M params (~0.05 GB)
- Adam optimizer for LoRA weights: ~0.1 GB
- **Fixed overhead: ~8.15 GB** vs ~40 GB for full fine-tuning

This leaves ~24 GB for activations — plenty for CachedMNRL with mini-batch processing.

### mini_batch_size scaling with model size and column count

The 4B model is ~7× the size of 0.6B. CachedMNRL's backward re-embed step holds intermediate activations for each sequence in the mini-batch. With 4B, each sequence is more expensive:

| Stage | Columns | mini_batch_size | VRAM result |
|-------|---------|----------------|-------------|
| 0.6B Stage 1 | 2 | 4 | Fits ✅ |
| 0.6B Stage 2 | 3 | 4 | Fits ✅ |
| 4B Stage 1 | 2 | 2 | Fits ✅ (~58s/step) |
| 4B Stage 2 | 3 | 2 | OOM ❌ |
| 4B Stage 2 | 3 | **1** | Fits ✅ |

**Why Stage 2 needs half the mini_batch_size of Stage 1:** During the backward re-embed pass, CachedMNRL holds a computation graph for every column (anchor, positive, + N negatives). Stage 2 adds one negative column: 2 columns → 3 columns = 50% more graph memory. Halving mini_batch_size restores the budget.

**Rule of thumb:** When adding hard negative columns in Stage 2, halve mini_batch_size from Stage 1's value.

### load_best_model_at_end is broken with PEFT + SentenceTransformer

HuggingFace Trainer's `load_best_model_at_end=True` fails silently when training a SentenceTransformer wrapping a PEFT model. The Trainer tries to restore the best checkpoint using a generic `model.load_state_dict()` call, but PEFT saves weights with the prefix `base_model.model.*` while SentenceTransformer's Transformer module expects unprefixed keys. The mismatch causes the load to fail, and the model in memory at the end of training (the last epoch, not the best) gets saved.

**Fix:** Set `load_best_model_at_end=False`. After `trainer.train()`, iterate over saved checkpoints, load each by applying the adapter to the Stage 1 merged base, evaluate, and save the best.

---

## Quick Reference: RTX 5090 Training Config

```bash
# Always use this environment variable (for e5-large fp32 runs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### multilingual-e5-large (fp32 required on Blackwell)

```python
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

# CachedMNRL (GradCache)
# per_device_train_batch_size = 128
# mini_batch_size = 4
```

### Qwen3-Embedding-0.6B (full fine-tuning, bf16 works)

```python
bf16 = True          # Qwen3 RMSNorm fp32 upcast prevents collapse
fp32 = False
attn_implementation = "sdpa"     # flash_attention_2 not installed, sdpa stable
gradient_checkpointing = False

# CachedMNRL
per_device_train_batch_size = 128
mini_batch_size = 4              # Stage 1 (2 cols) and Stage 2 (3 cols) both fit
learning_rate = 2e-5             # Stage 1
learning_rate = 1e-5             # Stage 2 (lower to prevent catastrophic forgetting)
```

### Qwen3-Embedding-4B (LoRA, bf16)

```python
bf16 = True
attn_implementation = "sdpa"
gradient_checkpointing = False

# LoRA config
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
# → 11.8M / 4034M trainable (0.29%)

# CachedMNRL
per_device_train_batch_size = 128
mini_batch_size = 2              # Stage 1 (2 cols) — ~58s/step
mini_batch_size = 1              # Stage 2 (3 cols) — halve when adding negatives column
learning_rate = 1e-4             # Stage 1 (higher LR typical for LoRA)
learning_rate = 1e-5             # Stage 2

# load_best_model_at_end = False  # broken with PEFT+ST — use manual checkpoint selection
```

---

## Qwen3-Embedding-8B with LoRA on Blackwell

The 8B model pushes the RTX 5090's 32GB VRAM to its absolute limit. With LoRA (r=16, targeting q/k/v/o_proj):
- Base weights (bf16): ~16 GB (frozen, no gradients)
- Trainable LoRA weights: 15.3M params (~0.03 GB)
- Adam optimizer for LoRA weights: ~0.12 GB
- **Fixed overhead: ~16.15 GB** — leaving only ~16 GB for activations and PyTorch overhead

### flash_attention_2 now available

After installing `flash-attn`, the model loads with `flash_attention_2` instead of falling back to SDPA. Flash attention computes attention in O(N) memory instead of O(N²), which saves meaningful VRAM at long sequence lengths. However, for the 8B model, the savings were not enough to increase mini_batch_size.

### mini_batch_size: everything must be 1

The 8B model is 2× larger than the 4B model. Even with flash_attention_2, every batch size > 1 OOMs:

| Setting | mini_batch=2 | mini_batch=1 |
|---------|-------------|-------------|
| Stage 1 training (2 cols) | OOM ❌ | Fits ✅ |
| Stage 1 eval (during training) | OOM ❌ | Fits ✅ |
| Stage 2 training (3 cols) | — | Fits ✅ |
| Inference (no training state) | eval_batch=2 ✅ | eval_batch=1 ✅ |

**Key nuance:** `eval_batch_size=2` works for standalone inference (just model weights in VRAM), but OOMs during in-trainer evaluation because training state (optimizer, gradients, cached embeddings) still occupies memory. We had to set `EVAL_BATCH_SIZE=1` for Stage 1 after the script crashed at the end-of-epoch eval.

### PEFT wrapper overhead during inference

Loading a model as `base + PeftModel.from_pretrained()` adds memory overhead from the adapter wrapper: duplicate weight references, extra bookkeeping tensors, and the LoRA computation graph. For the 8B model, this overhead is enough to cause OOM during inference even with eval_batch=1.

**Fix:** Always `merge_and_unload()` before evaluation:

```python
peft_model = PeftModel.from_pretrained(inner, adapter_path)
merged = peft_model.merge_and_unload()  # Merges LoRA into base weights
model[0].auto_model = merged
del peft_model
torch.cuda.empty_cache()
# Now evaluate — model is same size as base with no PEFT overhead
```

This also applies to the recovery/upload scripts: never evaluate with the PEFT wrapper active on the 8B model.

### Checkpoint recovery

Training completed all 3 epochs but the script OOM'd during the final evaluation (before `save_pretrained`). The Trainer's `save_strategy="epoch"` saved checkpoints at each epoch. Recovery approach:

1. Load base model
2. For each checkpoint: load adapter → merge → evaluate → save if best → delete from GPU
3. Critical: delete each model from GPU before loading the next (two 8B models = 32GB = full VRAM)

`save_total_limit=2` meant only checkpoint-32 (epoch 2) and checkpoint-48 (epoch 3) were available. Checkpoint-48 won (0.9682 vs 0.9650 NDCG@10 at dim=4096).

### Results (NDCG@10 on EU AI Act eval set, dim=4096)

| Stage | NDCG@10 | Δ from zero-shot |
|-------|---------|-----------------|
| Zero-shot | 0.8962 | — |
| Stage 1 (3 epochs, checkpoint-48) | **0.9682** | +0.072 |
| Stage 2 (2 epochs, checkpoint-16) | 0.9675 | +0.071 |

**Stage 2 did not improve over Stage 1** at the primary dim=4096 metric. At lower dims (768, 512, 256, 128) Stage 2 showed marginal gains (+0.001–0.002), but the primary metric regressed by 0.0007. With 127 in-batch negatives from GradCache, the contrastive signal is already saturated — explicit hard negatives provide no additional benefit. This is consistent with findings on 0.6B and 4B.

**Uploaded model uses Stage 1 checkpoint.**

Comparison with 4B (at dim=1024 for apple-to-apple):
- 8B zero-shot: 0.8836
- 4B zero-shot: ~0.88 (EU AI Act eval)
- 8B Stage 1: 0.9625
- 4B Stage 2 (best): 0.9658

The 8B model's zero-shot baseline is already higher, and Stage 1 alone nearly matches the 4B's best Stage 2 result.

### Quick reference config

```python
bf16 = True
attn_implementation = "flash_attention_2"  # installed; falls back to sdpa/eager
gradient_checkpointing = False

# LoRA config
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
# → 15.3M / 7583M trainable (0.20%)

# CachedMNRL
per_device_train_batch_size = 128
mini_batch_size = 1              # Both stages — 8B leaves no headroom
eval_batch_size = 1              # Must be 1 during in-trainer eval
learning_rate = 1e-4             # Stage 1
learning_rate = 1e-5             # Stage 2

# load_best_model_at_end = False  # broken with PEFT+ST
# Always merge_and_unload() before evaluation (PEFT overhead causes OOM)
```

---

## Training Duration & Energy Consumption

All training on a single NVIDIA RTX 5090 (32GB VRAM, 575W TDP).

### Per-model training time

| Model | Stage | Trainable | Samples | Format | Epochs | ~Time/epoch | ~Total |
|-------|-------|-----------|---------|--------|--------|-------------|--------|
| Qwen3-0.6B | Stage 1 | 560M (full) | 1,944 | pairs | 3 | 3.6 min | 11 min |
| Qwen3-0.6B | Stage 2 | 560M (full) | 1,944 | triplets | 2 | 7 min | 14 min |
| Qwen3-4B (LoRA) | Stage 1 | 11.8M | 1,944 | pairs | 3 | 13.3 min | 40 min |
| Qwen3-4B (LoRA) | Stage 2 | 11.8M | 1,944 | triplets | 2 | 24.9 min | 50 min |
| Qwen3-8B (LoRA) | Stage 1 | 15.3M | 1,944 | pairs | 3 | 18.3 min | 55 min |
| Qwen3-8B (LoRA) | Stage 2 | 15.3M | 1,944 | triplets | 2 | 32.3 min | 65 min |

**Grand total training time: ~235 min ≈ 3.9 hours**

**Why is 0.6B (full fine-tuning) faster than 4B/8B (LoRA) despite training 50× more parameters?** LoRA saves **memory**, not compute. Every training step still runs a full forward and backward pass through the entire model — the frozen weights are still involved in every computation. LoRA only reduces the weight *update* step (tiny fraction of total time) and the optimizer state memory. Training time scales with total model size, not trainable parameters.

Stage 2 takes ~1.8–1.9× longer per epoch than Stage 1 despite the same sample count: triplets require encoding 3 texts per sample (query + positive + hard negative) vs 2 for pairs, resulting in ~50% more forward passes through the model.

Times estimated from checkpoint directory timestamps (1-epoch gap between saved checkpoints). Excludes evaluation, hard negative mining, and checkpoint recovery.

### Energy estimate

RTX 5090 TDP is 575W; estimated ~500W average draw during training, ~100W for CPU/RAM/fans.

- **Training only (GPU+system)**: 3.9h × 0.60 kW ≈ **2.34 kWh**
- **Conservative total** (incl. eval, hard negative mining, checkpoint recovery): **~3 kWh**

### Cost: local GPU vs cloud

**Local electricity cost (Amsterdam, ~€0.25/kWh):**

- Conservative total: 3 kWh × €0.25 = **€0.75** for the entire project (all 3 models, both stages)

**Equivalent cloud cost (single H100 at ~$2.95/hr):**

An H100 has significantly higher memory bandwidth and tensor core throughput than an RTX 5090. The same training would likely complete in roughly 1–1.5 hours instead of 3.9 hours, putting a single full experiment cycle at **~$3–5**.

But a single cycle is misleading. Real projects involve many iterations:

| Activity | Typical runs | Est. cloud cost |
|----------|-------------|-----------------|
| Failed runs (bugs, config errors) | 10–20 | $30–60 |
| Hyperparameter tuning | 5–10 | $15–30 |
| Successful final runs | 6 (3 models × 2 stages) | $20–30 |
| **Total realistic project cost** | **20–50 runs** | **$100–250** |

On top of that: storage for model checkpoints (~46 GB across all models), and debugging time while the meter is running.

**The hidden cost of cloud: iteration speed.** On a local GPU you can start a run, spot an issue after 5 minutes, kill it, fix it, restart — at zero marginal cost. On cloud, every mistake costs money, which changes your behavior: you become more cautious, batch fewer experiments, and iterate more slowly. This "cost anxiety" tax on iteration speed is arguably the largest hidden cost of cloud training for research and experimentation.

**Bottom line:** Local 5090 cost for this entire project ≈ **€0.75**. Equivalent cloud cost including realistic iteration ≈ **$100–250**. The local GPU pays for itself quickly when experimentation speed matters.

### Disk footprint of saved models

| Directory | Size |
|-----------|------|
| qwen3_stage1 (0.6B) | 7.9 GB |
| qwen3_stage2 (0.6B) | 7.9 GB |
| qwen3_4b_stage1 | 349 MB |
| qwen3_4b_stage2 | 349 MB |
| qwen3_8b_stage1 | 15 GB |
| qwen3_8b_stage2 | 15 GB |
| **Total** | **~46 GB** |

The 4B directories are small because only LoRA adapter weights are saved (~15M params). The 8B directories are large because the `final/` subdirectory contains the fully merged model (LoRA merged back into base weights → full 8B saved to disk for upload). The 0.6B was fine-tuned without LoRA, so full model weights are saved in every checkpoint.

---

*Last updated: March 2026. Includes Qwen3-0.6B, Qwen3-4B LoRA, Qwen3-8B LoRA, PEFT checkpoint findings, and training duration/energy estimates.*
