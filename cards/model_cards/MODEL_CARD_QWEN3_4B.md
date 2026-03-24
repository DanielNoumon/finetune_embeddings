---
language:
- nl
- en
license: apache-2.0
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- loss:MatryoshkaLoss
- loss:CachedMultipleNegativesRankingLoss
- peft
- lora
base_model: Qwen/Qwen3-Embedding-4B
datasets:
- danielnoumon/eu-ai-act-nl-queries
metrics:
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
- cosine_accuracy@1
- cosine_accuracy@10
- cosine_recall@10
library_name: sentence-transformers
pipeline_tag: sentence-similarity
---

# Qwen3-Embedding-4B EU AI Act NL

Fine-tuned **Qwen3-Embedding-4B** with LoRA for Dutch/English retrieval on [EU AI Act documentation](https://huggingface.co/datasets/danielnoumon/eu-ai-act-nl-queries). Supports **Matryoshka embeddings** (2560, 1024, 768, 512, 256, 128 dimensions) for flexible speed/quality tradeoffs.

## Model Details

- **Base model**: [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- **Architecture**: Decoder-based (Qwen3), last-token pooling, left padding
- **Training approach**: Two-stage fine-tuning with LoRA (r=16, alpha=32)
  - **Stage 1**: CachedMNRL + Matryoshka on synthetic query-chunk pairs
  - **Stage 2**: CachedMNRL + Matryoshka with hard negatives mined from Stage 1 model
- **Dataset**: 1,944 synthetic queries generated from EU AI Act chunks (Dutch/English)
- **Hardware**: NVIDIA RTX 5090 (32GB VRAM, Blackwell)
- **Precision**: bf16 + SDPA

## Performance

Evaluated on 340 held-out queries across 85 chunks. All metrics measured with cosine similarity.

### NDCG@10 across Matryoshka dimensions

| Dim | Stage 1 | Stage 2 | Delta (S1 to S2) |
|-----|---------|---------|------------------|
| 2560 | 0.9626 | **0.9616** | -0.0010 |
| 1024 | 0.9631 | **0.9658** | +0.0027 |
| 768 | 0.9616 | **0.9609** | -0.0007 |
| 512 | 0.9537 | **0.9526** | -0.0011 |
| 256 | 0.9410 | **0.9420** | +0.0010 |
| 128 | 0.9186 | **0.9188** | +0.0002 |

Stage 2 results are essentially equivalent to Stage 1 — the model was already near-ceiling after Stage 1 on this dataset.

### Full metrics at dim=2560

| Metric | Stage 2 |
|--------|---------|
| NDCG@10 | **0.9616** |
| MRR@10 | **0.9500** |
| MAP@100 | **0.9502** |
| Accuracy@1 | **0.9235** |
| Accuracy@3 | **0.9676** |
| Accuracy@5 | **0.9941** |
| Accuracy@10 | **0.9971** |
| Recall@10 | **0.9971** |

### Full metrics at dim=1024

| Metric | Stage 2 |
|--------|---------|
| NDCG@10 | **0.9658** |
| MRR@10 | **0.9546** |
| MAP@100 | **0.9546** |
| Accuracy@1 | **0.9265** |
| Accuracy@3 | **0.9765** |
| Accuracy@5 | **0.9941** |
| Accuracy@10 | **1.0000** |
| Recall@10 | **1.0000** |

## Usage

### Installation

```bash
pip install sentence-transformers>=2.7.0 transformers>=4.51.0
```

### Basic usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("danielnoumon/qwen3-embedding-4b-ai-act-nl")

# Qwen3 uses instruct prompts for queries, no prefix for documents
queries = model.encode(
    ["What are the obligations for high-risk AI systems?"],
    prompt="Instruct: Given a question about EU AI regulation, retrieve the most relevant passage\nQuery:",
)
passages = model.encode([
    "High-risk AI systems must comply with requirements in Chapter III...",
    "The AI Act defines prohibited practices in Article 5...",
])

# Compute similarity
from sentence_transformers.util import cos_sim
scores = cos_sim(queries, passages)
```

### Matryoshka embeddings (dimension truncation)

```python
# Encode with full 2560 dimensions
embeddings_2560 = model.encode(queries)

# Truncate to 256 dimensions for faster search
embeddings_256 = embeddings_2560[:, :256]

# Or specify dimension at encoding time
model.truncate_dim = 256
embeddings_256 = model.encode(queries)
```

**Speed vs quality tradeoff**:
- **dim=2560**: Full quality (NDCG@10 = 0.962)
- **dim=1024**: Marginally better subspace (NDCG@10 = 0.966) — Matryoshka effect
- **dim=256**: ~90% fewer dimensions, 97.9% of dim=1024 quality (NDCG@10 = 0.942)
- **dim=128**: ~95% fewer dimensions, 95.4% of dim=1024 quality (NDCG@10 = 0.919)

### Important: Use instruct prompts

Qwen3 uses instruction-based prompting. Queries need the instruct prefix, documents do not:

```python
# Queries: use instruct prompt
query_emb = model.encode(
    ["your question here"],
    prompt="Instruct: Given a question about EU AI regulation, retrieve the most relevant passage\nQuery:",
)

# Documents: no prefix needed
doc_emb = model.encode(["your document here"])
```

## Training Details

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params | 11.8M / 4034M (0.29%) |

### Stage 1: CachedMNRL + Matryoshka

- **Loss**: `MatryoshkaLoss(CachedMultipleNegativesRankingLoss)`
- **Matryoshka dims**: [2560, 1024, 768, 512, 256, 128]
- **Batch size**: 128 (GradCache), mini-batch 2
- **Learning rate**: 1e-4 (typical for LoRA)
- **Epochs**: 3
- **Negatives**: 127 in-batch negatives per query (via GradCache)
- **Precision**: bf16 + SDPA

### Stage 2: Hard negatives

- **Starting point**: Stage 1 LoRA checkpoint (merged into base)
- **Hard negative mining**: Top-1 most similar wrong chunk per query (using Stage 1 model)
- **Learning rate**: 1e-5 (10× lower to prevent catastrophic forgetting)
- **Epochs**: 2
- **Batch size**: 128 (GradCache), mini-batch 1
- **Negatives**: 1 explicit hard negative + 127 in-batch negatives

### Dataset

- **Dataset**: [danielnoumon/eu-ai-act-nl-queries](https://huggingface.co/datasets/danielnoumon/eu-ai-act-nl-queries)
- **Train**: 1,944 synthetic query-chunk pairs
- **Eval**: 340 queries × 85 chunks
- **Split strategy**: Chunk-level (no chunk appears in both train and eval)
- **Query generation**: GPT-5-mini (Azure OpenAI) with structured prompts

### Hardware Notes

- **bf16 works on Blackwell (RTX 5090)** with Qwen3
- Qwen3's RMSNorm upcasts to fp32 internally
- CachedMNRL (GradCache) essential for fitting large contrastive pools in 32GB VRAM with a 4B model
- LoRA keeps trainable params at 0.29% — full fine-tuning would exceed 32GB VRAM
- flash_attention_2 recommended but not required (sdpa works as fallback)

## Limitations

- **Domain-specific**: Fine-tuned on EU AI Act documentation. Performance on other domains may vary.
- **Language**: Optimized for Dutch and English. Other languages supported by the base model may work but are not evaluated.
- **Chunk size**: Trained on chunks up to 512 tokens. Very long documents should be chunked.

## License

Apache 2.0

## Citation

```bibtex
@misc{qwen3embedding,
    title={Qwen3-Embedding: Advancing Text Embeddings with Qwen3},
    author={Qwen Team},
    year={2025},
    url={https://huggingface.co/Qwen/Qwen3-Embedding-4B}
}

@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    year = "2019",
    url = "https://arxiv.org/abs/1908.10084",
}

@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

@misc{hu2022lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
    year={2022},
    eprint={2106.09685},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
