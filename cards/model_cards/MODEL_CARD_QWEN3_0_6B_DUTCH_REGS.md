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
base_model: Qwen/Qwen3-Embedding-0.6B
datasets:
- danielnoumon/eu-regulations-nl-queries
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

# Qwen3-Embedding-0.6B Dutch Regulatory Retrieval

Fine-tuned **Qwen3-Embedding-0.6B** for Dutch/English retrieval on regulatory documentation: [EU AI Act, GDPR (AVG), and UAVG](https://huggingface.co/datasets/danielnoumon/eu-regulations-nl-queries). Supports **Matryoshka embeddings** (1024, 768, 512, 256, 128, 64 dimensions) for flexible speed/quality tradeoffs.

## Model Details

- **Base model**: [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- **Architecture**: Decoder-based (Qwen3), last-token pooling, left padding
- **Training approach**: Two-stage fine-tuning
  - **Stage 1**: CachedMNRL + Matryoshka on synthetic query-chunk pairs
  - **Stage 2**: CachedMNRL + Matryoshka with hard negatives mined from Stage 1 model
- **Dataset**: 5,732 synthetic queries generated from EU AI Act, GDPR (AVG), and UAVG chunks (Dutch/English)
- **Hardware**: NVIDIA RTX 5090 (32GB VRAM, Blackwell)
- **Precision**: bf16 + flash_attention_2 (sdpa fallback)

## Performance

Evaluated on 858 held-out queries across 145 chunks from all three regulatory documents. All metrics measured with cosine similarity.

### NDCG@10 across Matryoshka dimensions

| Dim | Zero-shot | Stage 1 | Stage 2 | Delta (ZS to S2) |
|-----|-----------|---------|---------|-------------------|
| 1024 | 0.7317 | 0.9012 | **0.9036** | +0.1719 |
| 768 | 0.7323 | 0.8972 | **0.9016** | +0.1693 |
| 512 | 0.7180 | 0.8976 | **0.8995** | +0.1815 |
| 256 | 0.6681 | 0.8812 | **0.8803** | +0.2122 |
| 128 | 0.6293 | 0.8677 | **0.8693** | +0.2400 |
| 64 | 0.5436 | 0.8305 | **0.8312** | +0.2876 |

### Full metrics at dim=1024

| Metric | Zero-shot | Stage 2 | Delta |
|--------|-----------|---------|-------|
| NDCG@10 | 0.7317 | **0.9036** | +0.1719 |
| MRR@10 | 0.6856 | **0.8772** | +0.1916 |
| MAP@100 | 0.6908 | **0.8781** | +0.1873 |
| Accuracy@1 | 0.5839 | **0.8089** | +0.2250 |
| Accuracy@3 | 0.7669 | **0.9417** | +0.1748 |
| Accuracy@5 | 0.8170 | **0.9615** | +0.1445 |
| Accuracy@10 | 0.8753 | **0.9837** | +0.1084 |
| Recall@10 | 0.8753 | **0.9837** | +0.1084 |

## Usage

### Installation

```bash
pip install sentence-transformers>=2.7.0 transformers>=4.51.0
```

### Basic usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("danielnoumon/qwen3-embedding-0.6b-dutch-regulations")

# Qwen3 uses instruct prompts for queries, no prefix for documents
queries = model.encode(
    ["What are the obligations for high-risk AI systems?"],
    prompt="Instruct: Given a question about Dutch/EU regulatory documentation, retrieve the most relevant passage\nQuery:",
)
passages = model.encode([
    "High-risk AI systems must comply with requirements in Chapter III...",
    "The GDPR defines personal data processing obligations...",
])

# Compute similarity
from sentence_transformers.util import cos_sim
scores = cos_sim(queries, passages)
```

### Matryoshka embeddings (dimension truncation)

```python
# Encode with full 1024 dimensions
embeddings_1024 = model.encode(queries)

# Truncate to 256 dimensions for faster search
embeddings_256 = embeddings_1024[:, :256]

# Or specify dimension at encoding time
model.truncate_dim = 256
embeddings_256 = model.encode(queries)
```

**Speed vs quality tradeoff**:
- **dim=1024**: Best quality (NDCG@10 = 0.9036)
- **dim=256**: 75% faster, 97.4% of quality (NDCG@10 = 0.8803)
- **dim=64**: 94% faster, 91.9% of quality (NDCG@10 = 0.8312)

### Important: Use instruct prompts

Qwen3 uses instruction-based prompting. Queries need the instruct prefix, documents do not:

```python
# Queries: use instruct prompt
query_emb = model.encode(
    ["your question here"],
    prompt="Instruct: Given a question about Dutch/EU regulatory documentation, retrieve the most relevant passage\nQuery:",
)

# Documents: no prefix needed
doc_emb = model.encode(["your document here"])
```

## Training Details

### Stage 1: CachedMNRL + Matryoshka

- **Loss**: `MatryoshkaLoss(CachedMultipleNegativesRankingLoss)`
- **Matryoshka dims**: [1024, 768, 512, 256, 128, 64]
- **Batch size**: 128 (GradCache), mini-batch 4
- **Learning rate**: 2e-5
- **Epochs**: 3
- **Negatives**: 127 in-batch negatives per query (via GradCache)
- **Precision**: bf16 + flash_attention_2 (sdpa fallback)

### Stage 2: Hard negatives

- **Starting point**: Stage 1 checkpoint
- **Hard negative mining**: Top-1 most similar wrong chunk per query (using Stage 1 model)
- **Learning rate**: 1e-5 (lower to prevent catastrophic forgetting)
- **Epochs**: 2
- **Batch size**: 128 (GradCache), mini-batch 4
- **Negatives**: 1 explicit hard negative + 127 in-batch negatives

### Dataset

- **Dataset**: [danielnoumon/eu-regulations-nl-queries](https://huggingface.co/datasets/danielnoumon/eu-regulations-nl-queries)
- **Source documents**: EU AI Act (NL), GDPR/AVG (NL), UAVG (NL)
- **Train**: 4,874 synthetic query-chunk pairs (824 unique chunks)
- **Eval**: 858 queries across 145 chunks
- **Split strategy**: Chunk-level (no chunk appears in both train and eval)
- **Query generation**: Qwen3-30B-MoE (local) with structured prompts

### Hardware Notes

- **bf16 works on Blackwell (RTX 5090)** with Qwen3
- Qwen3's RMSNorm upcasts to fp32 internally, limiting micro-batch size
- CachedMNRL (GradCache) essential for fitting large contrastive pools in 32GB VRAM
- flash_attention_2 recommended but not required (sdpa works as fallback)

## Limitations

- **Domain-specific**: Fine-tuned on Dutch/EU regulatory documentation (AI Act, GDPR, UAVG). Performance on other domains may vary.
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
    url={https://huggingface.co/Qwen/Qwen3-Embedding-0.6B}
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
```
