---
language:
- nl
- en
license: mit
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_generator
- loss:MatryoshkaLoss
- loss:MultipleNegativesRankingLoss
base_model: intfloat/multilingual-e5-large
datasets:
- custom
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

# multilingual-e5-large-ai-act-nl

Fine-tuned **multilingual-e5-large** for Dutch/English retrieval on EU AI Act documentation. Supports **Matryoshka embeddings** (1024, 768, 512, 256, 128, 64 dimensions) for flexible speed/quality tradeoffs.

## Model Details

- **Base model**: [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)
- **Training approach**: Two-stage fine-tuning
  - **Stage 1**: MNRL + Matryoshka on synthetic query-chunk pairs
  - **Stage 2**: MNRL + Matryoshka with hard negatives mined from Stage 1 model
- **Dataset**: 1,944 synthetic queries generated from EU AI Act chunks (Dutch/English)
- **Hardware**: Google Colab T4 GPU (16GB VRAM)
- **Training time**: ~32 minutes total (14 min Stage 1 + 18 min Stage 2)

## Performance

Evaluated on 340 held-out queries across 85 chunks. All metrics measured with cosine similarity.

### NDCG@10 across Matryoshka dimensions

| Dim | Base | Stage 1 | Stage 2 | Delta (Base to S2) |
|---|---|---|---|---|
| 1024 | 0.8612 | 0.9426 | **0.9465** | +0.0853 |
| 768 | 0.8577 | 0.9411 | **0.9445** | +0.0868 |
| 512 | 0.8495 | 0.9379 | **0.9412** | +0.0917 |
| 256 | 0.7848 | 0.9383 | **0.9423** | +0.1575 |
| 128 | 0.7283 | 0.9225 | **0.9277** | +0.1994 |
| 64 | 0.6009 | 0.9011 | **0.9058** | +0.3049 |

**Key insight**: Matryoshka training flattened the quality curve. Dim=64 retains 96% of dim=1024's quality (0.906 vs 0.947), compared to only 70% before fine-tuning.

### Full metrics at dim=1024

| Metric | Base | Stage 2 | Delta |
|---|---|---|---|
| NDCG@10 | 0.8612 | **0.9465** | +0.0853 |
| MRR@10 | 0.8315 | **0.9315** | +0.1000 |
| MAP@100 | 0.8336 | **0.9319** | +0.0983 |
| Accuracy@1 | 0.7618 | **0.8912** | +0.1294 |
| Accuracy@10 | 0.9529 | **0.9912** | +0.0383 |
| Recall@10 | 0.9529 | **0.9912** | +0.0383 |

## Usage

### Installation

```bash
pip install sentence-transformers
```

### Basic usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("DanielNoumon/multilingual-e5-large-ai-act-nl")

# Encode queries and passages with prefixes
queries = ["query: What are the obligations for high-risk AI systems?"]
passages = [
    "passage: High-risk AI systems must comply with requirements in Chapter III...",
    "passage: The AI Act defines prohibited practices in Article 5..."
]

query_emb = model.encode(queries)
passage_emb = model.encode(passages)

# Compute similarity
from sentence_transformers.util import cos_sim
scores = cos_sim(query_emb, passage_emb)
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
- **dim=1024**: Best quality (NDCG@10 = 0.947)
- **dim=256**: 75% faster, 99.6% of quality (NDCG@10 = 0.942)
- **dim=64**: 94% faster, 96% of quality (NDCG@10 = 0.906)

### Important: Use prefixes

This model requires **query:** and **passage:** prefixes (inherited from multilingual-e5-large):

```python
# ✅ Correct
queries = ["query: your question here"]
passages = ["passage: your document here"]

# ❌ Wrong (will degrade performance)
queries = ["your question here"]
passages = ["your document here"]
```

## Training Details

### Stage 1: MNRL + Matryoshka

- **Loss**: `MatryoshkaLoss(MultipleNegativesRankingLoss)`
- **Matryoshka dims**: [1024, 768, 512, 256, 128, 64]
- **Batch size**: 64 (per-device)
- **Learning rate**: 2e-5
- **Epochs**: 3
- **Negatives**: In-batch only (63 per query)
- **Optimizations**: SDPA attention + gradient checkpointing

### Stage 2: Hard negatives

- **Starting point**: Stage 1 checkpoint
- **Hard negative mining**: Top-1 most similar wrong chunk per query (using Stage 1 model)
- **Learning rate**: 1e-5 (lower to prevent catastrophic forgetting)
- **Epochs**: 2 (fewer due to stronger signal)
- **Batch size**: 16 × 4 grad accum = 64 effective (reduced due to 3-column input OOM)
- **Negatives**: 1 explicit hard negative + 15 in-batch negatives per micro-batch

### Dataset

- **Train**: 1,944 synthetic query-chunk pairs
- **Eval**: 340 queries × 85 chunks
- **Split strategy**: Chunk-level (no chunk appears in both train and eval)
- **Query generation**: Azure OpenAI GPT-5-mini with structured prompts

## Limitations

- **Domain-specific**: Fine-tuned on EU AI Act documentation. Performance on other domains may vary.
- **Language**: Optimized for Dutch and English. Other languages supported by the base model may work but are not evaluated.
- **Chunk size**: Trained on chunks up to 512 tokens. Very long documents should be chunked.

## License

MIT

## Citation

If you use this model, please cite the base model and training frameworks:

```bibtex
@misc{wang2024multilingual,
    title={Multilingual E5 Text Embeddings: A Technical Report},
    author={Liang Wang and Nan Yang and Xiaolong Huang and Linjun Yang and Rangan Majumder and Furu Wei},
    year={2024},
    eprint={2402.05672},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
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
