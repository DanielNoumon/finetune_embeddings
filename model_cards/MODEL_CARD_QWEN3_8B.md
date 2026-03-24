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
base_model: Qwen/Qwen3-Embedding-8B
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

# Qwen3-Embedding-8B EU AI Act NL

Fine-tuned **Qwen3-Embedding-8B** with LoRA for Dutch/English retrieval on [EU AI Act documentation](https://huggingface.co/datasets/danielnoumon/eu-ai-act-nl-queries). Supports **Matryoshka embeddings** (4096, 1024, 768, 512, 256, 128 dimensions) for flexible speed/quality tradeoffs.

## Model Details

- **Base model**: [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- **Architecture**: Decoder-based (Qwen3), last-token pooling, left padding
- **Training approach**: Two-stage fine-tuning with LoRA (r=16, alpha=32)
  - **Stage 1**: CachedMNRL + Matryoshka on synthetic query-chunk pairs
  - **Stage 2**: CachedMNRL + Matryoshka with hard negatives mined from Stage 1 model
- **Dataset**: 1,944 synthetic queries generated from EU AI Act chunks (Dutch/English)
- **Hardware**: NVIDIA RTX 5090 (32GB VRAM, Blackwell)
- **Precision**: bf16 + flash_attention_2

## Performance

Evaluated on 340 held-out queries across 85 chunks. All metrics measured with cosine similarity.

### NDCG@10 across Matryoshka dimensions

| Dim | Zero-shot | Stage 1 | Stage 2 | Best |
|-----|-----------|---------|---------|------|
| 4096 | 0.8962 | **0.9682** | 0.9675 | **0.9682** (S1) |
| 1024 | 0.8836 | 0.9625 | **0.9625** | **0.9625** (tie) |
| 768 | 0.8825 | 0.9607 | **0.9629** | **0.9629** (S2) |
| 512 | 0.8774 | 0.9577 | **0.9587** | **0.9587** (S2) |
| 256 | 0.8704 | 0.9524 | **0.9535** | **0.9535** (S2) |
| 128 | 0.8369 | 0.9238 | **0.9253** | **0.9253** (S2) |

Fine-tuning on ~2,000 synthetic pairs improves retrieval quality by +7–9 NDCG@10 points across all Matryoshka dimensions.

### Stage 2 analysis

Stage 2 (hard negative mining) produced marginal improvements at lower dimensions but a slight regression at the primary dim=4096 metric. With 127 in-batch negatives from GradCache already providing a strong contrastive signal, explicit hard negatives offer diminishing returns — consistent with findings on the smaller 0.6B and 4B models. The uploaded model uses the **Stage 1 checkpoint** (best at the primary dim=4096 metric).

## Usage

### Installation

```bash
pip install sentence-transformers>=2.7.0 transformers>=4.51.0
```

### Basic usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("danielnoumon/qwen3-embedding-8b-ai-act-nl")

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
# Encode with full 4096 dimensions
embeddings_4096 = model.encode(queries)

# Truncate to 256 dimensions for faster search
embeddings_256 = embeddings_4096[:, :256]

# Or specify dimension at encoding time
model.truncate_dim = 256
embeddings_256 = model.encode(queries)
```

**Speed vs quality tradeoff**:
- **dim=4096**: Full quality (NDCG@10 = 0.968)
- **dim=1024**: 75% fewer dimensions, 99.4% of full quality (NDCG@10 = 0.963)
- **dim=256**: ~94% fewer dimensions, 98.4% of full quality (NDCG@10 = 0.952)
- **dim=128**: ~97% fewer dimensions, 95.4% of full quality (NDCG@10 = 0.924)

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
| Trainable params | 15.3M / 7583M (0.20%) |

### Stage 1: CachedMNRL + Matryoshka

- **Loss**: `MatryoshkaLoss(CachedMultipleNegativesRankingLoss)`
- **Matryoshka dims**: [4096, 1024, 768, 512, 256, 128]
- **Batch size**: 128 (GradCache), mini-batch 1
- **Learning rate**: 1e-4 (typical for LoRA)
- **Epochs**: 3
- **Negatives**: 127 in-batch negatives per query (via GradCache)
- **Precision**: bf16 + flash_attention_2

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
- **Query generation**: Azure OpenAI GPT-4o-mini with structured prompts

### Hardware Notes

- **bf16 works on Blackwell (RTX 5090)** with Qwen3
- Qwen3's RMSNorm upcasts to fp32 internally, preventing gradient instability
- flash_attention_2 used for O(N) attention memory
- CachedMNRL (GradCache) essential: 8B model fills 16GB of 32GB VRAM with base weights alone
- LoRA keeps trainable params at 0.20% — full fine-tuning impossible on 32GB
- mini_batch_size=1 required due to extreme VRAM pressure

## Limitations

- **Domain-specific**: Fine-tuned on EU AI Act documentation. Performance on other domains may vary.
- **Language**: Optimized for Dutch and English. Other languages supported by the base model may work but are not evaluated.
- **Chunk size**: Trained on chunks up to 512 tokens. Very long documents should be chunked.
- **VRAM**: Requires ~16GB VRAM for inference in bf16. Use quantization or CPU offloading for smaller GPUs.

## License

Apache 2.0

## Citation

```bibtex
@misc{qwen3embedding,
    title={Qwen3-Embedding: Advancing Text Embeddings with Qwen3},
    author={Qwen Team},
    year={2025},
    url={https://huggingface.co/Qwen/Qwen3-Embedding-8B}
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
