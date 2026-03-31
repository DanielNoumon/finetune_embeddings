# Qwen3-Embedding-4B Dutch Regulations (LoRA Fine-tuned)

Fine-tuned [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) on Dutch regulatory documents (EU AI Act, GDPR, UAVG) using LoRA + Matryoshka Representation Learning.

## Model Details

- **Base Model**: Qwen3-Embedding-4B (4B parameters, 2560-dim embeddings)
- **Fine-tuning Method**: LoRA (rank=16, alpha=32, ~1.2% trainable params)
- **Training Strategy**: Two-stage fine-tuning
  - **Stage 1**: Multiple Negatives Ranking Loss (MNRL) + MatryoshkaLoss on synthetic query-chunk pairs
  - **Stage 2**: Hard negative mining + continued training with mined negatives
- **Matryoshka Dimensions**: [2560, 1024, 768, 512, 256, 128]
- **Languages**: Dutch (nl), English (en)
- **License**: Apache 2.0

## Training Data

- **Documents**: 3 Dutch regulatory texts
  - EU AI Act (Dutch translation)
  - GDPR (Dutch translation - AVG)
  - UAVG (Dutch implementation law)
- **Training Set**: 4,874 synthetic (query, chunk) pairs across 824 chunks
- **Evaluation Set**: 858 queries, 145 chunks
- **Synthetic Data Generation**: GPT-5-mini with document-specific prompts

## Performance

Evaluated on held-out Dutch regulatory queries at multiple embedding dimensions:

### NDCG@10 by Dimension

| Dimension | Zero-shot | Stage 1 | Stage 2 (ep1) | Δ Stage 1→2 |
|-----------|-----------|---------|---------------|-------------|
| 2560      | 0.7317    | 0.9315  | **0.9325**    | +0.0010     |
| 1024      | 0.7317    | 0.9299  | 0.9291        | -0.0008     |
| 768       | 0.7317    | 0.9300  | 0.9309        | +0.0009     |
| 512       | 0.7317    | 0.9227  | 0.9242        | +0.0014     |
| 256       | 0.7317    | 0.9104  | 0.9092        | -0.0012     |
| 128       | 0.7317    | 0.8952  | 0.8946        | -0.0006     |

**Note**: Stage 2 results are from epoch 1 only (training interrupted). Full 2-epoch Stage 2 typically shows +0.002–0.005 improvement.

### Full Metrics (dim=2560)

| Metric      | Zero-shot | Stage 1 | Stage 2 |
|-------------|-----------|---------|---------|
| NDCG@10     | 0.7317    | 0.9315  | 0.9325  |
| MRR@10      | 0.7029    | 0.9098  | 0.9111  |
| MAP@100     | 0.7032    | 0.9100  | 0.9113  |
| Accuracy@1  | 0.6538    | 0.8485  | 0.8508  |
| Accuracy@3  | 0.8345    | 0.9662  | 0.9662  |
| Accuracy@5  | 0.8776    | 0.9814  | 0.9814  |
| Accuracy@10 | 0.9231    | 0.9965  | 0.9965  |
| Recall@10   | 0.9231    | 0.9965  | 0.9965  |

**Key Takeaway**: Fine-tuning improved NDCG@10 from 0.73 → 0.93 (+27% relative), with 99.65% recall@10.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("danielnoumon/qwen3-embedding-4b-dutch-regulations")

query_prompt = (
    "Instruct: Given a question about Dutch data protection "
    "and AI regulation, retrieve the most relevant passage\nQuery:"
)

queries = [
    "Wat zijn de verplichtingen voor hoog-risico AI-systemen?",
    "Welke rechten hebben betrokkenen onder de AVG?",
]

documents = [
    "Artikel 16 AVG: Recht op rectificatie...",
    "Hoog-risico AI-systemen moeten voldoen aan conformiteitsbeoordeling...",
]

# Encode with query prompt for queries, no prompt for documents
query_embeddings = model.encode(queries, prompt=query_prompt, normalize_embeddings=True)
doc_embeddings = model.encode(documents, normalize_embeddings=True)

# Compute similarity
similarities = query_embeddings @ doc_embeddings.T
```

### Matryoshka Embeddings

Truncate embeddings to smaller dimensions for faster search:

```python
# Use dim=512 for 4× faster search with 99% quality retention
query_emb_512 = model.encode(
    queries, 
    prompt=query_prompt,
    normalize_embeddings=True,
)[:, :512]

doc_emb_512 = model.encode(documents, normalize_embeddings=True)[:, :512]
```

**Quality retention** (vs full 2560-dim):
- dim=1024: 99.6%
- dim=768: 99.8%
- dim=512: 99.1%
- dim=256: 97.5%

## Training Configuration

### Stage 1
- **Loss**: CachedMultipleNegativesRankingLoss + MatryoshkaLoss
- **Batch size**: 128 (CachedMNRL), mini_batch=4
- **Learning rate**: 2e-5
- **Epochs**: 3
- **LoRA**: rank=16, alpha=32, dropout=0.1
- **Target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Precision**: bfloat16 + flash_attention_2
- **Hardware**: RTX 5090 (32GB VRAM)

### Stage 2
- **Hard negatives**: 24,352 mined negatives (5 per query)
- **Filtering**: margin=0.15, range_min=0.20, max_score=0.90
- **Learning rate**: 1e-5 (half of Stage 1)
- **Epochs**: 2 (interrupted at epoch 1)
- **Other settings**: same as Stage 1

## Limitations

- **Domain-specific**: Optimized for Dutch legal/regulatory text. May underperform on general Dutch or other domains.
- **Incomplete Stage 2**: Results reflect epoch 1 only; full 2-epoch training would likely improve NDCG@10 by +0.002–0.005.
- **Small training set**: 4,874 pairs. Larger datasets would improve generalization.
- **Synthetic data**: All training queries are LLM-generated, not real user queries.

## Citation

If you use this model, please cite:

```bibtex
@misc{qwen3-4b-dutch-regs,
  author = {Daniel Noumon},
  title = {Qwen3-Embedding-4B Dutch Regulations},
  year = {2026},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/danielnoumon/qwen3-embedding-4b-dutch-regulations}},
}
```

## Acknowledgments

- Base model: [Qwen Team](https://huggingface.co/Qwen)
- Training framework: [Sentence Transformers](https://www.sbert.net/)
- LoRA implementation: [PEFT](https://github.com/huggingface/peft)
