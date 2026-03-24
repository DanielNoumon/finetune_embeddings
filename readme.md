# Fine-tuning Embeddings for Dutch EU Regulations

Fine-tunes multilingual embedding models for semantic search and retrieval on Dutch EU regulations (EU AI Act, GDPR) using synthetic query-chunk pairs.

## Project Structure

```
.
├── data/
│   ├── documents/              # Source PDFs (EU AI Act, GDPR)
│   ├── chunks/                 # Chunked text output per document
│   │   ├── eu_ai_act/          #   535 chunks (recitals + articles + annexes)
│   │   └── gdpr/               #   378 chunks (recitals + articles)
│   ├── synthetic/              # Generated query-chunk pairs (gitignored)
│   └── processed/              # Train/eval splits (gitignored)
├── synthetic_dataset_creation/
│   ├── chunker.py              # Semantic hierarchical chunking (multi-doc)
│   ├── generate_queries.py     # Synthetic query generation (any OpenAI-compatible LLM)
│   ├── prepare_hf_dataset.py   # Convert to HF format
│   ├── view_dataset.py         # Dataset viewer utility
│   └── analyze_queries.py      # Query quality analysis
├── evaluation/
│   └── eval_openai.py          # Benchmark vs text-embedding-3-large
├── finetuning/
│   ├── e5_large_stage1/        # e5-large: Stage 1 (MNRL)
│   │   ├── finetune_stage1.py
│   │   ├── diagnose.py         # Precision/attention diagnostics
│   │   └── diagnose_instructions.py
│   ├── e5_large_stage2/        # e5-large: Stage 2 (hard negatives)
│   │   ├── mine_negatives.py
│   │   └── finetune_stage2.py
│   ├── qwen3/                  # Qwen3-Embedding-0.6B (full fine-tune)
│   │   ├── eval_baseline.py    # Zero-shot evaluation
│   │   ├── finetune_stage1.py
│   │   └── finetune_stage2.py
│   └── qwen3_4b/               # Qwen3-Embedding-4B (LoRA)
│       ├── finetune_stage1.py
│       ├── mine_negatives.py
│       ├── finetune_stage2.py
│       └── upload_to_hf.py     # Merge LoRA + upload to HuggingFace
├── upload_to_hf/
│   ├── upload_dataset.py       # Upload dataset to HuggingFace
│   ├── upload_readme.py        # Upload dataset card
│   └── DATASET_CARD.md         # Complete dataset documentation
├── cards/
│   ├── model_cards/            # Model documentation for HuggingFace
│   └── dataset_cards/          # Dataset documentation (gitignored)
└── learnings/                  # Personal notes & process docs (gitignored)
```

## Workflow

### 1. Document Chunking
```bash
uv run python synthetic_dataset_creation/chunker.py
```
Chunks EU regulation PDFs using semantic hierarchical chunking. Supports multiple documents — set `DOC` in the config section (`"eu_ai_act"`, `"gdpr"`, or `"all"`).

### 2. Synthetic Query Generation
```bash
uv run python synthetic_dataset_creation/generate_queries.py
```
Generates diverse Dutch queries per chunk using any OpenAI-compatible endpoint (Ollama, vLLM, Azure OpenAI, OpenAI, etc.). Set `DOC`, `MODEL`, and `BASE_URL` in the config section.

### 3. Dataset Preparation
```bash
uv run python synthetic_dataset_creation/prepare_hf_dataset.py
```
Converts query pairs to HuggingFace Dataset format with train/eval splits.

### 4. Fine-tuning

**e5-large:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetuning/e5_large_stage1/finetune_stage1.py
python finetuning/e5_large_stage2/mine_negatives.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetuning/e5_large_stage2/finetune_stage2.py
```

**Qwen3-Embedding-0.6B:**
```bash
python finetuning/qwen3/eval_baseline.py
python finetuning/qwen3/finetune_stage1.py
python finetuning/qwen3/mine_negatives.py
python finetuning/qwen3/finetune_stage2.py
```

**Qwen3-Embedding-4B (LoRA):**
```bash
python finetuning/qwen3_4b/finetune_stage1.py
python finetuning/qwen3_4b/mine_negatives.py
python finetuning/qwen3_4b/finetune_stage2.py
```

### 5. Upload to HuggingFace
```bash
# Dataset
uv run python upload_to_hf/upload_dataset.py --repo-id "username/dataset-name"
uv run python upload_to_hf/upload_readme.py --repo-id "username/dataset-name"

# Qwen3-4B: merges LoRA adapters and uploads merged model
python finetuning/qwen3_4b/upload_to_hf.py
```

## Dataset

**Published at:** [danielnoumon/eu-ai-act-nl-queries](https://huggingface.co/datasets/danielnoumon/eu-ai-act-nl-queries)

- Synthetic Dutch query-chunk pairs from EU AI Act and GDPR
- For embedding fine-tuning, semantic search, and RAG applications

## Results

NDCG@10 at dim=1024 on held-out eval set (340 queries, 85 chunks):

| Model | Stage 1 | Stage 2 | HuggingFace |
|-------|---------|---------|-------------|
| text-embedding-3-large (zero-shot) | — | 0.8556 | — |
| multilingual-e5-large (batch 8, MNRL) | 0.9327 | **0.9492** | — |
| multilingual-e5-large (batch 128, CachedMNRL) | 0.9422 | 0.9463 | — |
| Qwen3-Embedding-0.6B (CachedMNRL) | 0.9419 | 0.9467 | [danielnoumon/qwen3-embedding-0.6b-ai-act-nl](https://huggingface.co/danielnoumon/qwen3-embedding-0.6b-ai-act-nl) |
| **Qwen3-Embedding-4B LoRA (CachedMNRL)** | **0.9631** | **0.9658** | [danielnoumon/qwen3-embedding-4b-ai-act-nl](https://huggingface.co/danielnoumon/qwen3-embedding-4b-ai-act-nl) |

Zero-shot baselines: e5-large 0.8612, Qwen3-0.6B 0.8013, Qwen3-4B not measured.

## Configuration

Create a `.env` file (gitignored) with the relevant credentials:
```
# Query generation — any OpenAI-compatible endpoint
# Option A: Local model (Ollama, vLLM, etc.)
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:30b-a3b
# Option B: OpenAI / Azure OpenAI
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o-mini
# LLM_API_KEY=sk-...

# Evaluation — OpenAI text-embedding-3-large benchmark (optional)
EMBEDDINGS_AZURE_OPENAI_ENDPOINT=your_endpoint
EMBEDDINGS_AZURE_OPENAI_KEY=your_key

# HuggingFace uploads (optional)
HF_TOKEN=hf_your_token_here
```
