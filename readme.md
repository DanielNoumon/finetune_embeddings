# Fine-tuning Embeddings for EU AI Act (NL)

Fine-tunes multilingual embedding models for semantic search and retrieval on the Dutch EU AI Act using synthetic query-chunk pairs.

## Project Structure

```
.
├── data/
│   ├── documents/              # Source PDFs
│   ├── chunks/                 # Chunked text output
│   ├── synthetic/              # Generated query-chunk pairs (gitignored)
│   └── processed/              # Train/eval splits (gitignored)
├── synthetic_dataset_creation/
│   ├── chunker.py              # Semantic hierarchical chunking
│   ├── generate_queries.py     # Synthetic query generation
│   ├── prepare_hf_dataset.py   # Convert to HF format
│   ├── view_dataset.py         # Dataset viewer utility
│   └── analyze_queries.py      # Query quality analysis
├── finetuning/
│   ├── stage_1_mnrl/           # Stage 1: in-batch negatives (MNRL)
│   │   ├── finetune_stage1.py
│   │   ├── diagnose.py         # Precision/attention diagnostics
│   │   └── diagnose_instructions.py
│   └── stage_2_hard_neg/       # Stage 2: mined hard negatives
│       ├── mine_negatives.py
│       └── finetune_stage2.py
├── upload_to_hf/
│   ├── upload_dataset.py       # Upload dataset to HuggingFace
│   ├── upload_readme.py        # Upload dataset card
│   └── DATASET_CARD.md         # Complete dataset documentation
├── model_cards/                # Model documentation (gitignored)
└── learnings/                  # Personal notes & process docs (gitignored)
```

## Workflow

### 1. Document Chunking
```bash
uv run python synthetic_dataset_creation/chunker.py
```
Chunks the EU AI Act PDF using semantic hierarchical chunking (~573 chunks).

### 2. Synthetic Query Generation
```bash
uv run python synthetic_dataset_creation/generate_queries.py
```
Generates 4 diverse Dutch queries per chunk using GPT-5-mini (2,284 pairs total).

### 3. Dataset Preparation
```bash
uv run python synthetic_dataset_creation/prepare_hf_dataset.py
```
Converts query pairs to HuggingFace Dataset format with train/eval splits.

### 4. Fine-tuning

**Stage 1** — In-batch negatives with MatryoshkaLoss:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetuning/stage_1_mnrl/finetune_stage1.py
```

**Stage 2** — Mine hard negatives, then fine-tune with explicit negatives:
```bash
python finetuning/stage_2_hard_neg/mine_negatives.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetuning/stage_2_hard_neg/finetune_stage2.py
```

### 5. Upload to HuggingFace
```bash
uv run python upload_to_hf/upload_dataset.py --repo-id "username/dataset-name"
uv run python upload_to_hf/upload_readme.py --repo-id "username/dataset-name"
```

## Dataset

**Published at:** [danielnoumon/eu-ai-act-nl-queries](https://huggingface.co/datasets/danielnoumon/eu-ai-act-nl-queries)

- 2,284 synthetic Dutch query-chunk pairs
- Source: EU AI Act (Dutch translation)
- For embedding fine-tuning, semantic search, and RAG applications

## Results

Base model: `intfloat/multilingual-e5-large` — NDCG@10 on held-out eval set:

| Stage | NDCG@10 | Δ from base |
|-------|---------|-------------|
| Base (zero-shot) | 0.8612 | — |
| Stage 1 (MNRL + MatryoshkaLoss) | 0.9327 | +0.0715 |
| Stage 2 (+ hard negatives) | **0.9492** | **+0.0880** |

## Configuration

Create a `.env` file with your Azure OpenAI credentials:
```
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT_GPT5_MINI=your_endpoint_here
DEPLOYMENT_NAME_GPT5_MINI=your_deployment_name
API_VERSION_GPT5_MINI=2024-10-01-preview
HF_TOKEN=hf_your_token_here  # Optional, for HF uploads
```
