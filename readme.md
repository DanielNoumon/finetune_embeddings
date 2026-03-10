# Fine-tuning Embeddings for EU AI Act (NL)

This project fine-tunes embedding models for semantic search and retrieval on the Dutch EU AI Act using synthetic query-chunk pairs.

##### TO ADD
Comparison of:
- OpenAI embeddings
- OpenAI embeddings + adapter
- SOTA open-source embeddings
- Multilingual embeddings base
- Multilingual embeddings fine-tuned
- Matryoshka embeddings

## Project Structure

```
.
├── data/
│   ├── documents/          # Source PDFs
│   ├── chunks/            # Chunked text output
│   ├── synthetic/         # Generated query-chunk pairs (gitignored)
│   └── hf_dataset/        # HuggingFace dataset format (gitignored)
├── synthetic_dataset_creation/
│   ├── chunker.py         # Semantic hierarchical chunking
│   ├── generate_queries.py   # Synthetic query generation
│   ├── prepare_hf_dataset.py # Convert to HF format
│   ├── view_dataset.py    # Dataset viewer utility
│   └── analyze_queries.py # Query quality analysis
├── upload_to_hf/
│   ├── upload_dataset.py  # Upload dataset to HuggingFace
│   ├── upload_readme.py   # Upload dataset card
│   └── DATASET_CARD.md    # Complete dataset documentation
└── process.md             # Detailed process documentation

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
Converts query pairs to HuggingFace Dataset format with metadata.

### 4. Upload to HuggingFace
```bash
# Upload dataset
uv run python upload_to_hf/upload_dataset.py --repo-id "username/dataset-name"

# Upload dataset card
uv run python upload_to_hf/upload_readme.py --repo-id "username/dataset-name"
```

## Dataset

**Published at:** [danielnoumon/eu-ai-act-nl-queries](https://huggingface.co/datasets/danielnoumon/eu-ai-act-nl-queries)

- 2,284 synthetic Dutch query-chunk pairs
- Source: EU AI Act (Dutch translation)
- For embedding fine-tuning, semantic search, and RAG applications

## Configuration

Create a `.env` file with your Azure OpenAI credentials:
```
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT_GPT5_MINI=your_endpoint_here
DEPLOYMENT_NAME_GPT5_MINI=your_deployment_name
API_VERSION_GPT5_MINI=2024-10-01-preview
HF_TOKEN=hf_your_token_here  # Optional, for HF uploads
```

## Documentation

See `process.md` for detailed explanations of:
- Chunking strategy and parameters
- Query generation prompt design
- Training format and batch size rationale
- Evaluation approach
