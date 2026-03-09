# HuggingFace Upload Scripts

This folder contains scripts for uploading the EU AI Act dataset and documentation to HuggingFace Hub.

## Files

- **`upload_dataset.py`** — Upload the prepared dataset to HuggingFace
- **`upload_readme.py`** — Upload the dataset card (README.md) to HuggingFace
- **`DATASET_CARD.md`** — Complete dataset documentation with metadata

## Usage

### Upload Dataset

```bash
uv run python upload_to_hf/upload_dataset.py --repo-id "username/dataset-name"
```

### Upload Dataset Card

```bash
uv run python upload_to_hf/upload_readme.py --repo-id "username/dataset-name"
```

## Authentication

Scripts will use existing HuggingFace credentials or prompt for login if needed.

You can also set `HF_TOKEN` environment variable in `.env`:
```
HF_TOKEN=hf_your_token_here
```
