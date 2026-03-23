"""
Upload the prepared dataset to Hugging Face Hub.

This script loads the prepared HF dataset and pushes it to the Hub.
Requires authentication via HF token.
"""

import os
from pathlib import Path
from datasets import load_from_disk
from huggingface_hub import HfApi, login


def upload_dataset(
    dataset_path: str,
    repo_id: str,
    private: bool = False,
    token: str | None = None
):
    """
    Upload dataset to Hugging Face Hub.
    
    Args:
        dataset_path: Path to the prepared HF dataset directory
        repo_id: Repository ID on HF Hub (e.g., 'username/dataset-name')
        private: Whether to make the dataset private
        token: HF token (if not provided, will use HF_TOKEN env var or login)
    """
    # Authenticate
    if token:
        login(token=token)
    elif os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    else:
        print("No token provided. Please login to Hugging Face:")
        login()
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    print(f"\nDataset info:")
    print(f"  Rows: {len(dataset)}")
    print(f"  Columns: {dataset.column_names}")
    print(f"  Features: {dataset.features}")
    
    # Upload to Hub
    print(f"\nUploading to: {repo_id}")
    print(f"  Private: {private}")
    
    dataset.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message="Upload EU AI Act NL query-chunk pairs dataset"
    )
    
    print(f"\n✓ Dataset uploaded successfully!")
    print(f"  View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # CONFIG: Adjust these parameters for your use case
    # -----------------------------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Path to the prepared HF dataset directory
    DATASET_PATH = PROJECT_ROOT / "data" / "hf_dataset"

    # HF repository ID
    REPO_ID = "danielnoumon/eu-regulations-nl-queries"

    # Dataset card to upload as README.md
    DATASET_CARD = PROJECT_ROOT / "upload_to_hf" / "DATASET_CARD_eu_regulations_combined.md"

    # Whether to make the dataset private
    PRIVATE = False

    # -----------------------------------------------------------------------

    upload_dataset(
        dataset_path=str(DATASET_PATH),
        repo_id=REPO_ID,
        private=PRIVATE,
    )

    # Upload dataset card as README.md
    if DATASET_CARD.exists():
        print(f"\nUploading dataset card: {DATASET_CARD.name}")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(DATASET_CARD),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Upload dataset card",
        )
        print(f"Dataset card uploaded.")
