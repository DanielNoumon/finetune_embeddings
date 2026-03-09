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
    import argparse
    
    # -----------------------------------------------------------------------
    # CONFIG: Adjust these parameters for your use case
    # -----------------------------------------------------------------------
    
    # Default dataset path
    DEFAULT_DATASET_PATH = (
        Path(__file__).resolve().parent.parent
        / "data" / "hf_dataset"
    )
    
    # Default repository ID (change to your username/dataset-name)
    DEFAULT_REPO_ID = "your-username/eu-ai-act-nl-queries"
    
    # Whether to make the dataset private by default
    DEFAULT_PRIVATE = False
    
    # -----------------------------------------------------------------------
    
    parser = argparse.ArgumentParser(
        description="Upload dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the prepared HF dataset directory"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Repository ID on HF Hub (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=DEFAULT_PRIVATE,
        help="Make the dataset private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (or set HF_TOKEN env var)"
    )
    args = parser.parse_args()
    
    # Validate repo_id
    if args.repo_id == DEFAULT_REPO_ID:
        print("ERROR: Please specify --repo-id with your HF username/dataset-name")
        print("Example: --repo-id 'danielnoumon/eu-ai-act-nl-queries'")
        exit(1)
    
    upload_dataset(
        dataset_path=args.dataset_path,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token
    )
