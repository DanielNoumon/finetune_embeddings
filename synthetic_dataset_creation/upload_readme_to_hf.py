"""
Upload README.md (dataset card) to HuggingFace dataset repository.

This script renames DATASET_CARD.md to README.md and uploads it to your
HuggingFace dataset repository.
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, login


def upload_readme(
    readme_path: str,
    repo_id: str,
    token: str | None = None
):
    """
    Upload README.md to HuggingFace dataset repository.
    
    Args:
        readme_path: Path to the README.md file
        repo_id: Repository ID on HF Hub (e.g., 'username/dataset-name')
        token: HF token (if not provided, will use HF_TOKEN env var or login)
    """
    # Authenticate
    if token:
        login(token=token)
    elif os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    else:
        print("Using existing HF credentials...")
    
    # Initialize HF API
    api = HfApi()
    
    # Upload README
    print(f"\nUploading README.md to: {repo_id}")
    print(f"  File: {readme_path}")
    
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add comprehensive dataset card"
    )
    
    print(f"\n✓ README.md uploaded successfully!")
    print(f"  View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import argparse
    
    # -----------------------------------------------------------------------
    # CONFIG: Adjust these parameters for your use case
    # -----------------------------------------------------------------------
    
    # Path to the dataset card file
    DEFAULT_README_PATH = (
        Path(__file__).resolve().parent.parent / "DATASET_CARD.md"
    )
    
    # Default repository ID (change to your username/dataset-name)
    DEFAULT_REPO_ID = "your-username/eu-ai-act-nl-queries"
    
    # -----------------------------------------------------------------------
    
    parser = argparse.ArgumentParser(
        description="Upload README.md to HuggingFace dataset"
    )
    parser.add_argument(
        "--readme-path",
        type=str,
        default=str(DEFAULT_README_PATH),
        help="Path to the README.md or DATASET_CARD.md file"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Repository ID on HF Hub (e.g., 'username/dataset-name')"
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
    
    # Check if file exists
    readme_path = Path(args.readme_path)
    if not readme_path.exists():
        print(f"ERROR: File not found: {readme_path}")
        exit(1)
    
    upload_readme(
        readme_path=str(readme_path),
        repo_id=args.repo_id,
        token=args.token
    )
