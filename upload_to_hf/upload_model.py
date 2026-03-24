"""
Upload a fine-tuned SentenceTransformer model to HuggingFace Hub.

Uploads the model files and a model card (README.md) to a new or
existing HF model repository.

Usage:
    python upload_to_hf/upload_model.py \
        --model-path models/qwen3_stage2/final \
        --repo-id DanielNoumon/qwen3-embedding-0.6b-ai-act-nl \
        --model-card cards/model_cards/MODEL_CARD_QWEN3_0_6B.md
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login


def upload_model(
    model_path: str,
    repo_id: str,
    model_card_path: str | None = None,
    private: bool = False,
    token: str | None = None,
):
    """Upload a SentenceTransformer model to HuggingFace Hub.

    Args:
        model_path: Path to the saved model directory.
        repo_id: HF repo ID (e.g. 'user/model-name').
        model_card_path: Optional path to a MODEL_CARD .md file
            to upload as README.md.
        private: Whether to make the repo private.
        token: HF token (or use HF_TOKEN env var).
    """
    # Authenticate
    if token:
        login(token=token)
    elif os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    else:
        print("No token provided. Please login to Hugging Face:")
        login()

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    print(f"Repository: https://huggingface.co/{repo_id}")

    # Upload model directory
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nUploading model from: {model_path}")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload fine-tuned model",
    )
    print("  Model files uploaded.")

    # Upload model card as README.md
    if model_card_path:
        card = Path(model_card_path)
        if not card.exists():
            raise FileNotFoundError(
                f"Model card not found: {card}"
            )
        print(f"\nUploading model card: {card}")
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        print("  Model card uploaded.")

    print(f"\nDone! View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Upload model to HuggingFace Hub"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HF repo ID (e.g. 'user/model-name')",
    )
    parser.add_argument(
        "--model-card",
        type=str,
        default=None,
        help="Path to model card .md file",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the repo private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    upload_model(
        model_path=args.model_path,
        repo_id=args.repo_id,
        model_card_path=args.model_card,
        private=args.private,
        token=args.token,
    )
