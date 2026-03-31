"""
Upload Qwen3-4B Stage 2 model to HuggingFace Hub (private).

Requires HF_TOKEN environment variable.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "qwen3_4b_stage2" / "final"
CARD_PATH = (
    PROJECT_ROOT / "cards" / "model_cards" / "MODEL_CARD_QWEN3_4B_DUTCH_REGS.md"
)

REPO_ID = "danielnoumon/qwen3-embedding-4b-dutch-regulations"
PRIVATE = True


def upload_model():
    """Upload model and README to HuggingFace Hub."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    if not CARD_PATH.exists():
        raise FileNotFoundError(f"Model card not found: {CARD_PATH}")

    print(f"Uploading model from: {MODEL_DIR}")
    print(f"Repository: {REPO_ID}")
    print(f"Private: {PRIVATE}")

    api = HfApi(token=token)

    try:
        create_repo(
            repo_id=REPO_ID,
            private=PRIVATE,
            exist_ok=True,
            token=token,
        )
        print(f"✓ Repository created/verified: {REPO_ID}")
    except Exception as e:
        print(f"Repository creation: {e}")

    print("\nUploading model files...")
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        token=token,
    )
    print("✓ Model files uploaded")

    print("\nUploading README.md...")
    api.upload_file(
        path_or_fileobj=str(CARD_PATH),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        token=token,
    )
    print("✓ README.md uploaded")

    print(f"\n✓ Upload complete: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    upload_model()
