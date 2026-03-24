"""
Upload the best fine-tuned Qwen3-Embedding-8B model to HuggingFace.

Stage 1 is the best checkpoint (Stage 2 hard negatives did not improve
over Stage 1 with 127 GradCache in-batch negatives). This script:
  1. Loads the already-merged Stage 1 model from models/qwen3_8b_stage1/final
  2. Runs a quick eval sanity check
  3. Uploads to HuggingFace with the custom model card

Run from the project root:
    python finetuning/qwen3_8b/upload_to_hf.py
"""

import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "qwen3_8b_stage1" / "final"
EVAL_DIR = PROJECT_ROOT / "data" / "processed" / "eval"

HF_REPO_ID = "danielnoumon/qwen3-embedding-8b-ai-act-nl"
MODEL_CARD_PATH = (
    PROJECT_ROOT / "model_cards" / "MODEL_CARD_QWEN3_8B.md"
)

QUERY_PROMPT = (
    "Instruct: Given a question about EU AI regulation, "
    "retrieve the most relevant passage\nQuery:"
)
MATRYOSHKA_DIMS = [4096, 1024, 768, 512, 256, 128]
PRIMARY_METRIC = "eu-ai-act-nl-dim4096_cosine_ndcg@10"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_eval_data(eval_dir):
    with open(eval_dir / "queries.json", encoding="utf-8") as f:
        queries = json.load(f)
    with open(eval_dir / "corpus.json", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(
        eval_dir / "relevant_docs.json", encoding="utf-8"
    ) as f:
        relevant_docs_raw = json.load(f)
    relevant_docs = {
        qid: set(cids) for qid, cids in relevant_docs_raw.items()
    }
    return queries, corpus, relevant_docs


def load_model(model_dir):
    """Load merged model with flash attention fallback."""
    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            print(f"  Trying {attn}...")
            model = SentenceTransformer(
                str(model_dir),
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": attn,
                },
                tokenizer_kwargs={"padding_side": "left"},
            )
            print(f"  Loaded with {attn} + bf16")
            return model
        except (ImportError, ValueError) as e:
            print(f"  {attn} failed: {e}")
            continue
    raise RuntimeError("All attention backends failed.")


def build_eval_suite(queries, corpus, relevant_docs):
    evaluators = [
        InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"eu-ai-act-nl-dim{dim}",
            truncate_dim=dim,
            query_prompt=QUERY_PROMPT,
            show_progress_bar=True,
        )
        for dim in MATRYOSHKA_DIMS
    ]
    return SequentialEvaluator(evaluators)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Verify model card exists
    if not MODEL_CARD_PATH.exists():
        raise FileNotFoundError(
            f"Model card not found: {MODEL_CARD_PATH}\n"
            "Create it before uploading."
        )
    print(f"Model card: {MODEL_CARD_PATH}")

    # Load model
    print(f"\nLoading model from {MODEL_DIR}...")
    model = load_model(MODEL_DIR)

    # Sanity-check eval
    queries, corpus, relevant_docs = load_eval_data(EVAL_DIR)
    eval_suite = build_eval_suite(queries, corpus, relevant_docs)

    print("\nRunning sanity-check evaluation...")
    results = eval_suite(model)

    print(f"\n  {PRIMARY_METRIC}: {results[PRIMARY_METRIC]:.4f}")
    for dim in MATRYOSHKA_DIMS:
        key = f"eu-ai-act-nl-dim{dim}_cosine_ndcg@10"
        print(f"    dim={dim}: {results[key]:.4f}")

    # Upload to HuggingFace
    print(f"\nUploading to {HF_REPO_ID}...")
    model.push_to_hub(
        HF_REPO_ID,
        private=False,
        exist_ok=True,
    )

    # Upload model card (overwrite auto-generated one)
    from huggingface_hub import HfApi
    model_card_content = MODEL_CARD_PATH.read_text(
        encoding="utf-8"
    )
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Update model card",
    )

    print(
        f"\nDone. Model live at: "
        f"https://huggingface.co/{HF_REPO_ID}"
    )
