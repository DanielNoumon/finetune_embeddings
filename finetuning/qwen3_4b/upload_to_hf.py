"""
Merge Stage 2 LoRA into base weights and upload to HuggingFace.

The Stage 2 checkpoints store only the LoRA adapter delta weights.
This script:
  1. Loads the Stage 1 merged model (base + Stage 1 LoRA baked in)
  2. Applies the best Stage 2 adapter (checkpoint with highest NDCG@10)
  3. Merges Stage 2 LoRA into weights
  4. Uploads the clean merged SentenceTransformer to HuggingFace

Run from the project root:
    python finetuning/qwen3_4b/upload_to_hf.py
"""

import torch
from pathlib import Path
from peft import PeftModel
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
import json

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_MODEL_NAME   = "Qwen/Qwen3-Embedding-4B"
STAGE1_DIR        = PROJECT_ROOT / "models" / "qwen3_4b_stage1" / "final"
STAGE2_DIR        = PROJECT_ROOT / "models" / "qwen3_4b_stage2"
EVAL_DIR          = PROJECT_ROOT / "data" / "processed" / "eval"

HF_REPO_ID        = "danielnoumon/qwen3-embedding-4b-ai-act-nl"
MODEL_CARD_PATH   = PROJECT_ROOT / "model_cards" / "MODEL_CARD_QWEN3_4B.md"

QUERY_PROMPT = (
    "Instruct: Given a question about EU AI regulation, "
    "retrieve the most relevant passage\nQuery:"
)
MATRYOSHKA_DIMS = [2560, 1024, 768, 512, 256, 128]
PRIMARY_METRIC = "eu-ai-act-nl-dim2560_cosine_ndcg@10"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_eval_data(eval_dir):
    with open(eval_dir / "queries.json", encoding="utf-8") as f:
        queries = json.load(f)
    with open(eval_dir / "corpus.json", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(eval_dir / "relevant_docs.json", encoding="utf-8") as f:
        relevant_docs_raw = json.load(f)
    relevant_docs = {qid: set(cids) for qid, cids in relevant_docs_raw.items()}
    return queries, corpus, relevant_docs


def load_st_bf16(model_name, card=None):
    kwargs = dict(
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )
    if card:
        kwargs["model_card_data"] = card
    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            return SentenceTransformer(
                model_name,
                model_kwargs={**kwargs["model_kwargs"], "attn_implementation": attn},
                tokenizer_kwargs=kwargs["tokenizer_kwargs"],
                **({"model_card_data": card} if card else {}),
            )
        except (ImportError, ValueError):
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


def load_stage1_merged():
    """Load Stage 1 checkpoint. Merge LoRA if saved in PEFT format."""
    adapter_cfg = STAGE1_DIR / "adapter_config.json"
    if adapter_cfg.exists():
        print(f"Stage 1 is PEFT format — loading base + merging...")
        base = load_st_bf16(BASE_MODEL_NAME)
        peft = PeftModel.from_pretrained(base[0].auto_model, str(STAGE1_DIR))
        base[0].auto_model = peft.merge_and_unload()
        print("  Stage 1 LoRA merged.")
        return base
    else:
        print(f"Stage 1 is merged format — loading directly...")
        return load_st_bf16(str(STAGE1_DIR))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    queries, corpus, relevant_docs = load_eval_data(EVAL_DIR)
    eval_suite = build_eval_suite(queries, corpus, relevant_docs)

    # Find all Stage 2 epoch checkpoints
    ckpt_dirs = sorted(
        [d for d in STAGE2_DIR.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    print(f"\nFound {len(ckpt_dirs)} Stage 2 checkpoints: "
          f"{[d.name for d in ckpt_dirs]}")

    best_score = -1.0
    best_model = None

    for ckpt_dir in ckpt_dirs:
        # Trainer saves ST structure: adapter is under 0_Transformer/
        adapter_dir = ckpt_dir / "0_Transformer"
        if not (adapter_dir / "adapter_config.json").exists():
            print(f"  Skipping {ckpt_dir.name}: no adapter found at {adapter_dir}")
            continue

        print(f"\nEvaluating {ckpt_dir.name}...")
        model = load_stage1_merged()
        inner = model[0].auto_model
        peft = PeftModel.from_pretrained(inner, str(adapter_dir))
        model[0].auto_model = peft

        results = eval_suite(model)
        score = results[PRIMARY_METRIC]
        print(f"  {PRIMARY_METRIC}: {score:.4f}")

        if score > best_score:
            best_score = score
            # Merge LoRA into weights before uploading
            merged = peft.merge_and_unload()
            model[0].auto_model = merged
            best_model = model
            print(f"  New best — will upload this checkpoint.")

        if best_model is not model:
            del model
            torch.cuda.empty_cache()

    if best_model is None:
        raise RuntimeError(
            "No valid Stage 2 checkpoints found. "
            "Check that checkpoint-*/0_Transformer/adapter_config.json exists."
        )

    print(f"\nBest checkpoint: {PRIMARY_METRIC} = {best_score:.4f}")

    # Final eval summary
    print("\nFinal NDCG@10 across dims:")
    for dim in MATRYOSHKA_DIMS:
        key = f"eu-ai-act-nl-dim{dim}_cosine_ndcg@10"
        print(f"  dim={dim}: {results[key]:.4f}")

    # Upload to HuggingFace
    print(f"\nUploading to {HF_REPO_ID}...")
    model_card_content = MODEL_CARD_PATH.read_text(encoding="utf-8")

    best_model.push_to_hub(
        HF_REPO_ID,
        private=False,
        exist_ok=True,
    )

    # Upload model card separately (overwrite the auto-generated one)
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Update model card",
    )

    print(f"\nDone. Model live at: https://huggingface.co/{HF_REPO_ID}")
