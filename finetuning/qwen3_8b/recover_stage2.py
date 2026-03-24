"""
Recovery script for Stage 2: evaluate existing checkpoints and save best.

Use this when Stage 2 training completed but the post-training checkpoint
selection failed (e.g. adapter_config.json not found, OOM during eval).

This script:
  1. Lists what's inside each checkpoint directory (for debugging)
  2. Searches for adapter_config.json in multiple locations
  3. Loads Stage 1 base + checkpoint adapter, merges, evaluates
  4. Saves the best merged model to final/
"""

import torch
from pathlib import Path

from peft import PeftModel
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)

from finetune_stage2 import (
    load_eval_data,
    _find_adapter_dir,
)

# ── CONFIG ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STAGE1_DIR = PROJECT_ROOT / "models" / "qwen3_8b_stage1" / "final"
STAGE2_DIR = PROJECT_ROOT / "models" / "qwen3_8b_stage2"
FINAL_PATH = STAGE2_DIR / "final"
EVAL_DIR = PROJECT_ROOT / "data" / "processed" / "eval"
BASE_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

MATRYOSHKA_DIMS = [4096, 1024, 768, 512, 256, 128]
QUERY_PROMPT = (
    "Instruct: Given a question about EU AI regulation, "
    "retrieve the most relevant passage\nQuery:"
)
CORPUS_PROMPT = None
PRIMARY_METRIC = f"eu-ai-act-nl-dim{MATRYOSHKA_DIMS[0]}_cosine_ndcg@10"


def load_base_model():
    """Load Stage 1 merged model with flash attention fallback."""
    card = SentenceTransformerModelCardData(
        model_name="Qwen3-Embedding-8B EU AI Act NL Recovery",
    )
    model_kwargs = {"torch_dtype": torch.bfloat16}
    tokenizer_kwargs = {"padding_side": "left"}

    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            print(f"  Trying {attn}...")
            model = SentenceTransformer(
                str(STAGE1_DIR),
                model_kwargs={
                    **model_kwargs,
                    "attn_implementation": attn,
                },
                tokenizer_kwargs=tokenizer_kwargs,
                model_card_data=card,
            )
            print(f"  Loaded with {attn} + bf16")
            return model
        except (ImportError, ValueError) as e:
            print(f"  {attn} failed: {e}")
            continue
    raise RuntimeError("All attention backends failed.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    queries, corpus, relevant_docs = load_eval_data(EVAL_DIR)
    print(f"Eval queries: {len(queries)}, corpus: {len(corpus)}")

    evaluators = []
    for dim in MATRYOSHKA_DIMS:
        evaluators.append(
            InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=f"eu-ai-act-nl-dim{dim}",
                truncate_dim=dim,
                query_prompt=QUERY_PROMPT,
                corpus_prompt=CORPUS_PROMPT,
                show_progress_bar=True,
            )
        )
    eval_suite = SequentialEvaluator(evaluators)

    # ── List checkpoints ────────────────────────────────────
    ckpt_dirs = sorted(
        [d for d in STAGE2_DIR.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    print(f"\nFound checkpoints: {[d.name for d in ckpt_dirs]}")

    # List contents of each checkpoint for debugging
    for ckpt_dir in ckpt_dirs:
        print(f"\n{ckpt_dir.name}/ contents:")
        for p in sorted(ckpt_dir.iterdir()):
            if p.is_dir():
                sub = [sp.name for sp in sorted(p.iterdir())]
                print(f"  {p.name}/ → {sub}")
            else:
                print(f"  {p.name} ({p.stat().st_size} bytes)")

    # ── Evaluate each checkpoint ────────────────────────────
    best_score = -1.0
    best_ckpt = None

    for ckpt_dir in ckpt_dirs:
        adapter_dir = _find_adapter_dir(ckpt_dir)
        if adapter_dir is None:
            print(f"\n  Skipping {ckpt_dir.name}: "
                  f"no adapter_config.json found")
            continue

        print(f"\n  Adapter found at: {adapter_dir}")
        print(f"Evaluating {ckpt_dir.name}...")

        model = load_base_model()
        inner = model[0].auto_model
        peft_model = PeftModel.from_pretrained(
            inner, str(adapter_dir)
        )
        # Merge LoRA before eval to avoid PEFT overhead OOM
        merged = peft_model.merge_and_unload()
        model[0].auto_model = merged
        del peft_model
        torch.cuda.empty_cache()

        results = eval_suite(model)
        score = results[PRIMARY_METRIC]
        print(f"  {PRIMARY_METRIC}: {score:.4f}")

        for dim in MATRYOSHKA_DIMS:
            k = f"eu-ai-act-nl-dim{dim}_cosine_ndcg@10"
            print(f"    dim={dim}: {results[k]:.4f}")

        if score > best_score:
            best_score = score
            best_ckpt = ckpt_dir.name
            model.save_pretrained(str(FINAL_PATH))
            print(f"  New best! Saved to {FINAL_PATH}")

        del model
        torch.cuda.empty_cache()

    if best_score < 0:
        print("\nERROR: No valid checkpoints found.")
        print("Check the checkpoint contents listed above.")
    else:
        print(f"\nBest checkpoint: {best_ckpt}")
        print(f"Best {PRIMARY_METRIC}: {best_score:.4f}")
        print(f"Final model at: {FINAL_PATH}")
