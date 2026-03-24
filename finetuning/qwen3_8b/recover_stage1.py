"""
Recovery script: load Stage 1 checkpoints, evaluate, save the best
as the final model. Use this when training completed but the script
crashed before save_pretrained (e.g. OOM during final eval).
"""

import json
import torch
from pathlib import Path
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
STAGE1_DIR = PROJECT_ROOT / "models" / "qwen3_8b_stage1"
EVAL_DIR = PROJECT_ROOT / "data" / "processed" / "eval"
FINAL_PATH = STAGE1_DIR / "final"

QUERY_PROMPT = (
    "Instruct: Given a question about EU AI regulation, "
    "retrieve the most relevant passage\nQuery:"
)
MATRYOSHKA_DIMS = [4096, 1024, 768, 512, 256, 128]
PRIMARY_METRIC = "eu-ai-act-nl-dim4096_cosine_ndcg@10"


def load_eval_data(eval_dir):
    with open(eval_dir / "queries.json", encoding="utf-8") as f:
        queries = json.load(f)
    with open(eval_dir / "corpus.json", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(eval_dir / "relevant_docs.json", encoding="utf-8") as f:
        raw = json.load(f)
    return queries, corpus, {q: set(c) for q, c in raw.items()}


def load_base_model():
    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            print(f"  Trying {attn}...")
            m = SentenceTransformer(
                BASE_MODEL_NAME,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": attn,
                },
                tokenizer_kwargs={"padding_side": "left"},
            )
            print(f"  Loaded with {attn} + bf16")
            return m
        except (ImportError, ValueError) as e:
            print(f"  {attn} failed: {e}")
    raise RuntimeError("All attention backends failed.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    queries, corpus, relevant_docs = load_eval_data(EVAL_DIR)
    evaluators = [
        InformationRetrievalEvaluator(
            queries=queries, corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"eu-ai-act-nl-dim{dim}",
            truncate_dim=dim,
            query_prompt=QUERY_PROMPT,
            show_progress_bar=True,
        )
        for dim in MATRYOSHKA_DIMS
    ]
    eval_suite = SequentialEvaluator(evaluators)

    # Find checkpoints
    ckpt_dirs = sorted(
        [d for d in STAGE1_DIR.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    print(f"Found checkpoints: {[d.name for d in ckpt_dirs]}")

    best_score = -1.0
    best_model = None

    for ckpt_dir in ckpt_dirs:
        # ST trainer saves adapter inside 0_Transformer/
        adapter_dir = ckpt_dir
        if (ckpt_dir / "0_Transformer" / "adapter_config.json").exists():
            adapter_dir = ckpt_dir / "0_Transformer"
        elif not (ckpt_dir / "adapter_config.json").exists():
            print(f"  Skipping {ckpt_dir.name}: no adapter found")
            continue

        print(f"\nEvaluating {ckpt_dir.name}...")
        model = load_base_model()
        inner = model[0].auto_model
        peft_model = PeftModel.from_pretrained(
            inner, str(adapter_dir)
        )
        model[0].auto_model = peft_model

        results = eval_suite(model)
        score = results[PRIMARY_METRIC]
        print(f"  {PRIMARY_METRIC}: {score:.4f}")

        for dim in MATRYOSHKA_DIMS:
            k = f"eu-ai-act-nl-dim{dim}_cosine_ndcg@10"
            print(f"    dim={dim}: {results[k]:.4f}")

        if score > best_score:
            best_score = score
            merged = peft_model.merge_and_unload()
            model[0].auto_model = merged
            best_model = model
            print(f"  New best! Saving to {FINAL_PATH}")

        if best_model is not model:
            del model
            torch.cuda.empty_cache()

    if best_model is None:
        raise RuntimeError("No valid checkpoints found.")

    best_model.save_pretrained(str(FINAL_PATH))
    print(f"\nFinal model saved to: {FINAL_PATH}")
    print(f"Best {PRIMARY_METRIC}: {best_score:.4f}")
