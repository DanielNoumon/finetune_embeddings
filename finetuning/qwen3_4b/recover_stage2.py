"""
Recovery script: evaluate checkpoint-39 from interrupted Stage 2 run
and save the best merged model to final/.

Run from project root:
  python finetuning/qwen3_4b/recover_stage2.py
"""

import json
import torch
from pathlib import Path
from finetuning.qwen3_4b.finetune_stage2 import (
    detect_device,
    load_eval_data,
    load_stage1_model,
    build_evaluators,
    select_and_save_best_checkpoint,
    print_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
STAGE1_DIR = PROJECT_ROOT / "models" / "qwen3_4b_stage1" / "final"
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen3_4b_stage2"
FINAL_DIR = OUTPUT_DIR / "final"
EVAL_DIR = PROJECT_ROOT / "data" / "processed" / "eval"

QUERY_PROMPT = (
    "Instruct: Given a question about Dutch data protection "
    "and AI regulation, retrieve the most relevant passage\nQuery:"
)
CORPUS_PROMPT = ""
MATRYOSHKA_DIMS = [2560, 1024, 768, 512, 256, 128]


if __name__ == "__main__":
    device, use_fp16, use_bf16 = detect_device()

    # Load eval data
    queries, corpus, relevant_docs = load_eval_data(EVAL_DIR)
    print(f"Eval: {len(queries)} queries, {len(corpus)} corpus chunks")

    # Build evaluators
    eval_suite, primary_metric = build_evaluators(
        queries, corpus, relevant_docs, MATRYOSHKA_DIMS,
        QUERY_PROMPT, CORPUS_PROMPT,
    )

    # Get Stage 1 baseline for comparison
    print("\n--- Evaluating Stage 1 baseline ---")
    stage1_model = load_stage1_model(STAGE1_DIR, BASE_MODEL_NAME)
    base_results = eval_suite(stage1_model)
    print(f"Stage 1 {primary_metric}: {base_results[primary_metric]:.4f}")
    del stage1_model
    torch.cuda.empty_cache()

    # Evaluate checkpoint(s) and save best as final
    print("\n--- Evaluating Stage 2 checkpoint(s) ---")
    best_results = select_and_save_best_checkpoint(
        output_dir=OUTPUT_DIR,
        stage1_dir=STAGE1_DIR,
        base_model_name=BASE_MODEL_NAME,
        eval_suite=eval_suite,
        primary_metric=primary_metric,
        final_path=FINAL_DIR,
    )

    # Print comparison
    print_summary(base_results, best_results, MATRYOSHKA_DIMS)
