"""
Recovery script: evaluate checkpoint-39 from interrupted Stage 2 run
and save the best merged model to final/.

Run from project root:
  python finetuning/qwen3_4b/recover_stage2.py
"""

import torch
from pathlib import Path
from peft import PeftModel
from finetuning.qwen3_4b.finetune_stage2 import (
    detect_device,
    load_eval_data,
    load_stage1_model,
    build_evaluators,
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

    # Evaluate checkpoint-39 and save as final
    print("\n--- Evaluating Stage 2 checkpoint-39 ---")
    ckpt_dir = OUTPUT_DIR / "checkpoint-39"
    
    if not (ckpt_dir / "adapter_config.json").exists():
        raise RuntimeError(f"No adapter_config.json in {ckpt_dir}")
    
    # Load Stage 1, apply checkpoint-39 adapter, evaluate
    ckpt_model = load_stage1_model(STAGE1_DIR, BASE_MODEL_NAME)
    inner = ckpt_model[0].auto_model
    peft_model = PeftModel.from_pretrained(inner, str(ckpt_dir))
    ckpt_model[0].auto_model = peft_model
    
    best_results = eval_suite(ckpt_model)
    score = best_results[primary_metric]
    print(f"  {primary_metric}: {score:.4f}")
    
    # Merge and save
    merged = peft_model.merge_and_unload()
    ckpt_model[0].auto_model = merged
    ckpt_model.save_pretrained(str(FINAL_DIR))
    print(f"\nMerged model saved to {FINAL_DIR}")

    # Print comparison
    print_summary(base_results, best_results, MATRYOSHKA_DIMS)
