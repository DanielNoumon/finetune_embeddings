"""
Zero-shot evaluation of Qwen3-Embedding-4B on the EU AI Act NL
retrieval task.

Tests:
  1. Baseline NDCG@10 across Matryoshka dimensions
  2. bf16 stability on Blackwell GPUs (RTX 5090)
  3. Attention backend compatibility

Run:
  python finetuning/qwen3_4b/eval_baseline.py
"""

import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)


def load_eval_data(
    eval_dir: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Load eval data in InformationRetrievalEvaluator format."""
    with open(eval_dir / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(eval_dir / "corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(
        eval_dir / "relevant_docs.json", "r", encoding="utf-8"
    ) as f:
        relevant_docs_raw = json.load(f)

    relevant_docs = {
        qid: set(cids) for qid, cids in relevant_docs_raw.items()
    }
    return queries, corpus, relevant_docs


def try_load_model(model_name, attempt_label, **kwargs):
    """Try loading a model with given kwargs, return None on failure."""
    print(f"\n--- Attempt: {attempt_label} ---")
    try:
        model = SentenceTransformer(model_name, **kwargs)
        print("  Loaded successfully.")
        return model
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def build_evaluators(
    queries, corpus, relevant_docs, matryoshka_dims,
    query_prompt, corpus_prompt,
):
    """Build SequentialEvaluator with one IR evaluator per dim."""
    evaluators = []
    for dim in matryoshka_dims:
        evaluators.append(
            InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=f"eu-ai-act-nl-dim{dim}",
                truncate_dim=dim,
                query_prompt=query_prompt,
                corpus_prompt=corpus_prompt,
                show_progress_bar=True,
            )
        )
    eval_suite = SequentialEvaluator(evaluators)
    primary_metric = (
        f"eu-ai-act-nl-dim{matryoshka_dims[0]}_cosine_ndcg@10"
    )
    return eval_suite, primary_metric


if __name__ == "__main__":

    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
    EVAL_DIR = PROJECT_ROOT / "data" / "processed" / "eval"
    # 4B model: max dim 2560, test common dims
    MATRYOSHKA_DIMS = [2560, 1024, 768, 512, 256, 128]

    QUERY_PROMPT = (
        "Instruct: Given a question about EU AI regulation, "
        "retrieve the most relevant passage\nQuery:"
    )
    CORPUS_PROMPT = ""

    # -------------------------------------------------------------------
    # Device detection
    # -------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # -------------------------------------------------------------------
    # Load model — try configurations in order of preference
    # -------------------------------------------------------------------
    model = None

    if device == "cuda":
        model = try_load_model(
            MODEL_NAME,
            "bf16 + flash_attention_2",
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
            },
            tokenizer_kwargs={"padding_side": "left"},
        )

    if model is None and device == "cuda":
        model = try_load_model(
            MODEL_NAME,
            "bf16 + sdpa",
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
            },
            tokenizer_kwargs={"padding_side": "left"},
        )

    if model is None and device == "cuda":
        model = try_load_model(
            MODEL_NAME,
            "bf16 + eager",
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "eager",
            },
            tokenizer_kwargs={"padding_side": "left"},
        )

    if model is None:
        model = try_load_model(
            MODEL_NAME,
            "fp32 + eager (safe fallback)",
            model_kwargs={
                "attn_implementation": "eager",
            },
            tokenizer_kwargs={"padding_side": "left"},
        )

    if model is None:
        raise RuntimeError("All model loading attempts failed.")

    dtype = next(model.parameters()).dtype
    print(f"\n  Model dtype: {dtype}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params / 1e9:.1f}B")

    # -------------------------------------------------------------------
    # Load eval data
    # -------------------------------------------------------------------
    queries, corpus, relevant_docs = load_eval_data(EVAL_DIR)
    print(
        f"\nEval set: {len(queries)} queries, "
        f"{len(corpus)} corpus chunks"
    )

    # -------------------------------------------------------------------
    # Run evaluation
    # -------------------------------------------------------------------
    eval_suite, primary_metric = build_evaluators(
        queries, corpus, relevant_docs, MATRYOSHKA_DIMS,
        QUERY_PROMPT, CORPUS_PROMPT,
    )

    print(f"\nEvaluating {MODEL_NAME} (zero-shot)...")
    print(f"  Query prompt: {repr(QUERY_PROMPT)}")
    print(f"  Corpus prompt: {repr(CORPUS_PROMPT)}")
    results = eval_suite(model)

    # -------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------
    qwen06_best = 0.9467  # Best fine-tuned Qwen3-0.6B
    e5_best = 0.9492      # Best fine-tuned e5-large

    print(f"\n{'='*66}")
    print("Qwen3-Embedding-4B Zero-Shot — NDCG@10")
    print(f"{'='*66}")
    print(
        f"{'Dim':>6}  {'NDCG@10':>10}  "
        f"{'vs Q3-0.6B-FT':>14}  {'vs e5-FT':>10}"
    )
    print(
        f"{'-'*6}  {'-'*10}  "
        f"{'-'*14}  {'-'*10}"
    )
    for dim in MATRYOSHKA_DIMS:
        key = f"eu-ai-act-nl-dim{dim}_cosine_ndcg@10"
        score = results[key]
        vs_q3 = score - qwen06_best
        vs_e5 = score - e5_best
        print(
            f"{dim:>6}  {score:>10.4f}  "
            f"{vs_q3:>+14.4f}  {vs_e5:>+10.4f}"
        )
    print(f"{'='*66}")
    print(f"\nReference: Qwen3-0.6B fine-tuned best = {qwen06_best}")
    print(f"Reference: e5-large fine-tuned best   = {e5_best}")

    # Full metrics at primary dim
    full_dim = MATRYOSHKA_DIMS[0]
    prefix = f"eu-ai-act-nl-dim{full_dim}_cosine_"
    detail_metrics = [
        ("NDCG@10", "ndcg@10"),
        ("MRR@10", "mrr@10"),
        ("MAP@100", "map@100"),
        ("Accuracy@1", "accuracy@1"),
        ("Accuracy@3", "accuracy@3"),
        ("Accuracy@5", "accuracy@5"),
        ("Accuracy@10", "accuracy@10"),
        ("Recall@10", "recall@10"),
    ]

    print(f"\n{'='*42}")
    print(f"Full Metrics at dim={full_dim}")
    print(f"{'='*42}")
    print(f"{'Metric':<16}  {'Score':>10}")
    print(f"{'-'*16}  {'-'*10}")
    for label, suffix in detail_metrics:
        key = prefix + suffix
        score = results.get(key, 0)
        print(f"{label:<16}  {score:>10.4f}")
    print(f"{'='*42}")
