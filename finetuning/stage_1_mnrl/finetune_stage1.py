"""
Fine-tune multilingual-e5-large on EU AI Act query-chunk pairs
using Matryoshka + MNRL.

Matryoshka Representation Learning wraps MNRL so the model learns
useful embeddings at multiple dimensionalities (1024, 768, 512, 256,
128, 64). At inference time, you can truncate embeddings to any of
these sizes for a speed/quality tradeoff — no retraining needed.

Supports two prefix modes (toggle INSTRUCT in config):
  - default:  queries → "query: ", passages → "passage: "
  - instruct: queries → instruction prefix, passages → no prefix

Designed for RTX 5090 (32GB VRAM).
Also works on CPU (slower) — auto-detects device.
"""

import json
import torch
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import (
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)


def load_eval_data(
    eval_dir: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Load eval data in InformationRetrievalEvaluator format.

    Returns:
        queries:       {qid: query_text}
        corpus:        {cid: chunk_text}
        relevant_docs: {qid: set(cid)}
    """
    with open(eval_dir / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(eval_dir / "corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(eval_dir / "relevant_docs.json", "r", encoding="utf-8") as f:
        relevant_docs_raw = json.load(f)

    # Convert lists back to sets (JSON doesn't support sets)
    relevant_docs = {qid: set(cids) for qid, cids in relevant_docs_raw.items()}

    return queries, corpus, relevant_docs


def detect_device():
    """Detect compute device and precision support.

    Returns (device, use_fp16, use_bf16).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device == "cuda" and not use_bf16
    print(f"Device: {device}")
    if device == "cpu":
        print("  Training on CPU \u2014 this will be slow (~2-4 hours).")
        print("  Consider using a GPU for faster training.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"  Precision: {'bf16' if use_bf16 else 'fp16'}")
    return device, use_fp16, use_bf16


def build_prompts(instruct, instruct_prefix):
    """Return (query_prompt, corpus_prompt) based on model type."""
    if instruct:
        return instruct_prefix, ""
    return "query: ", "passage: "


def load_model(model_name, instruct=False):
    """Load SentenceTransformer with SDPA attention."""
    print(f"\nLoading model: {model_name}")
    model_variant = "Instruct" if instruct else "Standard"
    return SentenceTransformer(
        model_name,
        model_kwargs={"attn_implementation": "sdpa"},
        model_card_data=SentenceTransformerModelCardData(
            language="nl",
            license="apache-2.0",
            model_name=(
                f"multilingual-e5-large {model_variant} EU AI Act NL Matryoshka"
            ),
        ),
    )


def build_evaluators(
    queries, corpus, relevant_docs, matryoshka_dims,
    query_prompt, corpus_prompt,
):
    """Build SequentialEvaluator with one IR evaluator per dim.

    Returns (eval_suite, primary_metric).
    """
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


def build_loss(model, matryoshka_dims):
    """Build MatryoshkaLoss wrapping MNRL."""
    inner_loss = MultipleNegativesRankingLoss(model)
    return MatryoshkaLoss(
        model, inner_loss, matryoshka_dims=matryoshka_dims
    )


def print_summary(
    base_results, final_results, matryoshka_dims,
    base_label="Base", final_label="Finetuned",
):
    """Print NDCG@10 and full metrics comparison tables."""
    delta_sym = "\u0394"
    print(f"\n{'='*58}")
    print("Matryoshka Dimension Comparison (NDCG@10)")
    print(f"{'='*58}")
    print(
        f"{'Dim':>6}  {base_label:>14}  "
        f"{final_label:>14}  {delta_sym:>8}"
    )
    print(f"{'-'*6}  {'-'*14}  {'-'*14}  {'-'*8}")
    for dim in matryoshka_dims:
        key = f"eu-ai-act-nl-dim{dim}_cosine_ndcg@10"
        base = base_results[key]
        final = final_results[key]
        delta = final - base
        print(
            f"{dim:>6}  {base:>14.4f}  "
            f"{final:>14.4f}  {delta:>+8.4f}"
        )
    print(f"{'='*58}")

    full_dim = matryoshka_dims[0]
    prefix = f"eu-ai-act-nl-dim{full_dim}_cosine_"
    detail_metrics = [
        ("NDCG@10", "ndcg@10"),
        ("MRR@10", "mrr@10"),
        ("MAP@100", "map@100"),
        ("Accuracy@1", "accuracy@1"),
        ("Accuracy@3", "accuracy@3"),
        ("Accuracy@5", "accuracy@5"),
        ("Accuracy@10", "accuracy@10"),
        ("Precision@1", "precision@1"),
        ("Precision@3", "precision@3"),
        ("Precision@5", "precision@5"),
        ("Precision@10", "precision@10"),
        ("Recall@1", "recall@1"),
        ("Recall@3", "recall@3"),
        ("Recall@5", "recall@5"),
        ("Recall@10", "recall@10"),
    ]

    print(f"\n{'='*58}")
    print(f"Full Metrics at dim={full_dim}")
    print(f"{'='*58}")
    print(
        f"{'Metric':<16}  {base_label:>10}  "
        f"{final_label:>10}  {delta_sym:>10}"
    )
    print(f"{'-'*16}  {'-'*10}  {'-'*10}  {'-'*10}")
    for label, suffix in detail_metrics:
        key = prefix + suffix
        base = base_results.get(key, 0)
        final = final_results.get(key, 0)
        delta = final - base
        print(
            f"{label:<16}  {base:>10.4f}  "
            f"{final:>10.4f}  {delta:>+10.4f}"
        )
    print(f"{'='*58}")


if __name__ == "__main__":

    # -------------------------------------------------------------------
    # CONFIG — edit these values directly
    # -------------------------------------------------------------------
    PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent

    MODEL_NAME      = "intfloat/multilingual-e5-large-instruct"
    TRAIN_DIR       = PROJECT_ROOT / "data" / "processed" / "train"
    EVAL_DIR        = PROJECT_ROOT / "data" / "processed" / "eval"
    OUTPUT_DIR      = PROJECT_ROOT / "models" / "stage_1_mnrl"
    INSTRUCT        = True
    INSTRUCT_PREFIX = (
        "Instruct: Given a question about EU AI regulation, "
        "retrieve the most relevant passage\nQuery: "
    )

    BATCH_SIZE      = 128
    GRAD_ACCUM      = 1
    EPOCHS          = 3
    LR              = 5e-6
    WARMUP_STEPS    = 5
    WEIGHT_DECAY    = 0.01
    MATRYOSHKA_DIMS = [1024, 768, 512, 256, 128, 64]

    # -------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------
    device, use_fp16, use_bf16 = detect_device()
    query_prompt, corpus_prompt = build_prompts(INSTRUCT, INSTRUCT_PREFIX)

    # -------------------------------------------------------------------
    # Load model & data
    # -------------------------------------------------------------------
    model = load_model(MODEL_NAME, instruct=INSTRUCT)

    train_dataset = load_from_disk(str(TRAIN_DIR))
    print(f"Train samples: {len(train_dataset)}")

    queries, corpus, relevant_docs = load_eval_data(EVAL_DIR)
    print(f"Eval queries: {len(queries)}, corpus: {len(corpus)}")

    # -------------------------------------------------------------------
    # Build evaluator & evaluate base model
    # -------------------------------------------------------------------
    eval_suite, primary_metric = build_evaluators(
        queries, corpus, relevant_docs, MATRYOSHKA_DIMS,
        query_prompt, corpus_prompt,
    )

    print("\nEvaluating base model at all Matryoshka dims...")
    base_results = eval_suite(model)
    for dim in MATRYOSHKA_DIMS:
        key = f"eu-ai-act-nl-dim{dim}_cosine_ndcg@10"
        print(f"  Base NDCG@10 (dim={dim}): {base_results[key]:.4f}")

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    loss = build_loss(model, MATRYOSHKA_DIMS)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        prompts={
            "anchor": query_prompt,
            "positive": corpus_prompt,
        },
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=primary_metric,
        logging_steps=5,
        logging_first_step=True,
        run_name="e5-large-ai-act-nl-mnrl",
        report_to="tensorboard",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=eval_suite,
    )

    effective_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = len(train_dataset) // effective_batch
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"  Effective batch: {BATCH_SIZE} x {GRAD_ACCUM} = {effective_batch}")
    print(f"  Steps per epoch: ~{steps_per_epoch}")
    print(f"  Total steps:     ~{steps_per_epoch * EPOCHS}")
    trainer.train()

    # -------------------------------------------------------------------
    # Final evaluation & save
    # -------------------------------------------------------------------
    print("\nEvaluating fine-tuned model at all Matryoshka dims...")
    final_results = eval_suite(model)

    final_path = OUTPUT_DIR / "final"
    model.save_pretrained(str(final_path))
    print(f"\nModel saved to: {final_path}")

    print_summary(base_results, final_results, MATRYOSHKA_DIMS)
