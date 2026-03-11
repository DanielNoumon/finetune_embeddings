"""
Stage 2: Fine-tune with hard negatives + Matryoshka + MNRL.

Starts from the Stage 1 checkpoint (already fine-tuned with in-batch
negatives only). The training dataset now includes an explicit 'negative'
column — the hardest wrong chunk for each query, mined by the Stage 1
model itself.

MNRL with a 'negative' column uses BOTH:
  - The explicit hard negative (from mining)
  - In-batch negatives (every other passage in the batch)
This gives the model a much harder training signal than Stage 1.

Key differences from Stage 1:
  - Model: loads Stage 1 checkpoint (not base model)
  - Dataset: (anchor, positive, negative) triplets
  - Learning rate: lower (1e-5) — already fine-tuned, smaller updates
  - Epochs: 2 — stronger signal needs fewer passes to avoid overfitting
  - Prompts: includes "negative" column prefix
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
    with open(
        eval_dir / "relevant_docs.json", "r", encoding="utf-8"
    ) as f:
        relevant_docs_raw = json.load(f)

    # Convert lists back to sets (JSON doesn't support sets)
    relevant_docs = {
        qid: set(cids) for qid, cids in relevant_docs_raw.items()
    }

    return queries, corpus, relevant_docs


if __name__ == "__main__":
    import argparse

    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # Stage 1 model — starting point for Stage 2 fine-tuning.
    # We continue training from the already-adapted checkpoint,
    # not from the base model.
    DEFAULT_MODEL_DIR = str(
        PROJECT_ROOT / "models" / "stage_1_mnrl" / "final"
    )

    # Training data with hard negatives (output of mine_negatives.py)
    DEFAULT_TRAIN_DIR = (
        PROJECT_ROOT / "data" / "processed" / "train_hard_neg"
    )

    # Same eval data as Stage 1 — enables direct comparison
    DEFAULT_EVAL_DIR = (
        PROJECT_ROOT / "data" / "processed" / "eval"
    )

    # Output directory for Stage 2 model
    DEFAULT_OUTPUT_DIR = (
        PROJECT_ROOT / "models" / "stage_2_hard_neg"
    )

    # Training hyperparameters — tuned for Stage 2
    #
    # BATCH_SIZE: Reduced from Stage 1's 64 to 32 because MNRL now
    # processes 3 columns (anchor, positive, negative) instead of 2.
    # With Matryoshka (6 dims), that's 18 forward passes vs 12 → OOM
    # at batch 64. Each sample gets 31 in-batch negatives + 1 hard
    # negative — the hard negative is far more informative than 32
    # extra random in-batch negatives would be.
    BATCH_SIZE = 32

    # Gradient accumulation 2 → effective batch 64 for weight updates.
    # NOTE: only per-device batch (32) determines in-batch negatives.
    GRAD_ACCUM_STEPS = 2

    # NUM_EPOCHS: Fewer than Stage 1. Hard negatives provide a
    # stronger training signal — the model learns faster per step
    # but is also more prone to overfitting.
    NUM_EPOCHS = 2

    # LEARNING_RATE: Lower than Stage 1 (2e-5). The model is
    # already fine-tuned — large updates risk catastrophic forgetting.
    # 1e-5 gives stable refinement without undoing Stage 1 gains.
    LEARNING_RATE = 1e-5

    # Same regularization as Stage 1
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01

    # Same Matryoshka dimensions as Stage 1
    MATRYOSHKA_DIMS = [1024, 768, 512, 256, 128, 64]

    # -------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Stage 2: Fine-tune with hard negatives + MNRL"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_DIR,
        help="Stage 1 model path (default: models/stage_1_mnrl/final)"
    )
    parser.add_argument(
        "--train-dir", type=str, default=str(DEFAULT_TRAIN_DIR),
        help="Training data with hard negatives"
    )
    parser.add_argument(
        "--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR),
        help="Eval data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for Stage 2 model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Per-device batch size (default: 64)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=GRAD_ACCUM_STEPS,
        help="Gradient accumulation steps (default: 1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help="Number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help="Learning rate (default: 1e-5)"
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Device detection
    # -------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    print(f"Device: {device}")
    if device == "cpu":
        print("  Training on CPU — this will be slow.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    effective_batch = args.batch_size * args.grad_accum
    print(
        f"Effective batch size: {args.batch_size} x "
        f"{args.grad_accum} = {effective_batch}"
    )

    # -------------------------------------------------------------------
    # 1. Load Stage 1 model
    # -------------------------------------------------------------------
    print(f"\nLoading Stage 1 model: {args.model}")
    model = SentenceTransformer(
        args.model,
        model_kwargs={"attn_implementation": "sdpa"},
        model_card_data=SentenceTransformerModelCardData(
            language="nl",
            license="apache-2.0",
            model_name=(
                "multilingual-e5-large EU AI Act NL "
                "Matryoshka Hard Negatives"
            ),
        ),
    )

    # -------------------------------------------------------------------
    # 2. Load training data (with hard negatives)
    # -------------------------------------------------------------------
    train_dir = Path(args.train_dir)
    print(f"Loading train dataset from: {train_dir}")
    train_dataset = load_from_disk(str(train_dir))
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Columns: {train_dataset.column_names}")

    # Verify the dataset has the expected columns
    expected_cols = {"anchor", "positive", "negative"}
    actual_cols = set(train_dataset.column_names)
    if not expected_cols.issubset(actual_cols):
        missing = expected_cols - actual_cols
        raise ValueError(
            f"Dataset missing columns: {missing}. "
            f"Run mine_negatives.py first."
        )

    # -------------------------------------------------------------------
    # 3. Load eval data & build evaluator
    # -------------------------------------------------------------------
    eval_dir = Path(args.eval_dir)
    print(f"Loading eval data from: {eval_dir}")
    queries, corpus, relevant_docs = load_eval_data(eval_dir)
    print(f"  Eval queries: {len(queries)}")
    print(f"  Eval corpus:  {len(corpus)}")

    # Build one IR evaluator per Matryoshka dimension
    evaluators = []
    for dim in MATRYOSHKA_DIMS:
        evaluators.append(
            InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=f"eu-ai-act-nl-dim{dim}",
                truncate_dim=dim,
                query_prompt="query: ",
                corpus_prompt="passage: ",
                show_progress_bar=True,
            )
        )
    eval_suite = SequentialEvaluator(evaluators)

    # Primary metric for best-model selection uses full 1024-dim
    primary_metric = "eu-ai-act-nl-dim1024_cosine_ndcg@10"

    # Evaluate Stage 1 model before Stage 2 training
    print("\nEvaluating Stage 1 model at all Matryoshka dims...")
    base_results = eval_suite(model)
    for dim in MATRYOSHKA_DIMS:
        key = f"eu-ai-act-nl-dim{dim}_cosine_ndcg@10"
        print(
            f"  Stage 1 NDCG@10 (dim={dim}): "
            f"{base_results[key]:.4f}"
        )

    # -------------------------------------------------------------------
    # 4. Define loss — Matryoshka wrapping MNRL
    # -------------------------------------------------------------------
    # Same loss structure as Stage 1. MNRL automatically detects
    # the 'negative' column and uses it alongside in-batch negatives.
    inner_loss = MultipleNegativesRankingLoss(model)
    loss = MatryoshkaLoss(
        model, inner_loss, matryoshka_dims=MATRYOSHKA_DIMS
    )

    # -------------------------------------------------------------------
    # 5. Training arguments
    # -------------------------------------------------------------------
    output_dir = Path(args.output_dir)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=use_fp16,
        gradient_checkpointing=True,
        # Prefixes for all three columns
        prompts={
            "anchor": "query: ",
            "positive": "passage: ",
            "negative": "passage: ",
        },
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # Evaluation and saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=primary_metric,
        # Logging
        logging_steps=5,
        logging_first_step=True,
        run_name="e5-large-ai-act-nl-hard-neg",
        report_to="tensorboard",
    )

    # -------------------------------------------------------------------
    # 6. Train
    # -------------------------------------------------------------------
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=eval_suite,
    )

    print(f"\nStarting Stage 2 training for {args.epochs} epochs...")
    steps_per_epoch = len(train_dataset) // effective_batch
    print(f"  Steps per epoch: ~{steps_per_epoch}")
    print(f"  Total steps:     ~{steps_per_epoch * args.epochs}")
    trainer.train()

    # -------------------------------------------------------------------
    # 7. Final evaluation at all Matryoshka dimensions
    # -------------------------------------------------------------------
    print("\nEvaluating Stage 2 model at all Matryoshka dims...")
    final_results = eval_suite(model)

    # -------------------------------------------------------------------
    # 8. Save
    # -------------------------------------------------------------------
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    print(f"\nModel saved to: {final_path}")

    # Print summary — NDCG@10 across all Matryoshka dimensions
    print(f"\n{'='*58}")
    print("Matryoshka Dimension Comparison (NDCG@10)")
    print(f"{'='*58}")
    print(
        f"{'Dim':>6}  {'Stage 1':>14}  {'Stage 2':>14}  "
        f"{'Δ':>8}"
    )
    print(f"{'-'*6}  {'-'*14}  {'-'*14}  {'-'*8}")
    for dim in MATRYOSHKA_DIMS:
        key = f"eu-ai-act-nl-dim{dim}_cosine_ndcg@10"
        s1 = base_results[key]
        s2 = final_results[key]
        delta = s2 - s1
        print(
            f"{dim:>6}  {s1:>14.4f}  {s2:>14.4f}  "
            f"{delta:>+8.4f}"
        )
    print(f"{'='*58}")

    # Detailed metrics at full dimensionality (1024)
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
        f"{'Metric':<16}  {'Stage 1':>10}  {'Stage 2':>10}  "
        f"{'Δ':>10}"
    )
    print(f"{'-'*16}  {'-'*10}  {'-'*10}  {'-'*10}")
    for label, suffix in detail_metrics:
        key = prefix + suffix
        s1 = base_results.get(key, 0)
        s2 = final_results.get(key, 0)
        delta = s2 - s1
        print(
            f"{label:<16}  {s1:>10.4f}  {s2:>10.4f}  "
            f"{delta:>+10.4f}"
        )
    print(f"{'='*58}")
