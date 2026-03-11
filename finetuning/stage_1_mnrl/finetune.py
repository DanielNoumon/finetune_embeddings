"""
Fine-tune multilingual-e5-large on EU AI Act query-chunk pairs
using Matryoshka + MNRL.

Matryoshka Representation Learning wraps MNRL so the model learns
useful embeddings at multiple dimensionalities (1024, 768, 512, 256,
128, 64). At inference time, you can truncate embeddings to any of
these sizes for a speed/quality tradeoff — no retraining needed.

Handles multilingual-e5-large prefix requirements:
  - queries  → "query: " prefix
  - passages → "passage: " prefix

Designed for Google Colab (T4 GPU, 16GB VRAM).
Also works on CPU (slower) — auto-detects device.
"""

import json
import torch
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers import (
    SentenceTransformer,
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


if __name__ == "__main__":
    import argparse

    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # Base model to fine-tune.
    # multilingual-e5-large: 560M params, supports 100+ languages
    # including Dutch. Stronger than e5-base on MTEB multilingual
    # benchmarks. Fits on Colab T4 (16GB) with batch 64 + fp16.
    # Requires "query: " and "passage: " prefixes.
    MODEL_NAME = "intfloat/multilingual-e5-large"

    # Input directories (output of prepare_dataset.py)
    DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
    DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "processed" / "eval"

    # Output directory for the fine-tuned model
    DEFAULT_OUTPUT_DIR = (
        PROJECT_ROOT / "models" / "multilingual-e5-large-ai-act-nl"
    )

    # Training hyperparameters
    #
    # BATCH_SIZE: Actual per-device batch size.
    # With MNRL, each sample gets (batch_size - 1) in-batch negatives.
    # 64 fits on Colab T4 (16GB) with e5-large + fp16.
    # For CPU/low-memory: reduce to 16 and set GRAD_ACCUM_STEPS=4.
    BATCH_SIZE = 64

    # GRAD_ACCUM_STEPS: Simulate larger effective batch by accumulating
    # gradients over multiple steps before updating weights.
    # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
    # On Colab T4 with batch 64: set to 1 (no accumulation needed).
    # On CPU with batch 16: set to 4 → effective batch = 64.
    GRAD_ACCUM_STEPS = 1

    # NUM_EPOCHS: Number of passes through the training data.
    # Small dataset (1,944 pairs) → keep low to avoid overfitting.
    NUM_EPOCHS = 3

    # LEARNING_RATE: Peak learning rate after warmup.
    # 2e-5 is standard for fine-tuning transformers.
    LEARNING_RATE = 2e-5

    # WARMUP_RATIO: Fraction of total steps for linear LR warmup.
    # 0.1 = gradual ramp-up over first 10% of training.
    WARMUP_RATIO = 0.1

    # WEIGHT_DECAY: L2 regularization to prevent overfitting.
    WEIGHT_DECAY = 0.01

    # MATRYOSHKA_DIMS: Dimensionalities to train Matryoshka embeddings
    # for. The model learns useful embeddings at each truncation point.
    # 1024 = full e5-large dimensionality (highest quality).
    # 64   = smallest (fastest retrieval, ~16x less storage than 1024).
    # At inference: model.encode(..., output_dimensionality=256)
    MATRYOSHKA_DIMS = [1024, 768, 512, 256, 128, 64]

    # -------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Fine-tune embedding model with MNRL"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        help="Base model name or path"
    )
    parser.add_argument(
        "--train-dir", type=str, default=str(DEFAULT_TRAIN_DIR),
        help="Train dataset directory"
    )
    parser.add_argument(
        "--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR),
        help="Eval data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for fine-tuned model"
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
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help="Learning rate (default: 2e-5)"
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Device detection
    # -------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    print(f"Device: {device}")
    if device == "cpu":
        print("  Training on CPU — this will be slow (~2-4 hours).")
        print("  Consider using a cloud GPU for faster training.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    effective_batch = args.batch_size * args.grad_accum
    print(f"Effective batch size: {args.batch_size} x {args.grad_accum} "
          f"= {effective_batch}")

    # -------------------------------------------------------------------
    # 1. Load model
    # -------------------------------------------------------------------
    print(f"\nLoading model: {args.model}")
    model = SentenceTransformer(args.model)

    # -------------------------------------------------------------------
    # 2. Load training data
    # -------------------------------------------------------------------
    train_dir = Path(args.train_dir)
    print(f"Loading train dataset from: {train_dir}")
    train_dataset = load_from_disk(str(train_dir))
    print(f"  Train samples: {len(train_dataset)}")

    # -------------------------------------------------------------------
    # 3. Load eval data & build evaluator
    # -------------------------------------------------------------------
    eval_dir = Path(args.eval_dir)
    print(f"Loading eval data from: {eval_dir}")
    queries, corpus, relevant_docs = load_eval_data(eval_dir)
    print(f"  Eval queries: {len(queries)}")
    print(f"  Eval corpus:  {len(corpus)}")

    # Build one IR evaluator per Matryoshka dimension.
    # During training, each evaluator measures retrieval quality at
    # that truncated dimensionality, so we can track quality/size
    # tradeoffs across epochs.
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

    # Evaluate base model before training
    print("\nEvaluating base model at all Matryoshka dims...")
    base_results = eval_suite(model)
    for dim in MATRYOSHKA_DIMS:
        key = f"eu-ai-act-nl-dim{dim}_cosine_ndcg@10"
        print(f"  Base NDCG@10 (dim={dim}): {base_results[key]:.4f}")

    # -------------------------------------------------------------------
    # 4. Define loss — Matryoshka wrapping MNRL
    # -------------------------------------------------------------------
    # MatryoshkaLoss trains the inner MNRL loss at each dimension.
    # The model learns that the first N dims of the embedding are
    # independently useful for retrieval at each truncation point.
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
        # multilingual-e5-large requires specific prefixes per column
        prompts={
            "anchor": "query: ",
            "positive": "passage: ",
        },
        # MNRL benefits from no duplicate samples in a batch
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # Evaluation and saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=primary_metric,
        # Logging — loss is logged every N steps.
        # With 1,944 train samples / batch 64 = ~30 steps/epoch.
        # logging_steps=5 gives ~6 log points per epoch (good granularity).
        logging_steps=5,
        logging_first_step=True,
        run_name="e5-large-ai-act-nl-mnrl",
        # TensorBoard logging for loss curves.
        # In Colab: %load_ext tensorboard; %tensorboard --logdir runs/
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

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Steps per epoch: ~{len(train_dataset) // effective_batch}")
    print(f"  Total steps:     ~{len(train_dataset) // effective_batch * args.epochs}")
    trainer.train()

    # -------------------------------------------------------------------
    # 7. Final evaluation at all Matryoshka dimensions
    # -------------------------------------------------------------------
    print("\nEvaluating fine-tuned model at all Matryoshka dims...")
    final_results = eval_suite(model)

    # -------------------------------------------------------------------
    # 8. Save
    # -------------------------------------------------------------------
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    print(f"\nModel saved to: {final_path}")

    # Print summary — show quality at every dimension
    print(f"\n{'='*58}")
    print(f"Training Complete — Matryoshka Evaluation")
    print(f"{'='*58}")
    print(f"{'Dim':>6}  {'Base NDCG@10':>14}  {'Finetuned':>14}  {'Δ':>8}")
    print(f"{'-'*6}  {'-'*14}  {'-'*14}  {'-'*8}")
    for dim in MATRYOSHKA_DIMS:
        key = f"eu-ai-act-nl-dim{dim}_cosine_ndcg@10"
        base = base_results[key]
        final = final_results[key]
        delta = final - base
        print(f"{dim:>6}  {base:>14.4f}  {final:>14.4f}  {delta:>+8.4f}")
    print(f"{'='*58}")
