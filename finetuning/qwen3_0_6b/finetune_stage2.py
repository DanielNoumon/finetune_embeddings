"""
Stage 2 fine-tuning of Qwen3-Embedding-0.6B with hard negatives.

Continues from the Stage 1 checkpoint with mined hard negatives.
Uses the same Matryoshka + MNRL setup but with lower learning rate
to avoid catastrophic forgetting.

Run mine_negatives.py first to generate the hard negative dataset.
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
    CachedMultipleNegativesRankingLoss,
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


def detect_device():
    """Detect compute device and precision support."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = False
    use_bf16 = False
    print(f"Device: {device}")
    if device == "cpu":
        print("  CPU mode — this will be slow.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("  Precision: bf16")
        else:
            use_fp16 = True
            print("  Precision: fp16")
    return device, use_fp16, use_bf16


def load_model(
    model_name,
    card_name="Qwen3-Embedding-0.6B EU AI Act NL Hard Neg Matryoshka",
):
    """Load Qwen3-Embedding with bf16, trying attention backends."""
    print(f"\nLoading model: {model_name}")
    card = SentenceTransformerModelCardData(
        language="nl",
        license="apache-2.0",
        model_name=card_name,
    )
    if not torch.cuda.is_available():
        return SentenceTransformer(
            model_name, model_card_data=card,
        )

    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            print(f"  Trying {attn}...")
            model = SentenceTransformer(
                model_name,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": attn,
                },
                tokenizer_kwargs={"padding_side": "left"},
                model_card_data=card,
            )
            print(f"  Loaded with {attn} + bf16")
            return model
        except (ImportError, ValueError) as e:
            print(f"  {attn} failed: {e}")
            continue

    raise RuntimeError("All attention backends failed.")


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
                name=f"dutch-regs-dim{dim}",
                truncate_dim=dim,
                query_prompt=query_prompt,
                corpus_prompt=corpus_prompt,
                show_progress_bar=True,
            )
        )
    eval_suite = SequentialEvaluator(evaluators)
    primary_metric = (
        f"dutch-regs-dim{matryoshka_dims[0]}_cosine_ndcg@10"
    )
    return eval_suite, primary_metric


def build_loss(model, matryoshka_dims=None, mini_batch_size=None):
    """Build MNRL loss, optionally cached and/or Matryoshka."""
    if mini_batch_size:
        inner = CachedMultipleNegativesRankingLoss(
            model, mini_batch_size=mini_batch_size,
        )
    else:
        inner = MultipleNegativesRankingLoss(model)
    if matryoshka_dims:
        return MatryoshkaLoss(
            model, inner, matryoshka_dims=matryoshka_dims
        )
    return inner


def print_summary(
    base_results, final_results, matryoshka_dims,
    base_label="Stage 1", final_label="Stage 2",
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
        key = f"dutch-regs-dim{dim}_cosine_ndcg@10"
        base = base_results[key]
        final = final_results[key]
        delta = final - base
        print(
            f"{dim:>6}  {base:>14.4f}  "
            f"{final:>14.4f}  {delta:>+8.4f}"
        )
    print(f"{'='*58}")

    full_dim = matryoshka_dims[0]
    prefix = f"dutch-regs-dim{full_dim}_cosine_"
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
    # CONFIG
    # -------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    MODEL_NAME = str(
        PROJECT_ROOT / "models" / "qwen3_0_6b_stage1" / "final"
    )
    TRAIN_DIR = (
        PROJECT_ROOT / "data" / "processed" / "qwen3_0_6b_train_hard_neg"
    )
    EVAL_DIR = PROJECT_ROOT / "data" / "processed" / "eval"
    OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen3_0_6b_stage2"

    # Qwen3 instruct prompt
    QUERY_PROMPT = (
        "Instruct: Given a question about Dutch data protection "
        "and AI regulation, retrieve the most relevant passage\nQuery:"
    )
    CORPUS_PROMPT = ""

    # Stage 2: lower LR, fewer epochs, CachedMNRL for VRAM
    BATCH_SIZE = 128
    MINI_BATCH_SIZE = 4
    GRAD_ACCUM = 1
    EVAL_BATCH_SIZE = 4
    EPOCHS = 2
    LR = 1e-5
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    WEIGHT_DECAY = 0.01
    USE_MATRYOSHKA = True
    MATRYOSHKA_DIMS = [1024, 768, 512, 256, 128, 64]

    # -------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------
    device, use_fp16, use_bf16 = detect_device()

    # -------------------------------------------------------------------
    # Load model & data
    # -------------------------------------------------------------------
    model = load_model(MODEL_NAME)

    train_dataset = load_from_disk(str(TRAIN_DIR))
    print(f"Train samples: {len(train_dataset)}")
    print(f"Columns: {train_dataset.column_names}")

    # Verify the dataset has the expected columns
    actual_cols = set(train_dataset.column_names)
    if "anchor" not in actual_cols or "positive" not in actual_cols:
        raise ValueError(
            "Dataset missing anchor/positive columns. "
            "Run mine_negatives.py first."
        )
    neg_cols = sorted(
        [c for c in actual_cols if c.startswith("negative")]
    )
    if not neg_cols:
        raise ValueError(
            "Dataset has no negative columns. "
            "Run mine_negatives.py first."
        )
    print(f"  Hard negative columns: {neg_cols}")

    queries, corpus, relevant_docs = load_eval_data(EVAL_DIR)
    print(f"Eval queries: {len(queries)}, corpus: {len(corpus)}")

    # -------------------------------------------------------------------
    # Build evaluator & evaluate Stage 1 model
    # -------------------------------------------------------------------
    eval_suite, primary_metric = build_evaluators(
        queries, corpus, relevant_docs, MATRYOSHKA_DIMS,
        QUERY_PROMPT, CORPUS_PROMPT,
    )

    print("\nEvaluating Stage 1 model at all Matryoshka dims...")
    base_results = eval_suite(model)
    for dim in MATRYOSHKA_DIMS:
        key = f"dutch-regs-dim{dim}_cosine_ndcg@10"
        val = base_results[key]
        print(f"  Stage 1 NDCG@10 (dim={dim}): {val:.4f}")

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    loss = build_loss(
        model,
        matryoshka_dims=MATRYOSHKA_DIMS if USE_MATRYOSHKA else None,
        mini_batch_size=MINI_BATCH_SIZE,
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=WEIGHT_DECAY,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=False,
        prompts={
            "anchor": QUERY_PROMPT,
            "positive": CORPUS_PROMPT,
            **{col: CORPUS_PROMPT for col in neg_cols},
        },
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=primary_metric,
        logging_steps=5,
        logging_first_step=True,
        run_name="qwen3-0.6b-dutch-regulations-hard-neg",
        report_to="wandb",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=eval_suite,
    )

    effective_batch = BATCH_SIZE * GRAD_ACCUM
    in_batch_neg = BATCH_SIZE - 1
    steps_per_epoch = len(train_dataset) // effective_batch
    print(f"\nStarting Stage 2 training for {EPOCHS} epochs...")
    print(f"  Batch size: {BATCH_SIZE} ({in_batch_neg} in-batch neg)")
    print(f"  Hard negatives: {len(neg_cols)} per query")
    if MINI_BATCH_SIZE:
        print(f"  Mini-batch (VRAM): {MINI_BATCH_SIZE}")
    print(f"  Grad accum: {GRAD_ACCUM}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Steps per epoch: ~{steps_per_epoch}")
    print(f"  Total steps: ~{steps_per_epoch * EPOCHS}")
    trainer.train()

    # -------------------------------------------------------------------
    # Final evaluation & save
    # -------------------------------------------------------------------
    print("\nEvaluating Stage 2 model at all Matryoshka dims...")
    final_results = eval_suite(model)

    final_path = OUTPUT_DIR / "final"
    model.save_pretrained(str(final_path))
    print(f"\nModel saved to: {final_path}")

    print_summary(base_results, final_results, MATRYOSHKA_DIMS)
