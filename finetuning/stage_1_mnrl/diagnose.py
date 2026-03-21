"""Quick diagnostic to isolate the cause of embedding collapse.

Tests:
1. Plain MNRL (no Matryoshka) — isolates if Matryoshka causes collapse
2. Checks embeddings after every training step
3. Prints loss values
"""
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import (
    SentenceTransformerTrainingArguments,
    BatchSamplers,
)
from datasets import load_from_disk
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"

TEST_TEXTS = [
    "What is the EU AI Act?",
    "AI systems are classified by risk.",
    "The weather is nice today.",
]

INSTRUCT_PREFIX = (
    "Instruct: Given a question about EU AI regulation, "
    "retrieve the most relevant passage\nQuery: "
)


def check_embeddings(model, label):
    """Encode test texts and print norms + cosine similarities."""
    emb = model.encode(TEST_TEXTS)
    norms = np.linalg.norm(emb, axis=1)
    cos01 = np.dot(emb[0], emb[1]) / (norms[0] * norms[1])
    cos02 = np.dot(emb[0], emb[2]) / (norms[0] * norms[2])
    print(f"  [{label}] cos(q,relevant)={cos01:.4f}  "
          f"cos(q,irrelevant)={cos02:.4f}  "
          f"norms={norms.round(4)}")
    return cos01, cos02


def run_test(test_name, attn_impl, use_bf16, max_steps=10):
    """Run a short training test and check for collapse."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"  attn={attn_impl}, bf16={use_bf16}")
    print(f"{'='*60}")

    model = SentenceTransformer(
        "intfloat/multilingual-e5-large-instruct",
        model_kwargs={"attn_implementation": attn_impl},
    )

    check_embeddings(model, "before training")

    train_dataset = load_from_disk(str(TRAIN_DIR))
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "diagnostic"),
        max_steps=max_steps,
        per_device_train_batch_size=16,
        learning_rate=5e-6,
        warmup_steps=2,
        bf16=use_bf16,
        fp16=False,
        gradient_checkpointing=False,
        prompts={"anchor": INSTRUCT_PREFIX, "positive": ""},
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args,
        train_dataset=train_dataset, loss=loss,
    )
    trainer.train()

    check_embeddings(model, "after training")
    print()


if __name__ == "__main__":
    # Test 1: SDPA + bf16 (current failing config)
    run_test("SDPA + bf16", attn_impl="sdpa", use_bf16=True)

    # Test 2: Eager attention + bf16 (isolates SDPA issue)
    run_test("Eager + bf16", attn_impl="eager", use_bf16=True)

    # Test 3: SDPA + fp32 (isolates bf16 issue)
    run_test("SDPA + fp32", attn_impl="sdpa", use_bf16=False)

    # Test 4: Eager + fp32 (safest config)
    run_test("Eager + fp32", attn_impl="eager", use_bf16=False)
