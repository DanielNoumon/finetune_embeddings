"""Quick diagnostic to isolate the cause of embedding collapse.

Tests:
1. Plain MNRL (no Matryoshka) — isolates if Matryoshka causes collapse
2. Checks embeddings after every training step
3. Prints loss values
"""
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import (
    MultipleNegativesRankingLoss,
    MatryoshkaLoss,
)
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


def run_test(test_name, loss_fn, use_prompts, max_steps=10):
    """Run a short training test and check for collapse."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    model = SentenceTransformer(
        "intfloat/multilingual-e5-large-instruct",
        model_kwargs={"attn_implementation": "sdpa"},
    )

    check_embeddings(model, "before training")

    train_dataset = load_from_disk(str(TRAIN_DIR))

    prompts = {}
    if use_prompts:
        prompts = {"anchor": INSTRUCT_PREFIX, "positive": ""}

    loss = loss_fn(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "diagnostic"),
        max_steps=max_steps,
        per_device_train_batch_size=64,
        learning_rate=5e-6,
        warmup_steps=2,
        bf16=True,
        gradient_checkpointing=False,
        prompts=prompts,
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
    # Test 1: Plain MNRL, WITH prompts
    run_test(
        "Plain MNRL + instruct prompts",
        loss_fn=lambda m: MultipleNegativesRankingLoss(m),
        use_prompts=True,
        max_steps=10,
    )

    # Test 2: Plain MNRL, WITHOUT prompts
    run_test(
        "Plain MNRL, no prompts",
        loss_fn=lambda m: MultipleNegativesRankingLoss(m),
        use_prompts=False,
        max_steps=10,
    )

    # Test 3: Matryoshka + MNRL, WITH prompts (what we've been running)
    run_test(
        "Matryoshka + MNRL + instruct prompts",
        loss_fn=lambda m: MatryoshkaLoss(
            m, MultipleNegativesRankingLoss(m),
            matryoshka_dims=[1024, 768, 512, 256, 128, 64],
        ),
        use_prompts=True,
        max_steps=10,
    )

    # Test 4: Matryoshka + MNRL, WITHOUT prompts
    run_test(
        "Matryoshka + MNRL, no prompts",
        loss_fn=lambda m: MatryoshkaLoss(
            m, MultipleNegativesRankingLoss(m),
            matryoshka_dims=[1024, 768, 512, 256, 128, 64],
        ),
        use_prompts=False,
        max_steps=10,
    )
