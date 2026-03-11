"""
Mine hard negatives using the Stage 1 fine-tuned model.

For each training query, encode all training corpus chunks with the
Stage 1 model, rank by cosine similarity, and pick the top-K most
similar chunks that are NOT the correct positive. These "hard negatives"
are passages the model finds similar but are actually wrong — the
hardest cases for the model to distinguish.

Output: a new HuggingFace Dataset with columns (anchor, positive, negative)
that can be used for Stage 2 MNRL training.

Why mine from the Stage 1 model (not the base model)?
- The Stage 1 model has already learned task-specific similarity.
- Its mistakes are more informative: hard negatives from an adapted
  model are genuinely confusing cases, not random noise.
- Base model negatives would be "easy" for the already-fine-tuned model.
"""

import numpy as np
import torch
from pathlib import Path
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer


def mine_hard_negatives(
    model: SentenceTransformer,
    train_dataset: Dataset,
    n_negatives: int = 1,
    batch_size: int = 64,
) -> Dataset:
    """
    Mine hard negatives for each query in the training dataset.

    For each (anchor, positive) pair:
    1. Encode the anchor with "query: " prefix
    2. Find the most similar positives in the corpus (by cosine sim)
    3. Exclude the correct positive
    4. Take the top-N most similar as hard negatives

    Args:
        model: Fine-tuned SentenceTransformer (Stage 1).
        train_dataset: Dataset with 'anchor' and 'positive' columns.
        n_negatives: Number of hard negatives per query (default: 1).
        batch_size: Encoding batch size.

    Returns:
        Dataset with columns: anchor, positive, negative
        (one row per hard negative — if n_negatives=1, same row count).
    """
    anchors = train_dataset["anchor"]
    positives = train_dataset["positive"]

    # Build corpus of unique positives (chunks) with index mapping.
    # Multiple queries may point to the same chunk text.
    unique_positives = list(set(positives))
    positive_to_idx = {text: i for i, text in enumerate(unique_positives)}

    # Map each training sample to its corpus index
    gold_indices = [positive_to_idx[p] for p in positives]

    # Encode queries and corpus
    print(f"  Encoding {len(anchors)} queries...")
    query_embeddings = model.encode(
        anchors,
        prompt="query: ",
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"  Encoding {len(unique_positives)} corpus chunks...")
    corpus_embeddings = model.encode(
        unique_positives,
        prompt="passage: ",
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Compute similarity matrix: (n_queries, n_corpus)
    # Embeddings are already normalized → dot product = cosine similarity
    print("  Computing similarity matrix...")
    sim_matrix = query_embeddings @ corpus_embeddings.T

    # For each query, mask the gold positive and pick top-K
    print(f"  Mining top-{n_negatives} hard negatives per query...")
    new_anchors = []
    new_positives = []
    new_negatives = []
    skipped = 0

    for i in range(len(anchors)):
        sims = sim_matrix[i].copy()
        # Mask the correct positive so it can't be selected as negative
        sims[gold_indices[i]] = -1.0

        # Get top-K indices (highest similarity = hardest negatives)
        top_k_indices = np.argsort(sims)[-n_negatives:][::-1]

        for neg_idx in top_k_indices:
            neg_text = unique_positives[neg_idx]
            # Safety: skip if somehow the negative IS the positive
            if neg_text == positives[i]:
                skipped += 1
                continue
            new_anchors.append(anchors[i])
            new_positives.append(positives[i])
            new_negatives.append(neg_text)

    if skipped > 0:
        print(f"  Warning: skipped {skipped} duplicate negatives")

    print(f"  Mined {len(new_negatives)} hard negative triplets "
          f"from {len(anchors)} queries")

    return Dataset.from_dict({
        "anchor": new_anchors,
        "positive": new_positives,
        "negative": new_negatives,
    })


if __name__ == "__main__":
    import argparse

    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # Stage 1 fine-tuned model — used to mine hard negatives
    DEFAULT_MODEL_DIR = str(
        PROJECT_ROOT / "models" / "stage_1_mnrl" / "final"
    )

    # Input: Stage 1 training data (anchor, positive pairs)
    DEFAULT_TRAIN_DIR = str(
        PROJECT_ROOT / "data" / "processed" / "train"
    )

    # Output: training data with hard negatives
    DEFAULT_OUTPUT_DIR = str(
        PROJECT_ROOT / "data" / "processed" / "train_hard_neg"
    )

    # Number of hard negatives per query.
    # 1 is standard and effective — each query gets its single hardest
    # wrong chunk. More negatives can help but increase dataset size
    # and training time proportionally.
    N_NEGATIVES = 1

    # -------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Mine hard negatives using Stage 1 model"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_DIR,
        help="Path to Stage 1 fine-tuned model"
    )
    parser.add_argument(
        "--train-dir", type=str, default=DEFAULT_TRAIN_DIR,
        help="Stage 1 training dataset directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for dataset with hard negatives"
    )
    parser.add_argument(
        "--n-negatives", type=int, default=N_NEGATIVES,
        help="Hard negatives per query (default: 1)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Encoding batch size (default: 64)"
    )
    args = parser.parse_args()

    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Load Stage 1 model
    print(f"\nLoading Stage 1 model: {args.model}")
    model = SentenceTransformer(
        args.model,
        model_kwargs={"attn_implementation": "sdpa"},
    )

    # Load training data
    print(f"Loading train dataset from: {args.train_dir}")
    train_dataset = load_from_disk(args.train_dir)
    print(f"  Train samples: {len(train_dataset)}")

    # Mine hard negatives
    print(f"\nMining hard negatives (top-{args.n_negatives} per query)...")
    hard_neg_dataset = mine_hard_negatives(
        model=model,
        train_dataset=train_dataset,
        n_negatives=args.n_negatives,
        batch_size=args.batch_size,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hard_neg_dataset.save_to_disk(str(output_dir))
    print(f"\nDataset with hard negatives saved to: {output_dir}")
    print(f"  Columns: {hard_neg_dataset.column_names}")
    print(f"  Rows: {len(hard_neg_dataset)}")

    # Print a few examples for verification
    print(f"\n{'='*60}")
    print("Sample hard negatives (first 3):")
    print(f"{'='*60}")
    for i in range(min(3, len(hard_neg_dataset))):
        row = hard_neg_dataset[i]
        print(f"\n--- Sample {i+1} ---")
        print(f"  Query:    {row['anchor'][:100]}...")
        print(f"  Positive: {row['positive'][:100]}...")
        print(f"  Negative: {row['negative'][:100]}...")
    print(f"{'='*60}")
