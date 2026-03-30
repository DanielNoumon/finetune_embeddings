"""
Mine hard negatives for Qwen3-Embedding-4B Stage 2 training.

Uses the Stage 1 fine-tuned model to encode all training queries
and corpus chunks, then selects the top-K most similar wrong
chunks as hard negatives for each query.

Output: a new HuggingFace Dataset with columns
(anchor, positive, negative_1, negative_2, ..., negative_N)
"""

import numpy as np
import torch
from pathlib import Path
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer


def mine_hard_negatives(
    model,
    train_dataset,
    query_prompt="",
    corpus_prompt="",
    n_negatives=1,
    range_min=0,
    margin=0.0,
    max_score=1.0,
):
    """Mine hard negatives using the given model.

    For each (anchor, positive) pair, finds the top-N most similar
    but incorrect corpus chunks as hard negatives.

    Args:
        range_min: Skip the top-N most similar candidates (too
            confusing, might be false negatives / true positives).
        margin: Negative similarity must be at least this much lower
            than the query-positive similarity.
        max_score: Skip negatives with similarity above this
            threshold (likely true positives).

    Returns:
        Dataset with columns: anchor, positive, negative_1, ..., N
    """
    anchors = train_dataset["anchor"]
    positives = train_dataset["positive"]
    unique_positives = sorted(set(positives))

    gold_indices = []
    for pos in positives:
        gold_indices.append(unique_positives.index(pos))

    print(f"  Encoding {len(anchors)} queries...")
    q_emb = model.encode(
        anchors,
        prompt=query_prompt,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"  Encoding {len(unique_positives)} corpus chunks...")
    c_emb = model.encode(
        unique_positives,
        prompt=corpus_prompt,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("  Computing similarity matrix...")
    sim_matrix = np.dot(q_emb, c_emb.T)

    # Compute positive similarity for each query
    pos_sims = np.array([
        sim_matrix[i, gold_indices[i]] for i in range(len(anchors))
    ])

    print("\n  Filtering params:")
    print(f"    range_min={range_min} (skip top-N most similar)")
    print(f"    margin={margin} (neg sim < pos sim - margin)")
    print(f"    max_score={max_score} (skip neg with sim > this)")
    print("  Positive similarity stats:")
    print(f"    mean={pos_sims.mean():.4f}, ")
    print(f"    min={pos_sims.min():.4f}, ")
    print(f"    max={pos_sims.max():.4f}")

    print(f"\n  Mining top-{n_negatives} hard negatives per query...")
    neg_columns = {
        f"negative_{j+1}": [] for j in range(n_negatives)
    }
    skipped_match = 0
    skipped_margin = 0
    skipped_max_score = 0
    skipped_range_min = 0
    insufficient = 0

    for i in range(len(anchors)):
        sims = sim_matrix[i].copy()
        sims[gold_indices[i]] = -1.0

        ranked_indices = np.argsort(sims)[::-1]

        collected = []
        for rank, neg_idx in enumerate(ranked_indices):
            if len(collected) >= n_negatives:
                break

            neg_sim = sims[neg_idx]

            if rank < range_min:
                skipped_range_min += 1
                continue

            if neg_sim > max_score:
                skipped_max_score += 1
                continue

            if margin > 0 and neg_sim > (pos_sims[i] - margin):
                skipped_margin += 1
                continue

            neg_text = unique_positives[neg_idx]
            if neg_text == positives[i]:
                skipped_match += 1
                continue

            collected.append(neg_text)

        if len(collected) < n_negatives:
            insufficient += 1
        while len(collected) < n_negatives:
            collected.append("")

        for j, neg_text in enumerate(collected):
            neg_columns[f"negative_{j+1}"].append(neg_text)

    total_neg = len(anchors) * n_negatives
    valid_neg = total_neg - sum(
        1 for texts in neg_columns.values() for t in texts if t == ""
    )
    print(f"  Mined {valid_neg}/{total_neg} valid hard negatives")
    print(f"    ({n_negatives} per query, {len(anchors)} queries)")
    if skipped_range_min > 0:
        print(f"    Skipped {skipped_range_min} (range_min)")
    if skipped_max_score > 0:
        print(f"    Skipped {skipped_max_score} (max_score)")
    if skipped_margin > 0:
        print(f"    Skipped {skipped_margin} (margin)")
    if skipped_match > 0:
        print(f"    Skipped {skipped_match} (matched positive)")
    if insufficient > 0:
        print(
            f"    Warning: {insufficient} queries had insufficient "
            f"negatives after filtering"
        )

    data = {
        "anchor": anchors,
        "positive": positives,
        **neg_columns,
    }
    return Dataset.from_dict(data)


if __name__ == "__main__":

    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    MODEL_NAME = str(
        PROJECT_ROOT / "models" / "qwen3_4b_stage1" / "final"
    )
    TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
    OUTPUT_DIR = (
        PROJECT_ROOT / "data" / "processed" / "qwen3_4b_train_hard_neg"
    )
    N_NEGATIVES = 5
    RANGE_MIN = 5
    MARGIN = 0.1
    MAX_SCORE = 0.9

    QUERY_PROMPT = (
        "Instruct: Given a question about Dutch data protection "
        "and AI regulation, retrieve the most relevant passage\nQuery:"
    )
    CORPUS_PROMPT = ""

    # -------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    print(f"\nLoading Stage 1 model: {MODEL_NAME}")
    model = None
    if device == "cuda":
        for attn in ["flash_attention_2", "sdpa", "eager"]:
            try:
                print(f"  Trying {attn}...")
                model = SentenceTransformer(
                    MODEL_NAME,
                    model_kwargs={
                        "torch_dtype": torch.bfloat16,
                        "attn_implementation": attn,
                    },
                    tokenizer_kwargs={"padding_side": "left"},
                )
                print(f"  Loaded with {attn} + bf16")
                break
            except (ImportError, ValueError) as e:
                print(f"  {attn} failed: {e}")
                continue
    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    train_dataset = load_from_disk(str(TRAIN_DIR))
    print(f"Loading train dataset from: {TRAIN_DIR}")
    print(f"  Train samples: {len(train_dataset)}")

    # -------------------------------------------------------------------
    # Mine
    # -------------------------------------------------------------------
    print(
        f"\nMining hard negatives "
        f"(top-{N_NEGATIVES} per query)..."
    )
    hard_neg_dataset = mine_hard_negatives(
        model,
        train_dataset,
        query_prompt=QUERY_PROMPT,
        corpus_prompt=CORPUS_PROMPT,
        n_negatives=N_NEGATIVES,
        range_min=RANGE_MIN,
        margin=MARGIN,
        max_score=MAX_SCORE,
    )

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    hard_neg_dataset.save_to_disk(str(OUTPUT_DIR))
    print(
        f"\nDataset with hard negatives saved to: {OUTPUT_DIR}"
    )
    print(f"  Columns: {hard_neg_dataset.column_names}")
    print(f"  Rows: {len(hard_neg_dataset)}")

    print(f"\n{'='*60}")
    print("Sample hard negatives (first 3):")
    print(f"{'='*60}")
    for i in range(min(3, len(hard_neg_dataset))):
        row = hard_neg_dataset[i]
        print(f"\n--- Sample {i+1} ---")
        print(f"  Query:    {row['anchor'][:100]}...")
        print(f"  Positive: {row['positive'][:100]}...")
        for j in range(1, N_NEGATIVES + 1):
            col = f"negative_{j}"
            if col in row:
                print(f"  Neg {j}:    {row[col][:80]}...")
    print(f"{'='*60}")
