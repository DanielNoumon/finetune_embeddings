"""
Mine hard negatives for Qwen3-Embedding-0.6B Stage 2 training.

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
):
    """Mine hard negatives using the given model.

    For each (anchor, positive) pair, finds the top-N most similar
    but incorrect corpus chunks as hard negatives.

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

    print(f"  Mining top-{n_negatives} hard negatives per query...")
    neg_columns = {
        f"negative_{j+1}": [] for j in range(n_negatives)
    }
    skipped = 0

    for i in range(len(anchors)):
        sims = sim_matrix[i].copy()
        sims[gold_indices[i]] = -1.0

        top_k_indices = np.argsort(sims)[-n_negatives:][::-1]

        for j, neg_idx in enumerate(top_k_indices):
            neg_text = unique_positives[neg_idx]
            if neg_text == positives[i]:
                skipped += 1
                neg_text = ""
            neg_columns[f"negative_{j+1}"].append(neg_text)

    if skipped > 0:
        print(f"  Warning: {skipped} negatives matched positive")

    total_neg = len(anchors) * n_negatives
    print(
        f"  Mined {total_neg} hard negatives "
        f"({n_negatives} per query, {len(anchors)} queries)"
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
        PROJECT_ROOT / "models" / "qwen3_stage1" / "final"
    )
    TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
    OUTPUT_DIR = (
        PROJECT_ROOT / "data" / "processed" / "qwen3_train_hard_neg"
    )
    N_NEGATIVES = 1

    # Qwen3 instruct prompt
    QUERY_PROMPT = (
        "Instruct: Given a question about EU AI regulation, "
        "retrieve the most relevant passage\nQuery:"
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
    model_kwargs = {}
    if device == "cuda":
        model_kwargs["model_kwargs"] = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        }
        model_kwargs["tokenizer_kwargs"] = {
            "padding_side": "left",
        }
    model = SentenceTransformer(MODEL_NAME, **model_kwargs)

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
