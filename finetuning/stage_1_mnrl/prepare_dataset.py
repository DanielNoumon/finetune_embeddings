"""
Prepare train/eval splits from synthetic query-chunk pairs.

Splits at the CHUNK level (not pair level) to prevent data leakage:
multiple queries for the same chunk always stay in the same split.

Train set  → HuggingFace Dataset with 'anchor' and 'positive' columns
             (native Sentence Transformers format for MNRL).
Eval set   → InformationRetrievalEvaluator format:
             queries.json, corpus.json, relevant_docs.json
"""

import json
import random
from pathlib import Path
from datasets import Dataset, load_dataset


def load_pairs(repo_id: str) -> list[dict]:
    """Load query-chunk pairs from HuggingFace Hub.

    Maps HF column names to internal format:
        query -> anchor, chunk -> positive
    """
    ds = load_dataset(repo_id, split="train")
    pairs = []
    for row in ds:
        pairs.append({
            "anchor": row["query"],
            "positive": row["chunk"],
            "chunk_id": row["chunk_id"],
        })
    return pairs


def chunk_level_split(
    pairs: list[dict],
    eval_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split pairs into train/eval by unique chunk_id.

    All queries belonging to the same chunk stay in the same split.
    This prevents the model from seeing eval chunks during training.

    Args:
        pairs: List of query-chunk pair dicts.
        eval_fraction: Fraction of unique chunks reserved for eval.
        seed: Random seed for reproducibility.

    Returns:
        (train_pairs, eval_pairs)
    """
    # Collect unique chunk IDs
    chunk_ids = sorted(set(p["chunk_id"] for p in pairs))

    # Shuffle deterministically and split
    rng = random.Random(seed)
    rng.shuffle(chunk_ids)
    n_eval = max(1, int(len(chunk_ids) * eval_fraction))
    eval_chunk_ids = set(chunk_ids[:n_eval])

    train_pairs = [p for p in pairs if p["chunk_id"] not in eval_chunk_ids]
    eval_pairs = [p for p in pairs if p["chunk_id"] in eval_chunk_ids]

    return train_pairs, eval_pairs


def build_train_dataset(pairs: list[dict]) -> Dataset:
    """
    Build a HuggingFace Dataset for Sentence Transformers training.

    Columns:
        anchor   — the query text
        positive — the relevant chunk text
    """
    return Dataset.from_dict({
        "anchor": [p["anchor"] for p in pairs],
        "positive": [p["positive"] for p in pairs],
    })


def build_eval_dicts(
    pairs: list[dict],
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """
    Build InformationRetrievalEvaluator inputs from eval pairs.

    Returns:
        queries:       {query_id: query_text}
        corpus:        {chunk_id: chunk_text}
        relevant_docs: {query_id: {chunk_id}}
    """
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}

    for i, p in enumerate(pairs):
        qid = f"q_{i}"
        cid = str(p["chunk_id"])

        queries[qid] = p["anchor"]
        corpus[cid] = p["positive"]
        relevant_docs[qid] = {cid}

    return queries, corpus, relevant_docs


def save_eval_dicts(
    queries: dict,
    corpus: dict,
    relevant_docs: dict,
    output_dir: Path,
) -> None:
    """Save eval dicts as JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # relevant_docs contains sets — convert to lists for JSON
    relevant_docs_serializable = {
        qid: list(cids) for qid, cids in relevant_docs.items()
    }

    for name, data in [
        ("queries.json", queries),
        ("corpus.json", corpus),
        ("relevant_docs.json", relevant_docs_serializable),
    ]:
        path = output_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {path.name}: {len(data)} entries")


def print_split_stats(
    train_pairs: list[dict],
    eval_pairs: list[dict],
) -> None:
    """Print summary statistics for train/eval split."""
    train_chunks = set(p["chunk_id"] for p in train_pairs)
    eval_chunks = set(p["chunk_id"] for p in eval_pairs)
    overlap = train_chunks & eval_chunks

    print(f"\n{'='*50}")
    print(f"Dataset Split Statistics")
    print(f"{'='*50}")
    print(f"\nTotal pairs:  {len(train_pairs) + len(eval_pairs)}")
    print(f"Total chunks: {len(train_chunks) + len(eval_chunks)}")
    print(f"\nTrain:")
    print(f"  Pairs:  {len(train_pairs)}")
    print(f"  Chunks: {len(train_chunks)}")
    print(f"  Avg queries/chunk: "
          f"{len(train_pairs) / len(train_chunks):.1f}")
    print(f"\nEval:")
    print(f"  Pairs:  {len(eval_pairs)}")
    print(f"  Chunks: {len(eval_chunks)}")
    print(f"  Avg queries/chunk: "
          f"{len(eval_pairs) / len(eval_chunks):.1f}")
    print(f"\nChunk overlap (should be 0): {len(overlap)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import argparse

    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # HuggingFace dataset repository ID
    DEFAULT_REPO_ID = "danielnoumon/eu-ai-act-nl-queries"

    # Output directories
    DEFAULT_TRAIN_DIR = (
        PROJECT_ROOT / "data" / "processed" / "train"
    )
    DEFAULT_EVAL_DIR = (
        PROJECT_ROOT / "data" / "processed" / "eval"
    )

    # Fraction of unique chunks held out for evaluation.
    # 15% gives ~86 eval chunks (~344 pairs) — enough for
    # stable MRR/NDCG/Recall estimates without starving training.
    EVAL_FRACTION = 0.15

    # Random seed for reproducible splits.
    SEED = 42

    # -------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Prepare train/eval splits for MNRL fine-tuning"
    )
    parser.add_argument(
        "--repo-id", type=str, default=DEFAULT_REPO_ID,
        help="HuggingFace dataset repo ID"
    )
    parser.add_argument(
        "--train-dir", type=str, default=str(DEFAULT_TRAIN_DIR),
        help="Output directory for train dataset"
    )
    parser.add_argument(
        "--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR),
        help="Output directory for eval dataset"
    )
    parser.add_argument(
        "--eval-fraction", type=float, default=EVAL_FRACTION,
        help="Fraction of chunks for eval (default: 0.15)"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # Load pairs from HuggingFace
    print(f"Loading pairs from HuggingFace: {args.repo_id}")
    pairs = load_pairs(args.repo_id)
    print(f"Loaded {len(pairs)} pairs")

    # Split
    train_pairs, eval_pairs = chunk_level_split(
        pairs,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    print_split_stats(train_pairs, eval_pairs)

    # Build and save train dataset
    train_dir = Path(args.train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = build_train_dataset(train_pairs)
    train_dataset.save_to_disk(str(train_dir))
    print(f"Train dataset saved to: {train_dir}")

    # Build and save eval dicts
    eval_dir = Path(args.eval_dir)
    queries, corpus, relevant_docs = build_eval_dicts(eval_pairs)
    save_eval_dicts(queries, corpus, relevant_docs, eval_dir)
    print(f"\nEval data saved to: {eval_dir}")
