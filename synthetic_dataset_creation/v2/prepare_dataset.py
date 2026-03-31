"""
Merge and split synthetic query-chunk pairs (v2).

Loads single-hop, intra-doc multi-hop, and cross-doc multi-hop pairs
from data/synthetic/v2/, optionally preferring quality-filtered versions.
Merges into a single dataset with a chunk-stratified train/eval split
(no chunk appears in both splits = no data leakage).

Usage:
    python -m synthetic_dataset_creation.v2.prepare_dataset
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from collections import Counter

from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V2_DIR = PROJECT_ROOT / "data" / "synthetic" / "v2"

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

# Single-hop pair files (one per document)
SINGLE_HOP_FILES = {
    "eu_ai_act": "eu_ai_act_query_pairs.jsonl",
    "gdpr": "gdpr_query_pairs.jsonl",
    "uavg": "uavg_query_pairs.jsonl",
}

# Intra-document multi-hop pair files
INTRA_MULTIHOP_FILES = {
    "eu_ai_act": "eu_ai_act_multihop_pairs.jsonl",
    "gdpr": "gdpr_multihop_pairs.jsonl",
    "uavg": "uavg_multihop_pairs.jsonl",
}

# Cross-document multi-hop
CROSSDOC_FILE = "crossdoc_multihop_pairs.jsonl"

DISPLAY_NAMES = {
    "eu_ai_act": "EU AI Act (NL)",
    "gdpr": "AVG/GDPR (NL)",
    "uavg": "UAVG (NL)",
    "crossdoc": "Cross-document",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pairs(path: Path, doc_name: str) -> list[dict]:
    """Load pairs from JSONL, namespace chunk_id to prevent cross-doc collisions."""
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return []

    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            pair["document_name"] = DISPLAY_NAMES.get(doc_name, doc_name)
            # Namespace chunk_id: use source_doc if present (cross-doc pairs),
            # otherwise use the doc_name from the file
            source = pair.get("source_doc", doc_name)
            pair["chunk_id"] = f"{source}_{pair['chunk_id']}"
            pairs.append(pair)
    print(f"  Loaded {len(pairs)} pairs from {path.name}")
    return pairs


def _resolve_path(data_dir: Path, filename: str, use_filtered: bool) -> Path:
    """Return filtered path if it exists and use_filtered=True, else raw path."""
    stem = filename.replace(".jsonl", "")
    filtered = data_dir / f"{stem}_filtered.jsonl"
    raw = data_dir / filename
    if use_filtered and filtered.exists():
        return filtered
    return raw


def load_all_pairs(
    data_dir: Path,
    use_filtered: bool = True,
) -> list[dict]:
    """Load all pair files (single-hop + multi-hop + cross-doc)."""
    all_pairs = []

    print("\n--- Single-hop pairs ---")
    for doc_name, filename in SINGLE_HOP_FILES.items():
        path = _resolve_path(data_dir, filename, use_filtered)
        all_pairs.extend(load_pairs(path, doc_name))

    print("\n--- Intra-doc multi-hop pairs ---")
    for doc_name, filename in INTRA_MULTIHOP_FILES.items():
        path = _resolve_path(data_dir, filename, use_filtered)
        all_pairs.extend(load_pairs(path, doc_name))

    print("\n--- Cross-doc multi-hop pairs ---")
    path = _resolve_path(data_dir, CROSSDOC_FILE, use_filtered)
    all_pairs.extend(load_pairs(path, "crossdoc"))

    return all_pairs


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_queries(pairs: list[dict]) -> list[dict]:
    """Remove pairs with duplicate (anchor, chunk_id) combinations."""
    seen: set[tuple[str, str]] = set()
    deduped = []
    for pair in pairs:
        key = (pair["anchor"], pair["chunk_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(pair)
    removed = len(pairs) - len(deduped)
    if removed:
        print(f"  Removed {removed} duplicate (query, chunk) pairs "
              f"({len(deduped)} remaining)")
    return deduped


# ---------------------------------------------------------------------------
# Train / eval split
# ---------------------------------------------------------------------------

def train_eval_split(
    pairs: list[dict],
    eval_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Chunk-stratified train/eval split — no chunk leaks across splits.

    Groups pairs by chunk_id, then assigns entire chunks (with all their
    pairs) to either train or eval. Stratified by document so each
    document has proportional representation in both splits.
    """
    rng = random.Random(seed)

    # Group pairs by chunk_id
    by_chunk: dict[str, list[dict]] = {}
    for pair in pairs:
        cid = pair["chunk_id"]
        by_chunk.setdefault(cid, []).append(pair)

    # Group chunk_ids by document
    chunks_by_doc: dict[str, list[str]] = {}
    for cid, chunk_pairs in by_chunk.items():
        doc = chunk_pairs[0].get("document_name", "unknown")
        chunks_by_doc.setdefault(doc, []).append(cid)

    # Sample eval chunks per document
    eval_chunks: set[str] = set()
    for doc, chunk_ids in chunks_by_doc.items():
        rng.shuffle(chunk_ids)
        n_eval = max(1, int(len(chunk_ids) * eval_fraction))
        eval_chunks.update(chunk_ids[:n_eval])

    train_pairs = []
    eval_pairs = []
    for pair in pairs:
        if pair["chunk_id"] in eval_chunks:
            eval_pairs.append(pair)
        else:
            train_pairs.append(pair)

    return train_pairs, eval_pairs


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_pair(pair: dict) -> dict:
    """Keep only the fields needed for training + minimal metadata."""
    return {
        "anchor": pair["anchor"],
        "positive": pair["positive"],
        "chunk_id": pair["chunk_id"],
        "section_type": pair.get("section_type", ""),
        "hierarchy_path": pair.get("hierarchy_path", ""),
        "document_name": pair.get("document_name", ""),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(pairs: list[dict], label: str):
    """Print dataset statistics."""
    if not pairs:
        print(f"\n=== {label}: EMPTY ===")
        return

    print(f"\n=== {label} Statistics ===")
    print(f"Total pairs: {len(pairs)}")

    # By document
    doc_counts = Counter(p.get("document_name", "?") for p in pairs)
    print(f"\nBy document:")
    for doc, count in doc_counts.most_common():
        print(f"  {doc}: {count} ({count / len(pairs) * 100:.1f}%)")

    # Unique chunks & queries
    unique_chunks = len(set(p["chunk_id"] for p in pairs))
    unique_queries = len(set(p["anchor"] for p in pairs))
    print(f"\nUnique chunks: {unique_chunks}")
    print(f"Unique queries: {unique_queries}")

    # Query length
    q_lens = [len(p["anchor"]) for p in pairs]
    print(f"\nQuery length (chars): "
          f"min={min(q_lens)}, max={max(q_lens)}, "
          f"mean={sum(q_lens) / len(q_lens):.0f}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    use_filtered: bool = True,
    eval_fraction: float = 0.15,
    seed: int = 42,
):
    """Merge all pair files and create train/eval split."""
    if data_dir is None:
        data_dir = V2_DIR
    if output_dir is None:
        output_dir = data_dir / "dataset"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print("Loading all pairs...")
    pairs = load_all_pairs(data_dir, use_filtered=use_filtered)
    print(f"\nTotal loaded: {len(pairs)} pairs")

    if not pairs:
        print("No pairs found. Run generation steps first.")
        return

    # Deduplicate
    print("\nDeduplicating...")
    pairs = deduplicate_queries(pairs)

    # Split
    print(f"\nSplitting (eval_fraction={eval_fraction})...")
    train, eval_set = train_eval_split(pairs, eval_fraction, seed)

    print_stats(train, "Train")
    print_stats(eval_set, "Eval")

    # Save JSONL
    for split_name, split_data in [("train", train), ("eval", eval_set)]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for pair in split_data:
                f.write(json.dumps(clean_pair(pair), ensure_ascii=False) + "\n")
        print(f"\n  {split_name}: {len(split_data)} pairs -> {path}")

    # Save HF dataset
    for split_name, split_data in [("train", train), ("eval", eval_set)]:
        ds = Dataset.from_dict({
            "anchor": [p["anchor"] for p in split_data],
            "positive": [p["positive"] for p in split_data],
            "chunk_id": [p["chunk_id"] for p in split_data],
            "document_name": [p.get("document_name", "") for p in split_data],
        })
        hf_path = output_dir / f"hf_{split_name}"
        ds.save_to_disk(str(hf_path))
        print(f"  HF {split_name} dataset: {hf_path}")

    print("\nDataset preparation complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # CONFIG
    # -----------------------------------------------------------------------

    # Use filtered (quality-scored) pairs if available, else fall back to raw
    USE_FILTERED = True

    # Fraction of chunks to hold out for eval (by chunk, not by pair)
    EVAL_FRACTION = 0.15

    SEED = 42

    # -----------------------------------------------------------------------

    run(
        use_filtered=USE_FILTERED,
        eval_fraction=EVAL_FRACTION,
        seed=SEED,
    )
