"""
Prepare query-chunk pairs for Hugging Face upload.

Merges query pairs from multiple documents (EU AI Act, GDPR, etc.)
into a single Hugging Face Dataset with proper schema and metadata.
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Document configs: maps JSONL filename stem to display name
DOCUMENT_SOURCES = {
    "eu_ai_act_query_pairs": "EU AI Act (NL)",
    "gdpr_query_pairs": "AVG/GDPR (NL)",
    "uavg_query_pairs": "UAVG (NL)",
}


def load_query_pairs(jsonl_path: Path, document_name: str) -> list[dict]:
    """Load query pairs from JSONL file, adding document_name to each pair."""
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            pair["document_name"] = document_name
            pairs.append(pair)
    return pairs


def load_all_pairs(synthetic_dir: Path) -> list[dict]:
    """Load and merge query pairs from all document JSONL files."""
    all_pairs = []
    for stem, doc_name in DOCUMENT_SOURCES.items():
        path = synthetic_dir / f"{stem}.jsonl"
        if path.exists():
            pairs = load_query_pairs(path, doc_name)
            print(f"  Loaded {len(pairs)} pairs from {path.name}")
            all_pairs.extend(pairs)
        else:
            print(f"  [SKIP] {path.name} not found")
    return all_pairs


def deduplicate_queries(pairs: list[dict]) -> list[dict]:
    """Remove pairs with duplicate query text, keeping the first occurrence."""
    seen = set()
    deduped = []
    for pair in pairs:
        query = pair["anchor"]
        if query not in seen:
            seen.add(query)
            deduped.append(pair)
    n_removed = len(pairs) - len(deduped)
    if n_removed:
        print(f"  Removed {n_removed} duplicate queries ({len(deduped)} remaining)")
    return deduped


def create_hf_dataset(
    pairs: list[dict],
    include_metadata: bool = True,
) -> Dataset:
    """
    Convert query pairs to Hugging Face Dataset format.

    Args:
        pairs: List of query pair dicts (with document_name field)
        include_metadata: If True, include chunk_id, section_type, etc.

    Returns:
        Hugging Face Dataset
    """
    question_ids = list(range(len(pairs)))

    if include_metadata:
        data = {
            "question_id": question_ids,
            "query": [p["anchor"] for p in pairs],
            "chunk": [p["positive"] for p in pairs],
            "document_name": [p["document_name"] for p in pairs],
            "chunk_id": [p["chunk_id"] for p in pairs],
            "section_type": [p["section_type"] for p in pairs],
            "hierarchy_path": [p["hierarchy_path"] for p in pairs],
        }
    else:
        data = {
            "question_id": question_ids,
            "query": [p["anchor"] for p in pairs],
            "chunk": [p["positive"] for p in pairs],
            "document_name": [p["document_name"] for p in pairs],
        }

    return Dataset.from_dict(data)


def print_dataset_stats(dataset: Dataset):
    """Print dataset statistics."""
    print(f"\n=== Dataset Statistics ===\n")
    print(f"Total pairs: {len(dataset)}")
    
    # Query length stats
    query_lengths = [len(q) for q in dataset["query"]]
    print(f"\nQuery length (chars):")
    print(f"  Min: {min(query_lengths)}")
    print(f"  Max: {max(query_lengths)}")
    print(f"  Mean: {sum(query_lengths) / len(query_lengths):.1f}")
    
    # Chunk length stats
    chunk_lengths = [len(c) for c in dataset["chunk"]]
    print(f"\nChunk length (chars):")
    print(f"  Min: {min(chunk_lengths)}")
    print(f"  Max: {max(chunk_lengths)}")
    print(f"  Mean: {sum(chunk_lengths) / len(chunk_lengths):.1f}")
    
    # Section type distribution (if metadata included)
    if "section_type" in dataset.column_names:
        section_counts = Counter(dataset["section_type"])
        print(f"\nSection type distribution:")
        for stype, count in section_counts.most_common():
            print(f"  {stype}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Sample
    print(f"\n=== Sample Pairs ===\n")
    for i in range(min(3, len(dataset))):
        print(f"Query {i+1}: {dataset['query'][i]}")
        print(f"Chunk {i+1}: {dataset['chunk'][i][:100]}...")
        print()


if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # CONFIG: Adjust these parameters for your use case
    # -----------------------------------------------------------------------

    # Directory containing the per-document JSONL files
    SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

    # Output directory for HF dataset
    OUTPUT_DIR = PROJECT_ROOT / "data" / "hf_dataset"

    # Include metadata columns (chunk_id, section_type, hierarchy_path)
    # Set to False for minimal dataset (just query + chunk)
    INCLUDE_METADATA = True

    # -----------------------------------------------------------------------

    # Load and merge all document pairs
    print(f"Loading query pairs from: {SYNTHETIC_DIR}")
    pairs = load_all_pairs(SYNTHETIC_DIR)
    print(f"Total: {len(pairs)} pairs")

    if not pairs:
        print("No pairs found. Run generate_queries.py first.")
        exit(1)

    # Deduplicate exact-match queries
    pairs = deduplicate_queries(pairs)

    # Create HF dataset
    dataset = create_hf_dataset(pairs, include_metadata=INCLUDE_METADATA)

    # Print stats
    print_dataset_stats(dataset)

    # Save to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))
    print(f"\nDataset saved to: {OUTPUT_DIR}")

    # Also save as parquet for easy HF upload
    parquet_path = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}.parquet"
    dataset.to_parquet(str(parquet_path))
    print(f"Parquet file saved to: {parquet_path}")
