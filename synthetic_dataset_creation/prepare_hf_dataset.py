"""
Prepare query-chunk pairs for Hugging Face upload.

Converts the JSONL format to a Hugging Face Dataset with proper schema
and metadata for embedding fine-tuning tasks.
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from collections import Counter


def load_query_pairs(jsonl_path: str) -> list[dict]:
    """Load query pairs from JSONL file."""
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def create_hf_dataset(
    pairs: list[dict],
    include_metadata: bool = True,
    document_name: str = "EU AI Act (NL)"
) -> Dataset:
    """
    Convert query pairs to Hugging Face Dataset format.
    
    Args:
        pairs: List of query pair dicts
        include_metadata: If True, include chunk_id, section_type, hierarchy_path
        document_name: Name of the source document
    
    Returns:
        Hugging Face Dataset
    """
    # Generate unique question IDs (sequential across all pairs)
    question_ids = list(range(len(pairs)))
    
    if include_metadata:
        data = {
            "question_id": question_ids,
            "query": [p["anchor"] for p in pairs],
            "chunk": [p["positive"] for p in pairs],
            "document_name": [document_name] * len(pairs),
            "chunk_id": [p["chunk_id"] for p in pairs],
            "section_type": [p["section_type"] for p in pairs],
            "hierarchy_path": [p["hierarchy_path"] for p in pairs],
        }
    else:
        data = {
            "question_id": question_ids,
            "query": [p["anchor"] for p in pairs],
            "chunk": [p["positive"] for p in pairs],
            "document_name": [document_name] * len(pairs),
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
    import argparse
    
    # -----------------------------------------------------------------------
    # CONFIG: Adjust these parameters for your use case
    # -----------------------------------------------------------------------
    
    # Input JSONL file path
    DEFAULT_INPUT = (
        Path(__file__).resolve().parent.parent
        / "data" / "synthetic" / "query_pairs.jsonl"
    )
    
    # Output directory for HF dataset
    DEFAULT_OUTPUT = (
        Path(__file__).resolve().parent.parent
        / "data" / "hf_dataset"
    )
    
    # Include metadata columns (chunk_id, section_type, hierarchy_path)
    # Set to False for minimal dataset (just query + chunk)
    INCLUDE_METADATA = True
    
    # -----------------------------------------------------------------------
    
    parser = argparse.ArgumentParser(
        description="Prepare query-chunk pairs for Hugging Face upload"
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT),
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help="Output directory for HF dataset"
    )
    parser.add_argument(
        "--no-metadata", action="store_true",
        help="Exclude metadata columns (minimal dataset)"
    )
    args = parser.parse_args()
    
    # Load pairs
    print(f"Loading query pairs from: {args.input}")
    pairs = load_query_pairs(args.input)
    print(f"Loaded {len(pairs)} pairs")
    
    # Create HF dataset
    include_metadata = INCLUDE_METADATA and not args.no_metadata
    dataset = create_hf_dataset(pairs, include_metadata=include_metadata)
    
    # Print stats
    print_dataset_stats(dataset)
    
    # Save to disk
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    print(f"\n✓ Dataset saved to: {output_path}")
    
    # Also save as parquet for easy HF upload
    parquet_path = output_path.parent / f"{output_path.name}.parquet"
    dataset.to_parquet(str(parquet_path))
    print(f"✓ Parquet file saved to: {parquet_path}")
    
    print(f"\n=== Upload Instructions ===\n")
    print("To upload to Hugging Face:")
    print("1. Install: pip install huggingface_hub")
    print("2. Login: huggingface-cli login")
    print("3. Upload:")
    print(f"   from datasets import load_from_disk")
    print(f"   dataset = load_from_disk('{output_path}')")
    print(f"   dataset.push_to_hub('your-username/eu-ai-act-nl-queries')")
    print("\nOr upload the parquet file directly via the HF web UI.")
