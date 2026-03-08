"""Interactive dataset viewer for the HF dataset."""

from pathlib import Path
from datasets import load_from_disk
import random


def view_dataset(dataset_path: str, num_samples: int = 10):
    """Load and display dataset samples."""
    dataset = load_from_disk(dataset_path)
    
    print(f"=== Dataset Overview ===\n")
    print(f"Total examples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print(f"Features: {dataset.features}\n")
    
    print(f"=== Random Samples (n={num_samples}) ===\n")
    
    # Get random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for i, idx in enumerate(indices, 1):
        example = dataset[idx]
        print(f"--- Sample {i} (index {idx}) ---")
        print(f"Query: {example['query']}")
        print(f"Chunk: {example['chunk'][:200]}...")
        if 'section_type' in example:
            print(f"Section: {example['section_type']}")
            print(f"Path: {example['hierarchy_path']}")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View HF dataset samples")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent / "data" / "hf_dataset"
        ),
        help="Path to HF dataset directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of random samples to display",
    )
    args = parser.parse_args()
    
    view_dataset(args.dataset_path, args.num_samples)
