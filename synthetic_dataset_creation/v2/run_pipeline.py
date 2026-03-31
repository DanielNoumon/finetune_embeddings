"""
V2 Synthetic Dataset Pipeline — Orchestrator

Runs the full generation pipeline or individual steps. All output goes to
data/synthetic/v2/ so v1 data in data/synthetic/ is untouched.

Steps:
    1  Single-hop query generation   (reuses v1 generate_queries)
    2  Intra-doc multi-hop           (reuses v1 generate_multihop_queries)
    3  Cross-doc multi-hop           (NEW — TF-IDF grouping)
    4  Quality scoring + filtering   (reuses v1 score_and_filter_queries)
    5  Merge + train/eval split      (NEW — v2 prepare_dataset)

Usage:
    python -m synthetic_dataset_creation.v2.run_pipeline              # all
    python -m synthetic_dataset_creation.v2.run_pipeline --step 1     # single-hop only
    python -m synthetic_dataset_creation.v2.run_pipeline --step 3     # cross-doc only
    python -m synthetic_dataset_creation.v2.run_pipeline --step 4 5   # score + merge

Environment variables:
    LLM_BASE_URL  — OpenAI-compatible endpoint (default: http://localhost:11434/v1)
    LLM_MODEL     — Model name (default: qwen3:30b-a3b)
    LLM_API_KEY   — API key (default: ollama)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V2_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "v2"

# Ensure project root is on path for v1 imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Per-document configuration
# ---------------------------------------------------------------------------

# Single-hop queries per chunk (UAVG boosted to compensate for small corpus)
QUERIES_PER_CHUNK = {
    "eu_ai_act": 6,
    "gdpr": 6,
    "uavg": 10,
}

MULTIHOP_QUERIES_PER_GROUP = 2
CROSSDOC_QUERIES_PER_GROUP = 2
SCORE_THRESHOLD = 7.0


# ---------------------------------------------------------------------------
# Step 1: Single-hop query generation
# ---------------------------------------------------------------------------

async def step_1_single_hop(
    base_url: str, model: str, api_key: str, max_concurrent: int,
):
    """Generate single-hop queries for all documents (v1 logic, v2 output)."""
    from synthetic_dataset_creation.generate_queries import (
        run as gen_run, DOCUMENTS, QueryGenConfig,
    )

    print("\n" + "=" * 60)
    print("STEP 1: Single-hop query generation")
    print("=" * 60)

    V2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for doc_name, orig in DOCUMENTS.items():
        n_queries = QUERIES_PER_CHUNK.get(doc_name, 6)

        v2_config = QueryGenConfig(
            name=orig.name,
            chunks_path=orig.chunks_path,
            output_path=V2_OUTPUT_DIR / f"{doc_name}_query_pairs.jsonl",
            system_prompt=orig.system_prompt,
            user_prompt_template=orig.user_prompt_template,
        )

        await gen_run(
            config=v2_config,
            base_url=base_url,
            model=model,
            api_key=api_key,
            n_queries=n_queries,
            max_concurrent=max_concurrent,
        )


# ---------------------------------------------------------------------------
# Step 2: Intra-document multi-hop
# ---------------------------------------------------------------------------

async def step_2_intra_multihop(
    base_url: str, model: str, api_key: str, max_concurrent: int,
):
    """Generate intra-document multi-hop queries (v1 logic, v2 output)."""
    from synthetic_dataset_creation.generate_multihop_queries import (
        run as mh_run, DOCUMENTS, MultiHopConfig,
    )

    print("\n" + "=" * 60)
    print("STEP 2: Intra-document multi-hop query generation")
    print("=" * 60)

    for doc_name, orig in DOCUMENTS.items():
        v2_config = MultiHopConfig(
            name=orig.name,
            chunks_path=orig.chunks_path,
            output_path=V2_OUTPUT_DIR / f"{doc_name}_multihop_pairs.jsonl",
            system_prompt=orig.system_prompt,
            user_prompt_template=orig.user_prompt_template,
        )

        await mh_run(
            config=v2_config,
            base_url=base_url,
            model=model,
            api_key=api_key,
            n_queries=MULTIHOP_QUERIES_PER_GROUP,
            max_concurrent=max_concurrent,
        )


# ---------------------------------------------------------------------------
# Step 3: Cross-document multi-hop
# ---------------------------------------------------------------------------

async def step_3_crossdoc_multihop(
    base_url: str, model: str, api_key: str, max_concurrent: int,
):
    """Generate cross-document multi-hop queries (v2-only)."""
    from synthetic_dataset_creation.v2.generate_crossdoc_multihop import (
        run as xdoc_run,
    )

    print("\n" + "=" * 60)
    print("STEP 3: Cross-document multi-hop query generation")
    print("=" * 60)

    await xdoc_run(
        base_url=base_url,
        model=model,
        api_key=api_key,
        n_queries=CROSSDOC_QUERIES_PER_GROUP,
        max_concurrent=max_concurrent,
    )


# ---------------------------------------------------------------------------
# Step 4: Quality scoring + filtering
# ---------------------------------------------------------------------------

async def step_4_score_and_filter(
    base_url: str, model: str, api_key: str, max_concurrent: int,
):
    """Score and filter all generated pair files in v2 output dir."""
    from synthetic_dataset_creation.score_and_filter_queries import (
        run as score_run,
    )

    print("\n" + "=" * 60)
    print("STEP 4: Quality scoring and filtering")
    print("=" * 60)

    # Find all *_pairs.jsonl files (exclude already-filtered/rejected/scored)
    pair_files = sorted(
        p for p in V2_OUTPUT_DIR.glob("*_pairs.jsonl")
        if "_filtered" not in p.stem
        and "_rejected" not in p.stem
        and "_scored" not in p.stem
    )

    if not pair_files:
        print("No pair files found in v2 output dir. Run steps 1-3 first.")
        return

    for input_path in pair_files:
        stem = input_path.stem
        output_path = V2_OUTPUT_DIR / f"{stem}_filtered.jsonl"
        rejected_path = V2_OUTPUT_DIR / f"{stem}_rejected.jsonl"

        print(f"\n--- Scoring: {input_path.name} ---")

        await score_run(
            input_path=input_path,
            output_path=output_path,
            rejected_path=rejected_path,
            base_url=base_url,
            model=model,
            api_key=api_key,
            threshold=SCORE_THRESHOLD,
            max_concurrent=max_concurrent,
        )


# ---------------------------------------------------------------------------
# Step 5: Merge + train/eval split
# ---------------------------------------------------------------------------

def step_5_prepare_dataset():
    """Merge all filtered pairs and create train/eval split."""
    from synthetic_dataset_creation.v2.prepare_dataset import run as prep_run

    print("\n" + "=" * 60)
    print("STEP 5: Merge and train/eval split")
    print("=" * 60)

    prep_run(data_dir=V2_OUTPUT_DIR, use_filtered=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V2 Synthetic Dataset Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step", type=int, nargs="+", choices=[1, 2, 3, 4, 5],
        help="Run specific step(s). Default: all steps 1-5.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=2,
        help="Max concurrent LLM API requests (default: 2)",
    )
    args = parser.parse_args()

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("LLM_MODEL", "qwen3:30b-a3b")
    api_key = os.getenv("LLM_API_KEY", "ollama")

    print(f"LLM endpoint: {base_url}")
    print(f"Model: {model}")
    print(f"Output directory: {V2_OUTPUT_DIR}")

    steps = args.step if args.step else [1, 2, 3, 4, 5]

    for step in steps:
        if step == 1:
            asyncio.run(step_1_single_hop(
                base_url, model, api_key, args.max_concurrent,
            ))
        elif step == 2:
            asyncio.run(step_2_intra_multihop(
                base_url, model, api_key, args.max_concurrent,
            ))
        elif step == 3:
            asyncio.run(step_3_crossdoc_multihop(
                base_url, model, api_key, args.max_concurrent,
            ))
        elif step == 4:
            asyncio.run(step_4_score_and_filter(
                base_url, model, api_key, args.max_concurrent,
            ))
        elif step == 5:
            step_5_prepare_dataset()

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
