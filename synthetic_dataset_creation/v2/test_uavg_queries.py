"""
Quick test: generate 6 queries per chunk for 5 UAVG chunks.
Inspect output to decide the right queries-per-chunk setting.

Usage:
    python -m synthetic_dataset_creation.v2.test_uavg_queries
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from synthetic_dataset_creation.generate_queries import (
    run as gen_run, DOCUMENTS, QueryGenConfig,
)

V2_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "v2"


async def test():
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("LLM_MODEL", "qwen3:30b-a3b")
    api_key = os.getenv("LLM_API_KEY", "ollama")

    orig = DOCUMENTS["uavg"]
    test_config = QueryGenConfig(
        name=orig.name,
        chunks_path=orig.chunks_path,
        output_path=V2_OUTPUT_DIR / "test_uavg_6q_5chunks.jsonl",
        system_prompt=orig.system_prompt,
        user_prompt_template=orig.user_prompt_template,
    )

    pairs = await gen_run(
        config=test_config,
        base_url=base_url,
        model=model,
        api_key=api_key,
        n_queries=6,
        max_chunks=5,
        max_concurrent=2,
    )

    # Print results grouped by chunk for easy inspection
    by_chunk: dict[int, list[str]] = {}
    for p in pairs:
        by_chunk.setdefault(p.chunk_id, []).append(p.anchor)

    print("\n" + "=" * 60)
    print("RESULTS: 6 queries per chunk, 5 UAVG chunks")
    print("=" * 60)

    for chunk_id, queries in sorted(by_chunk.items()):
        print(f"\n--- Chunk {chunk_id} ({len(queries)} queries) ---")
        for i, q in enumerate(queries, 1):
            print(f"  {i:2d}. {q}")


if __name__ == "__main__":
    asyncio.run(test())
