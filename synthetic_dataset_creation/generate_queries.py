"""
Synthetic query generation for embedding fine-tuning.

For each chunk from the EU AI Act (NL), generates diverse Dutch queries
that the chunk would answer. Outputs (anchor, positive) pairs in JSONL
format, ready for Sentence Transformers MNRL training.

Requires: Azure OpenAI credentials in .env or environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNKS_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "chunks" / "chunks_without_context.jsonl"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "synthetic" / "query_pairs.jsonl"
)

DEPLOYMENT = os.getenv("DEPLOYMENT_NAME_GPT5_MINI", "gpt-5-mini")
QUERIES_PER_CHUNK = 4
MAX_CONCURRENT = 10

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Je bent een expert in de EU AI-verordening (EU AI Act) in het Nederlands.

Je taak: gegeven een tekstfragment uit de EU AI Act, genereer {n} diverse \
zoekquery's in het Nederlands die een gebruiker zou kunnen stellen en waarvoor \
dit fragment het relevante antwoord bevat.

Regels:
- Schrijf ALLEEN in het Nederlands.
- Elke query moet een andere invalshoek hebben. Wissel af tussen:
  * Feitelijke vragen ("Wat bepaalt artikel X over...?")
  * Definitievragen ("Wat wordt bedoeld met...?")
  * Procedurele vragen ("Welke stappen zijn vereist voor...?")
  * Scenariovragen ("Een bedrijf wil X inzetten voor Y, welke regels gelden?")
- Query's moeten realistisch zijn — alsof een jurist, beleidsmaker of \
compliance officer ze zou stellen.
- Verwijs NIET letterlijk naar artikelnummers in de query (de gebruiker kent \
die nummers vaak niet).
- Houd query's beknopt (1-2 zinnen).
- Antwoord met een JSON-object met een "queries" veld.

Voorbeeld output:
{{"queries": ["Welke AI-systemen zijn verboden onder de EU AI Act?", \
"Wanneer wordt een AI-systeem als hoog risico beschouwd?", \
"Moet een aanbieder van AI een conformiteitsbeoordeling laten uitvoeren?", \
"Wat zijn de transparantieverplichtingen voor chatbots?"]}}
"""


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------

class GeneratedQueries(BaseModel):
    """Structured output schema for LLM query generation."""
    queries: list[str] = Field(
        description="Lijst van diverse zoekquery's in het Nederlands"
    )


USER_PROMPT_TEMPLATE = """\
Tekstfragment (bron: EU AI Act NL):
---
{chunk_text}
---

Metadata:
- Type: {section_type}
- Locatie: {hierarchy_path}

Genereer {n} diverse zoekquery's waarvoor bovenstaand fragment het antwoord is.\
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QueryPair:
    anchor: str
    positive: str
    chunk_id: int
    section_type: str
    hierarchy_path: str


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

async def generate_queries_for_chunk(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    deployment: str,
    chunk: dict,
    semaphore: asyncio.Semaphore,
    n: int = QUERIES_PER_CHUNK,
) -> list[QueryPair]:
    """Call the LLM to generate N queries for a single chunk using Responses API."""
    user_msg = USER_PROMPT_TEMPLATE.format(
        chunk_text=chunk["text"],
        section_type=chunk["section_type"],
        hierarchy_path=chunk["hierarchy_path"],
        n=n,
    )

    # Combine system and user messages into single input
    full_input = f"{SYSTEM_PROMPT.format(n=n)}\n\n{user_msg}"

    async with semaphore:
        try:
            response = await client.post(
                endpoint,
                headers={
                    "api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": deployment,
                    "input": full_input,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

            # Responses API returns output as a list with message content
            output_items = data.get("output", [])
            # Find the message item (type='message')
            content = "{}"
            for item in output_items:
                if item.get("type") == "message":
                    msg_content = item.get("content", [])
                    if msg_content and msg_content[0].get("type") == "output_text":
                        content = msg_content[0].get("text", "{}")
                        break

            parsed = GeneratedQueries.model_validate_json(content)
            queries = parsed.queries
        except Exception as e:
            print(f"  [ERROR] chunk {chunk['chunk_id']}: {e}")
            return []

    pairs = []
    for q in queries:
        pairs.append(QueryPair(
            anchor=q,
            positive=chunk["text"],
            chunk_id=chunk["chunk_id"],
            section_type=chunk["section_type"],
            hierarchy_path=chunk["hierarchy_path"],
        ))
    return pairs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(
    chunks_path: Path | None = None,
    output_path: Path | None = None,
    n_queries: int = QUERIES_PER_CHUNK,
    max_chunks: int | None = None,
):
    """Generate queries for all chunks and save to JSONL."""
    chunks_path = chunks_path or CHUNKS_PATH
    output_path = output_path or OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    if max_chunks:
        chunks = chunks[:max_chunks]

    print(f"Generating {n_queries} queries per chunk for {len(chunks)} chunks "
          f"(deployment: {DEPLOYMENT}) ...")

    # Use Responses API endpoint directly
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_GPT5_MINI", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = DEPLOYMENT

    client = httpx.AsyncClient()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        generate_queries_for_chunk(
            client, endpoint, api_key, deployment, chunk, semaphore, n_queries
        )
        for chunk in chunks
    ]

    all_pairs: list[QueryPair] = []
    start = time.time()

    # Process with progress tracking
    completed = 0
    for coro in asyncio.as_completed(tasks):
        pairs = await coro
        all_pairs.extend(pairs)
        completed += 1
        if completed % 50 == 0 or completed == len(tasks):
            elapsed = time.time() - start
            print(f"  {completed}/{len(tasks)} chunks processed "
                  f"({len(all_pairs)} pairs, {elapsed:.0f}s)")

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    print(f"\nDone. {len(all_pairs)} query-chunk pairs saved to {output_path}")
    print(f"Time: {elapsed:.0f}s")

    return all_pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic queries for embedding fine-tuning"
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None,
        help="Limit number of chunks to process (for testing)"
    )
    parser.add_argument(
        "--queries-per-chunk", type=int, default=QUERIES_PER_CHUNK,
        help=f"Number of queries to generate per chunk (default: {QUERIES_PER_CHUNK})"
    )
    args = parser.parse_args()

    asyncio.run(run(
        n_queries=args.queries_per_chunk,
        max_chunks=args.max_chunks,
    ))
