"""
Cross-document multi-hop query generation for embedding fine-tuning (v2).

Finds related chunk pairs across different legal documents using TF-IDF
similarity, then generates queries that require information from both
documents to answer. Each query is "unrolled" into multiple (query, chunk)
pairs for contrastive training.

Trustworthiness is ensured by:
1. TF-IDF band-pass filter (only semantically related cross-doc pairs)
2. The LLM must produce a coherent multi-hop query (natural quality gate)
3. Downstream quality scoring filters out forced/weak connections

Usage:
    python -m synthetic_dataset_creation.v2.generate_crossdoc_multihop
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project root & paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CHUNK_PATHS = {
    "eu_ai_act": PROJECT_ROOT / "data" / "chunks" / "eu_ai_act" / "chunks_without_context.jsonl",
    "gdpr": PROJECT_ROOT / "data" / "chunks" / "gdpr" / "chunks_without_context.jsonl",
    "uavg": PROJECT_ROOT / "data" / "chunks" / "uavg" / "chunks_without_context.jsonl",
}

DISPLAY_NAMES = {
    "eu_ai_act": "EU AI Act",
    "gdpr": "AVG/GDPR",
    "uavg": "UAVG",
}

V2_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "v2"


# ---------------------------------------------------------------------------
# Cross-document system & user prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_CROSSDOC = """\
Je bent een expert in Nederlandse en Europese regelgeving op het gebied van \
AI, gegevensbescherming en privacy (EU AI Act, AVG/GDPR, UAVG).

Je taak: gegeven TWEE tekstfragmenten uit VERSCHILLENDE wetten ({doc_a_name} \
en {doc_b_name}), genereer {n} zoekquery's in het Nederlands die informatie \
uit BEIDE wetten combineren. De query moet niet beantwoord kunnen worden door \
slechts één wet — het antwoord vereist het verbinden van bepalingen uit \
beide wetten.

Regels:
- Schrijf ALLEEN in het Nederlands.
- Elke query moet informatie uit beide fragmenten/wetten combineren.
- Varieer tussen:
  * Vergelijkingsvragen ("Hoe verhoudt [concept uit {doc_a_name}] zich tot \
[concept uit {doc_b_name}]?")
  * Toepassingsvragen ("Hoe werken de vereisten uit {doc_a_name} samen met \
{doc_b_name} in de praktijk?")
  * Scenariovragen die bepalingen uit beide wetten raken ("Een organisatie \
wil X doen, welke verplichtingen gelden vanuit zowel {doc_a_name} als \
{doc_b_name}?")
  * Conflictvragen ("Kan er spanning ontstaan tussen [verplichting uit \
{doc_a_name}] en [recht uit {doc_b_name}]?")
- Query's moeten realistisch zijn — alsof een jurist, privacyfunctionaris \
of compliance officer ze zou stellen.
- Verwijs NIET naar artikelnummers of fragmentnummers.
- Antwoord met een JSON-object met een "queries" veld.

Voorbeeld output:
{{"queries": [\
"Hoe beïnvloeden de AVG-vereisten voor gegevensbescherming de \
conformiteitsbeoordeling van hoog-risico AI-systemen onder de AI Act?", \
"Een zorginstelling wil AI inzetten voor diagnoses met patiëntgegevens, \
welke combinatie van verplichtingen geldt vanuit beide wetten?"]}}"""

USER_PROMPT_CROSSDOC = """\
Fragment 1 (bron: {doc_a_name}):
---
{chunk_a_text}
---
(Type: {chunk_a_section_type} | Locatie: {chunk_a_hierarchy_path})

Fragment 2 (bron: {doc_b_name}):
---
{chunk_b_text}
---
(Type: {chunk_b_section_type} | Locatie: {chunk_b_hierarchy_path})

Genereer {n} zoekquery's die informatie uit beide wetten combineren.\
"""


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class GeneratedQueries(BaseModel):
    queries: list[str] = Field(
        description="Lijst van cross-document zoekquery's in het Nederlands"
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CrossDocPair:
    anchor: str
    positive: str
    chunk_id: int
    section_type: str
    hierarchy_path: str
    source_doc: str
    is_multihop: bool = True
    is_crossdoc: bool = True
    hop_count: int = 2
    source_chunk_ids: list[str] | None = None


# ---------------------------------------------------------------------------
# TF-IDF cross-document grouping
# ---------------------------------------------------------------------------

def load_all_chunks() -> dict[str, list[dict]]:
    """Load chunks from all configured documents."""
    all_chunks = {}
    for doc_name, path in CHUNK_PATHS.items():
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        all_chunks[doc_name] = chunks
        print(f"  Loaded {len(chunks)} chunks from {doc_name}")
    return all_chunks


def group_chunks_cross_document(
    doc_chunks: dict[str, list[dict]],
    min_similarity: float = 0.25,
    max_similarity: float = 0.85,
    max_per_chunk: int = 2,
    max_groups: int | None = None,
    seed: int = 42,
) -> list[list[dict]]:
    """Find cross-document chunk pairs using TF-IDF cosine similarity.

    Band-pass filter ensures pairs are:
    - Similar enough (>min) to be genuinely related
    - Not too similar (<max) to avoid near-duplicate text

    Each chunk can appear in at most max_per_chunk groups to avoid
    over-representation.

    Args:
        doc_chunks: Mapping of doc_name -> list of chunk dicts
        min_similarity: Minimum cosine similarity to consider
        max_similarity: Maximum cosine similarity (avoid near-duplicates)
        max_per_chunk: Max groups a single chunk can appear in
        max_groups: Total cap on groups returned
        seed: Random seed for reproducibility

    Returns:
        List of [chunk_a, chunk_b] groups with added "source_doc" field.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    rng = random.Random(seed)
    doc_names = list(doc_chunks.keys())

    # Flatten with labels
    all_texts = []
    all_meta = []  # (doc_name, chunk_dict)
    for doc_name in doc_names:
        for chunk in doc_chunks[doc_name]:
            all_texts.append(chunk["text"])
            all_meta.append((doc_name, chunk))

    print(f"\n  Computing TF-IDF for {len(all_texts)} chunks "
          f"across {len(doc_names)} documents...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    sim_matrix = cos_sim(tfidf_matrix)

    # Collect cross-doc pairs within similarity band
    candidates = []
    for i in range(len(all_meta)):
        for j in range(i + 1, len(all_meta)):
            if all_meta[i][0] == all_meta[j][0]:
                continue  # skip same-document
            sim = sim_matrix[i, j]
            if min_similarity <= sim <= max_similarity:
                candidates.append((i, j, float(sim)))

    candidates.sort(key=lambda x: x[2], reverse=True)
    print(f"  Found {len(candidates)} candidate cross-doc pairs "
          f"(cosine in [{min_similarity}, {max_similarity}])")

    # Greedy selection respecting per-chunk limit
    chunk_usage: dict[int, int] = {}
    doc_pair_counts: dict[tuple[str, str], int] = {}
    groups = []

    for i, j, sim in candidates:
        if chunk_usage.get(i, 0) >= max_per_chunk:
            continue
        if chunk_usage.get(j, 0) >= max_per_chunk:
            continue

        chunk_a = {**all_meta[i][1], "source_doc": all_meta[i][0]}
        chunk_b = {**all_meta[j][1], "source_doc": all_meta[j][0]}
        groups.append([chunk_a, chunk_b])

        chunk_usage[i] = chunk_usage.get(i, 0) + 1
        chunk_usage[j] = chunk_usage.get(j, 0) + 1

        pair_key = tuple(sorted([all_meta[i][0], all_meta[j][0]]))
        doc_pair_counts[pair_key] = doc_pair_counts.get(pair_key, 0) + 1

    rng.shuffle(groups)
    if max_groups:
        groups = groups[:max_groups]

    print(f"  Selected {len(groups)} cross-doc groups:")
    for pair_key, count in sorted(doc_pair_counts.items()):
        print(f"    {pair_key[0]} <-> {pair_key[1]}: {count}")

    return groups


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Extract a JSON object from LLM output that may contain markdown fences or thinking tags."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text.strip()


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

async def generate_crossdoc_queries(
    client: AsyncOpenAI,
    model: str,
    chunk_group: list[dict],
    semaphore: asyncio.Semaphore,
    n: int = 2,
) -> list[CrossDocPair]:
    """Generate cross-document multi-hop queries for a pair of chunks."""
    doc_a_name = DISPLAY_NAMES[chunk_group[0]["source_doc"]]
    doc_b_name = DISPLAY_NAMES[chunk_group[1]["source_doc"]]

    system_msg = SYSTEM_PROMPT_CROSSDOC.format(
        doc_a_name=doc_a_name,
        doc_b_name=doc_b_name,
        n=n,
    )
    user_msg = USER_PROMPT_CROSSDOC.format(
        doc_a_name=doc_a_name,
        doc_b_name=doc_b_name,
        chunk_a_text=chunk_group[0]["text"],
        chunk_a_section_type=chunk_group[0]["section_type"],
        chunk_a_hierarchy_path=chunk_group[0]["hierarchy_path"],
        chunk_b_text=chunk_group[1]["text"],
        chunk_b_section_type=chunk_group[1]["section_type"],
        chunk_b_hierarchy_path=chunk_group[1]["hierarchy_path"],
        n=n,
    )

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
            )
            content = response.choices[0].message.content or "{}"
            content = _extract_json(content)
            parsed = GeneratedQueries.model_validate_json(content)
            queries = parsed.queries
        except Exception as e:
            ids = [f"{c['source_doc']}:{c['chunk_id']}" for c in chunk_group]
            print(f"  [ERROR] chunks {ids}: {e}")
            return []

    # Unroll: one pair per (query, chunk)
    source_ids = [f"{c['source_doc']}_{c['chunk_id']}" for c in chunk_group]
    pairs = []
    for query in queries:
        for chunk in chunk_group:
            pairs.append(CrossDocPair(
                anchor=query,
                positive=chunk["text"],
                chunk_id=chunk["chunk_id"],
                section_type=chunk["section_type"],
                hierarchy_path=chunk["hierarchy_path"],
                source_doc=chunk["source_doc"],
                is_multihop=True,
                is_crossdoc=True,
                hop_count=2,
                source_chunk_ids=source_ids,
            ))
    return pairs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(
    base_url: str,
    model: str,
    api_key: str = "ollama",
    n_queries: int = 2,
    min_similarity: float = 0.25,
    max_similarity: float = 0.85,
    max_per_chunk: int = 2,
    max_groups: int | None = None,
    max_concurrent: int = 10,
    output_path: Path | None = None,
):
    """Generate cross-document multi-hop queries.

    1. Load chunks from all documents
    2. Find cross-doc pairs via TF-IDF similarity (band-pass filter)
    3. Generate multi-hop queries per pair
    4. Unroll into (query, chunk) training pairs
    """
    if output_path is None:
        output_path = V2_OUTPUT_DIR / "crossdoc_multihop_pairs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all chunks
    print("Loading chunks from all documents...")
    doc_chunks = load_all_chunks()

    if len(doc_chunks) < 2:
        print("Need at least 2 documents for cross-doc multi-hop. Skipping.")
        return []

    # Group chunks across documents
    print("\nGrouping chunks across documents (TF-IDF similarity)...")
    groups = group_chunks_cross_document(
        doc_chunks,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
        max_per_chunk=max_per_chunk,
        max_groups=max_groups,
    )

    if not groups:
        print("No cross-doc groups found. Try lowering min_similarity.")
        return []

    print(f"\nGenerating {n_queries} queries per group for {len(groups)} groups...")
    print(f"  Expected pairs: ~{len(groups) * n_queries * 2}")
    print(f"  Model: {model}")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        generate_crossdoc_queries(client, model, group, semaphore, n_queries)
        for group in groups
    ]

    all_pairs: list[CrossDocPair] = []
    start = time.time()
    completed = 0

    for coro in asyncio.as_completed(tasks):
        pairs = await coro
        all_pairs.extend(pairs)
        completed += 1
        if completed % 25 == 0 or completed == len(tasks):
            elapsed = time.time() - start
            print(f"  {completed}/{len(tasks)} groups processed "
                  f"({len(all_pairs)} pairs, {elapsed:.0f}s)")

    # Backup existing file
    if output_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output_path.parent / f"{output_path.stem}_backup_{timestamp}{output_path.suffix}"
        shutil.copy2(output_path, backup)
        print(f"\n[BACKUP] Existing file backed up to: {backup.name}")

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")

    unique_queries = len(set(p.anchor for p in all_pairs))
    elapsed = time.time() - start
    print(f"\nDone. {len(all_pairs)} pairs ({unique_queries} unique queries) "
          f"saved to {output_path}")
    print(f"Time: {elapsed:.0f}s")

    return all_pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # CONFIG: Adjust these parameters for your use case
    # -----------------------------------------------------------------------

    # --- LLM endpoint (OpenAI-compatible: Ollama, vLLM, Azure, etc.) ---
    BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    MODEL = os.getenv("LLM_MODEL", "qwen3:30b-a3b")
    API_KEY = os.getenv("LLM_API_KEY", "ollama")

    # Queries per cross-doc group (each produces 2 unrolled pairs)
    QUERIES_PER_GROUP = 2

    # TF-IDF similarity band-pass thresholds
    MIN_SIMILARITY = 0.25
    MAX_SIMILARITY = 0.85

    # Max groups a single chunk can appear in
    MAX_PER_CHUNK = 2

    # Total cap on cross-doc groups (None = all valid pairs)
    MAX_GROUPS = None

    # Max concurrent API requests (keep low for local models)
    MAX_CONCURRENT = 2

    # -----------------------------------------------------------------------

    asyncio.run(run(
        base_url=BASE_URL,
        model=MODEL,
        api_key=API_KEY,
        n_queries=QUERIES_PER_GROUP,
        min_similarity=MIN_SIMILARITY,
        max_similarity=MAX_SIMILARITY,
        max_per_chunk=MAX_PER_CHUNK,
        max_groups=MAX_GROUPS,
        max_concurrent=MAX_CONCURRENT,
    ))
