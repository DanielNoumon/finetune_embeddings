"""
Synthetic query generation for embedding fine-tuning.

For each chunk from a Dutch EU regulation (EU AI Act, GDPR, etc.),
generates diverse Dutch queries that the chunk would answer. Outputs
(anchor, positive) pairs in JSONL format, ready for Sentence
Transformers MNRL training.

Uses an OpenAI-compatible endpoint (Ollama, vLLM, Azure, etc.)
for query generation.

Usage:
  Set DOC and endpoint config in the __main__ section at the bottom.
"""

from __future__ import annotations

import asyncio
import json
import os
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
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Document-specific configurations
# ---------------------------------------------------------------------------

@dataclass
class QueryGenConfig:
    """Configuration for query generation per document."""
    name: str
    chunks_path: Path
    output_path: Path
    system_prompt: str
    user_prompt_template: str


SYSTEM_PROMPT_EU_AI_ACT = """\
Je bent een expert in de EU AI-verordening (EU AI Act) in het Nederlands.

Je taak: gegeven een tekstfragment uit de EU AI Act, genereer {n} diverse \
zoekquery's in het Nederlands die een gebruiker zou kunnen stellen en waarvoor \
dit fragment het relevante antwoord bevat.

Regels:
- Schrijf ALLEEN in het Nederlands.
- Elke query moet een andere invalshoek hebben. Wissel VERPLICHT af tussen:
  * Feitelijke vragen ("Welke AI-systemen zijn verboden?")
  * Definitievragen ("Wat wordt bedoeld met hoog-risico AI?")
  * Procedurele vragen ("Hoe voldoe ik aan de transparantieverplichtingen?")
  * Scenariovragen ("Een bedrijf wil gezichtsherkenning inzetten, welke regels \
gelden?")
  * Trefwoordzoekopdrachten — korte fragmenten zonder vraagteken, zoals een \
gebruiker in een zoekveld zou typen ("verboden AI-systemen", \
"conformiteitsbeoordeling hoog-risico")
- BELANGRIJK: Genereer minstens 1 procedurele vraag ("Hoe...?", \
"Welke stappen...?"), 1 scenariovraag ("Een bedrijf/organisatie wil..."), \
en 1 trefwoordzoekopdracht per {n} queries.
- Varieer de lengte: mix van korte trefwoorden (20-50 tekens), middellange \
vragen (60-100 tekens) en langere queries (100-150 tekens).
- Query's moeten realistisch zijn — alsof een jurist, beleidsmaker of \
compliance officer ze zou stellen.
- Verwijs NIET letterlijk naar artikelnummers in de query (de gebruiker kent \
die nummers vaak niet).
- Antwoord met een JSON-object met een "queries" veld.

Voorbeeld output:
{{"queries": ["verboden AI-systemen", \
"Hoe voldoe ik aan de conformiteitseisen voor hoog-risico AI-systemen?", \
"Een zorginstelling wil AI gebruiken voor diagnoses, welke verplichtingen gelden?", \
"Wat wordt bedoeld met transparantieverplichtingen voor AI?", \
"AI transparantie eisen", \
"Welke AI-toepassingen vallen onder de hoog-risico categorie?"]}}
"""

SYSTEM_PROMPT_GDPR = """\
Je bent een expert in de Algemene Verordening Gegevensbescherming (AVG/GDPR) \
in het Nederlands.

Je taak: gegeven een tekstfragment uit de AVG, genereer {n} diverse \
zoekquery's in het Nederlands die een gebruiker zou kunnen stellen en waarvoor \
dit fragment het relevante antwoord bevat.

Regels:
- Schrijf ALLEEN in het Nederlands.
- Elke query moet een andere invalshoek hebben. Wissel VERPLICHT af tussen:
  * Feitelijke vragen ("Wanneer is toestemming vereist voor gegevensverwerking?")
  * Definitievragen ("Wat zijn persoonsgegevens volgens de AVG?")
  * Procedurele vragen ("Hoe voer ik een DPIA uit?")
  * Scenariovragen ("Een webshop wil klantgedrag bijhouden, welke regels gelden?")
  * Trefwoordzoekopdrachten — korte fragmenten zonder vraagteken, zoals een \
gebruiker in een zoekveld zou typen ("datalek meldplicht", \
"rechten betrokkenen AVG")
- BELANGRIJK: Genereer minstens 1 procedurele vraag ("Hoe...?", \
"Welke stappen...?"), 1 scenariovraag ("Een bedrijf/organisatie wil..."), \
en 1 trefwoordzoekopdracht per {n} queries.
- Varieer de lengte: mix van korte trefwoorden (20-50 tekens), middellange \
vragen (60-100 tekens) en langere queries (100-150 tekens).
- Query's moeten realistisch zijn — alsof een jurist, privacyfunctionaris (DPO), \
beleidsmaker of compliance officer ze zou stellen.
- Verwijs NIET letterlijk naar artikelnummers in de query (de gebruiker kent \
die nummers vaak niet).
- Antwoord met een JSON-object met een "queries" veld.

Voorbeeld output:
{{"queries": ["datalek meldplicht", \
"Hoe meld ik een datalek bij de toezichthouder?", \
"Een ziekenhuis wil patiëntgegevens delen met een onderzoeksinstelling, mag dat?", \
"Wat houdt het recht op vergetelheid in volgens de privacywet?", \
"rechten betrokkenen AVG", \
"Wanneer moet een organisatie een functionaris gegevensbescherming aanstellen?"]}}
"""

USER_PROMPT_EU_AI_ACT = """\
Tekstfragment (bron: EU AI Act NL):
---
{chunk_text}
---

Metadata:
- Type: {section_type}
- Locatie: {hierarchy_path}

Genereer {n} diverse zoekquery's waarvoor bovenstaand fragment het antwoord is.\
"""

USER_PROMPT_GDPR = """\
Tekstfragment (bron: AVG/GDPR NL):
---
{chunk_text}
---

Metadata:
- Type: {section_type}
- Locatie: {hierarchy_path}

Genereer {n} diverse zoekquery's waarvoor bovenstaand fragment het antwoord is.\
"""

SYSTEM_PROMPT_UAVG = """\
Je bent een expert in de Uitvoeringswet Algemene verordening \
gegevensbescherming (UAVG) — de Nederlandse implementatiewet van de AVG/GDPR.

Je taak: gegeven een tekstfragment uit de UAVG, genereer {n} diverse \
zoekquery's in het Nederlands die een gebruiker zou kunnen stellen en waarvoor \
dit fragment het relevante antwoord bevat.

Regels:
- Schrijf ALLEEN in het Nederlands.
- Elke query moet een andere invalshoek hebben. Wissel VERPLICHT af tussen:
  * Feitelijke vragen ("Welke uitzonderingen kent de UAVG voor bijzondere \
persoonsgegevens?")
  * Definitievragen ("Wat zijn persoonsgegevens van strafrechtelijke aard?")
  * Procedurele vragen ("Hoe legt de Autoriteit persoonsgegevens een boete \
op?")
  * Scenariovragen ("Een onderzoeksinstelling wil medische gegevens \
verwerken zonder toestemming, mag dat?")
  * Trefwoordzoekopdrachten — korte fragmenten zonder vraagteken, zoals een \
gebruiker in een zoekveld zou typen ("UAVG uitzonderingen bijzondere \
gegevens", "boete Autoriteit persoonsgegevens")
- BELANGRIJK: Genereer minstens 1 procedurele vraag ("Hoe...?", \
"Welke stappen...?"), 1 scenariovraag ("Een bedrijf/organisatie wil..."), \
en 1 trefwoordzoekopdracht per {n} queries.
- Varieer de lengte: mix van korte trefwoorden (20-50 tekens), middellange \
vragen (60-100 tekens) en langere queries (100-150 tekens).
- Query's moeten realistisch zijn — alsof een jurist, privacyfunctionaris \
(DPO), beleidsmaker of compliance officer ze zou stellen.
- Verwijs NIET letterlijk naar artikelnummers in de query (de gebruiker kent \
die nummers vaak niet).
- Antwoord met een JSON-object met een "queries" veld.

Voorbeeld output:
{{"queries": ["uitzonderingen verwerking bijzondere persoonsgegevens", \
"Hoe kan de Autoriteit persoonsgegevens handhavend optreden?", \
"Een werkgever wil gezondheidsgegevens van werknemers bijhouden, \
welke regels gelden volgens de Nederlandse privacywet?", \
"Wat is de rol van de Autoriteit persoonsgegevens?", \
"boetebevoegdheid AP UAVG", \
"Wanneer mag een organisatie strafrechtelijke gegevens verwerken?"]}}
"""

USER_PROMPT_UAVG = """\
Tekstfragment (bron: UAVG NL):
---
{chunk_text}
---

Metadata:
- Type: {section_type}
- Locatie: {hierarchy_path}

Genereer {n} diverse zoekquery's waarvoor bovenstaand fragment het antwoord is.\
"""

DOCUMENTS: dict[str, QueryGenConfig] = {
    "eu_ai_act": QueryGenConfig(
        name="eu_ai_act",
        chunks_path=PROJECT_ROOT / "data" / "chunks" / "eu_ai_act" / "chunks_without_context.jsonl",
        output_path=PROJECT_ROOT / "data" / "synthetic" / "eu_ai_act_query_pairs.jsonl",
        system_prompt=SYSTEM_PROMPT_EU_AI_ACT,
        user_prompt_template=USER_PROMPT_EU_AI_ACT,
    ),
    "gdpr": QueryGenConfig(
        name="gdpr",
        chunks_path=PROJECT_ROOT / "data" / "chunks" / "gdpr" / "chunks_without_context.jsonl",
        output_path=PROJECT_ROOT / "data" / "synthetic" / "gdpr_query_pairs.jsonl",
        system_prompt=SYSTEM_PROMPT_GDPR,
        user_prompt_template=USER_PROMPT_GDPR,
    ),
    "uavg": QueryGenConfig(
        name="uavg",
        chunks_path=PROJECT_ROOT / "data" / "chunks" / "uavg" / "chunks_without_context.jsonl",
        output_path=PROJECT_ROOT / "data" / "synthetic" / "uavg_query_pairs.jsonl",
        system_prompt=SYSTEM_PROMPT_UAVG,
        user_prompt_template=USER_PROMPT_UAVG,
    ),
}


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------

class GeneratedQueries(BaseModel):
    """Structured output schema for LLM query generation."""
    queries: list[str] = Field(
        description="Lijst van diverse zoekquery's in het Nederlands"
    )


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

def _extract_json(text: str) -> str:
    """Extract a JSON object from LLM output that may contain markdown fences or thinking tags."""
    # Strip <think>...</think> blocks (e.g. DeepSeek-R1, Qwen3)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Try to find JSON inside markdown code fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # Try to find a raw JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text.strip()


async def generate_queries_for_chunk(
    client: AsyncOpenAI,
    model: str,
    chunk: dict,
    config: QueryGenConfig,
    semaphore: asyncio.Semaphore,
    n: int = 4,
) -> list[QueryPair]:
    """Call the LLM to generate N queries for a single chunk."""
    user_msg = config.user_prompt_template.format(
        chunk_text=chunk["text"],
        section_type=chunk["section_type"],
        hierarchy_path=chunk["hierarchy_path"],
        n=n,
    )

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": config.system_prompt.format(n=n)},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
            )
            content = response.choices[0].message.content or "{}"
            content = _extract_json(content)
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
    config: QueryGenConfig,
    base_url: str,
    model: str,
    api_key: str = "ollama",
    n_queries: int = 4,
    max_chunks: int | None = None,
    max_concurrent: int = 10,
):
    """Generate queries for all chunks and save to JSONL."""
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load chunks
    with open(config.chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    if max_chunks:
        chunks = chunks[:max_chunks]

    print(f"[{config.name}] Generating {n_queries} queries per chunk for "
          f"{len(chunks)} chunks (model: {model}) ...")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        generate_queries_for_chunk(
            client, model, chunk, config, semaphore, n_queries,
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

    # Backup existing file if it exists
    if config.output_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config.output_path.parent / f"{config.output_path.stem}_backup_{timestamp}{config.output_path.suffix}"
        shutil.copy2(config.output_path, backup_path)
        print(f"\n[BACKUP] Existing file backed up to: {backup_path.name}")

    # Save
    with open(config.output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    print(f"\nDone. {len(all_pairs)} query-chunk pairs saved to {config.output_path}")
    print(f"Time: {elapsed:.0f}s")

    return all_pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # CONFIG: Adjust these parameters for your use case
    # -----------------------------------------------------------------------

    # Which document(s) to process: "eu_ai_act", "gdpr", "uavg", or "all"
    DOC = "uavg"

    # --- LLM endpoint (OpenAI-compatible: Ollama, vLLM, Azure, etc.) ---
    # Set LLM_BASE_URL and LLM_MODEL in your .env file
    BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    MODEL = os.getenv("LLM_MODEL", "qwen3:30b-a3b")
    API_KEY = os.getenv("LLM_API_KEY", "ollama")

    # Number of queries to generate per chunk
    # Higher = more training data, but diminishing returns beyond 5
    # Typical range: 3-5 queries per chunk
    QUERIES_PER_CHUNK = 6

    # Limit number of chunks to process (for testing). None = all chunks
    MAX_CHUNKS = None

    # Max concurrent API requests
    # Local models: keep low (2-4) to avoid OOM
    # Cloud APIs: can go higher (10+)
    MAX_CONCURRENT = 2

    # -----------------------------------------------------------------------

    docs = list(DOCUMENTS.keys()) if DOC == "all" else [DOC]
    for doc_name in docs:
        config = DOCUMENTS[doc_name]
        asyncio.run(run(
            config=config,
            base_url=BASE_URL,
            model=MODEL,
            api_key=API_KEY,
            n_queries=QUERIES_PER_CHUNK,
            max_chunks=MAX_CHUNKS,
            max_concurrent=MAX_CONCURRENT,
        ))
