"""
Multi-hop synthetic query generation for embedding fine-tuning.

Generates queries that require information from 2-3 chunks to answer.
Each multi-hop query is "unrolled" into multiple (query, chunk) pairs,
teaching the embedding model that all referenced chunks are relevant.

Inspired by NVIDIA's NeMo Embed fine-tuning recipe:
https://huggingface.co/blog/nvidia/domain-specific-embedding-finetune

Usage:
  Set DOC and endpoint config in the __main__ section at the bottom.
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
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Document-specific configurations
# ---------------------------------------------------------------------------

@dataclass
class MultiHopConfig:
    """Configuration for multi-hop query generation per document."""
    name: str
    chunks_path: Path
    output_path: Path
    system_prompt: str
    user_prompt_template: str


SYSTEM_PROMPT_MULTIHOP_EU_AI_ACT = """\
Je bent een expert in de EU AI-verordening (EU AI Act) in het Nederlands.

Je taak: gegeven TWEE of DRIE gerelateerde tekstfragmenten uit de EU AI Act, \
genereer {n} zoekquery's in het Nederlands die informatie uit MEERDERE \
fragmenten combineren. De query moet niet beantwoord kunnen worden door \
slechts één fragment — het antwoord vereist het verbinden van informatie \
uit de gegeven fragmenten.

Regels:
- Schrijf ALLEEN in het Nederlands.
- Elke query moet informatie uit minstens 2 van de gegeven fragmenten \
combineren.
- Varieer tussen:
  * Vergelijkingsvragen ("Hoe verhoudt [concept A] zich tot [concept B]?")
  * Oorzaak-gevolgvragen ("Welke gevolgen heeft [regel X] voor [situatie Y]?")
  * Toepassingsvragen ("Hoe moeten de eisen uit [fragment 1] worden \
toegepast in combinatie met [fragment 2]?")
  * Scenariovragen die meerdere regels combineren ("Een bedrijf wil X doen, \
welke verplichtingen gelden op basis van zowel [onderwerp A] als \
[onderwerp B]?")
- Query's moeten realistisch zijn — alsof een jurist, beleidsmaker of \
compliance officer ze zou stellen.
- Verwijs NIET letterlijk naar artikelnummers, fragmentnummers of \
"fragment 1/2/3" in de query.
- Antwoord met een JSON-object met een "queries" veld.

Voorbeeld output:
{{"queries": [\
"Hoe beïnvloeden de transparantieverplichtingen de conformiteitsbeoordeling \
van hoog-risico AI-systemen?", \
"Een zorginstelling wil AI inzetten voor diagnoses én patiëntmonitoring, \
welke combinatie van verplichtingen geldt?"]}}
"""

SYSTEM_PROMPT_MULTIHOP_GDPR = """\
Je bent een expert in de Algemene Verordening Gegevensbescherming (AVG/GDPR) \
in het Nederlands.

Je taak: gegeven TWEE of DRIE gerelateerde tekstfragmenten uit de AVG, \
genereer {n} zoekquery's in het Nederlands die informatie uit MEERDERE \
fragmenten combineren. De query moet niet beantwoord kunnen worden door \
slechts één fragment — het antwoord vereist het verbinden van informatie \
uit de gegeven fragmenten.

Regels:
- Schrijf ALLEEN in het Nederlands.
- Elke query moet informatie uit minstens 2 van de gegeven fragmenten \
combineren.
- Varieer tussen:
  * Vergelijkingsvragen ("Hoe verhoudt [concept A] zich tot [concept B]?")
  * Oorzaak-gevolgvragen ("Welke gevolgen heeft [regel X] voor [situatie Y]?")
  * Toepassingsvragen ("Hoe werken [recht A] en [recht B] samen?")
  * Scenariovragen die meerdere regels combineren
- Query's moeten realistisch zijn — alsof een jurist, privacyfunctionaris \
(DPO), beleidsmaker of compliance officer ze zou stellen.
- Verwijs NIET letterlijk naar artikelnummers of fragmentnummers.
- Antwoord met een JSON-object met een "queries" veld.

Voorbeeld output:
{{"queries": [\
"Hoe verhoudt het recht op vergetelheid zich tot de bewaarplicht bij \
wettelijke verwerkingsgronden?", \
"Een webshop wil klantprofielen aanmaken én delen met een externe \
marketingpartner, welke rechten en plichten gelden voor beide partijen?"]}}
"""

SYSTEM_PROMPT_MULTIHOP_UAVG = """\
Je bent een expert in de Uitvoeringswet Algemene verordening \
gegevensbescherming (UAVG) — de Nederlandse implementatiewet van de AVG/GDPR.

Je taak: gegeven TWEE of DRIE gerelateerde tekstfragmenten uit de UAVG, \
genereer {n} zoekquery's in het Nederlands die informatie uit MEERDERE \
fragmenten combineren. De query moet niet beantwoord kunnen worden door \
slechts één fragment — het antwoord vereist het verbinden van informatie \
uit de gegeven fragmenten.

Regels:
- Schrijf ALLEEN in het Nederlands.
- Elke query moet informatie uit minstens 2 van de gegeven fragmenten \
combineren.
- Varieer tussen:
  * Vergelijkingsvragen ("Hoe verhoudt [uitzondering A] zich tot \
[uitzondering B]?")
  * Oorzaak-gevolgvragen ("Welke gevolgen heeft [regel X] voor [situatie Y]?")
  * Toepassingsvragen
  * Scenariovragen die meerdere regels combineren
- Query's moeten realistisch zijn — alsof een jurist, privacyfunctionaris \
(DPO), beleidsmaker of compliance officer ze zou stellen.
- Verwijs NIET letterlijk naar artikelnummers of fragmentnummers.
- Antwoord met een JSON-object met een "queries" veld.

Voorbeeld output:
{{"queries": [\
"Hoe verhoudt de uitzondering voor wetenschappelijk onderzoek zich tot de \
boetebevoegdheid van de AP bij verwerking van bijzondere persoonsgegevens?", \
"Een werkgever wil gezondheidsgegevens verwerken voor ziekteverzuimbeheer \
én delen met de bedrijfsarts, welke UAVG-regels zijn van toepassing?"]}}
"""

USER_PROMPT_MULTIHOP = """\
Tekstfragmenten (bron: {source_label}):

--- Fragment 1 ---
{chunk_1_text}
(Type: {chunk_1_section_type} | Locatie: {chunk_1_hierarchy_path})

--- Fragment 2 ---
{chunk_2_text}
(Type: {chunk_2_section_type} | Locatie: {chunk_2_hierarchy_path})
{chunk_3_block}
Genereer {n} zoekquery's die informatie uit meerdere fragmenten combineren.\
"""

CHUNK_3_TEMPLATE = """
--- Fragment 3 ---
{chunk_3_text}
(Type: {chunk_3_section_type} | Locatie: {chunk_3_hierarchy_path})
"""


DOCUMENTS: dict[str, MultiHopConfig] = {
    "eu_ai_act": MultiHopConfig(
        name="eu_ai_act",
        chunks_path=PROJECT_ROOT / "data" / "chunks" / "eu_ai_act" / "chunks_without_context.jsonl",
        output_path=PROJECT_ROOT / "data" / "synthetic" / "eu_ai_act_multihop_pairs.jsonl",
        system_prompt=SYSTEM_PROMPT_MULTIHOP_EU_AI_ACT,
        user_prompt_template=USER_PROMPT_MULTIHOP,
    ),
    "gdpr": MultiHopConfig(
        name="gdpr",
        chunks_path=PROJECT_ROOT / "data" / "chunks" / "gdpr" / "chunks_without_context.jsonl",
        output_path=PROJECT_ROOT / "data" / "synthetic" / "gdpr_multihop_pairs.jsonl",
        system_prompt=SYSTEM_PROMPT_MULTIHOP_GDPR,
        user_prompt_template=USER_PROMPT_MULTIHOP,
    ),
    "uavg": MultiHopConfig(
        name="uavg",
        chunks_path=PROJECT_ROOT / "data" / "chunks" / "uavg" / "chunks_without_context.jsonl",
        output_path=PROJECT_ROOT / "data" / "synthetic" / "uavg_multihop_pairs.jsonl",
        system_prompt=SYSTEM_PROMPT_MULTIHOP_UAVG,
        user_prompt_template=USER_PROMPT_MULTIHOP,
    ),
}

SOURCE_LABELS = {
    "eu_ai_act": "EU AI Act NL",
    "gdpr": "AVG/GDPR NL",
    "uavg": "UAVG NL",
}


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------

class GeneratedQueries(BaseModel):
    """Structured output schema for LLM multi-hop query generation."""
    queries: list[str] = Field(
        description="Lijst van multi-hop zoekquery's in het Nederlands"
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MultiHopPair:
    """One (query, chunk) pair from a multi-hop query.
    
    Multi-hop queries reference multiple chunks. After "unrolling",
    each (query, chunk) combination becomes a separate training pair,
    so the contrastive loss sees each positive independently.
    """
    anchor: str
    positive: str
    chunk_id: int
    section_type: str
    hierarchy_path: str
    is_multihop: bool = True
    hop_count: int = 2
    source_chunk_ids: list[int] | None = None


# ---------------------------------------------------------------------------
# Chunk grouping strategies
# ---------------------------------------------------------------------------

def group_chunks_by_relatedness(
    chunks: list[dict],
    hops: int = 2,
    max_groups: int | None = None,
    seed: int = 42,
) -> list[list[dict]]:
    """Group chunks into related pairs/triples for multi-hop query generation.
    
    Strategy:
    1. Same chapter, different articles → most related
    2. Same section type, adjacent chunk IDs → structurally related
    3. Random pairs within same section type → topically related
    
    Args:
        chunks: List of chunk dicts from JSONL
        hops: Number of chunks per group (2 or 3)
        max_groups: Maximum number of groups to generate
        seed: Random seed for reproducibility
    """
    rng = random.Random(seed)
    groups = []
    used_ids = set()
    
    # Index chunks by various attributes
    by_chapter: dict[str, list[dict]] = {}
    by_section_type: dict[str, list[dict]] = {}
    
    for chunk in chunks:
        chapter = chunk.get("chapter", "") or ""
        section_type = chunk.get("section_type", "") or ""
        
        if chapter:
            by_chapter.setdefault(chapter, []).append(chunk)
        if section_type:
            by_section_type.setdefault(section_type, []).append(chunk)
    
    # Strategy 1: Same chapter, different articles (strongest relatedness)
    for chapter, chapter_chunks in by_chapter.items():
        if len(chapter_chunks) < hops:
            continue
        # Group chunks that share a chapter but have different article numbers
        articles: dict[str | None, list[dict]] = {}
        for c in chapter_chunks:
            art = c.get("article_number")
            articles.setdefault(art, []).append(c)
        
        article_keys = [k for k in articles if k is not None]
        if len(article_keys) >= hops:
            # Sample one chunk per article
            for i in range(0, len(article_keys) - hops + 1, hops):
                group = []
                for j in range(hops):
                    candidates = articles[article_keys[i + j]]
                    picked = rng.choice(candidates)
                    group.append(picked)
                group_ids = tuple(c["chunk_id"] for c in group)
                if not any(cid in used_ids for cid in group_ids):
                    groups.append(group)
                    used_ids.update(group_ids)
    
    # Strategy 2: Adjacent chunks within same section type
    for section_type, st_chunks in by_section_type.items():
        sorted_chunks = sorted(st_chunks, key=lambda c: c["chunk_id"])
        for i in range(len(sorted_chunks) - hops + 1):
            group = sorted_chunks[i:i + hops]
            group_ids = tuple(c["chunk_id"] for c in group)
            if not any(cid in used_ids for cid in group_ids):
                groups.append(group)
                used_ids.update(group_ids)
    
    # Strategy 3: Random pairs within same section type (fill remaining)
    for section_type, st_chunks in by_section_type.items():
        available = [c for c in st_chunks if c["chunk_id"] not in used_ids]
        rng.shuffle(available)
        for i in range(0, len(available) - hops + 1, hops):
            group = available[i:i + hops]
            group_ids = tuple(c["chunk_id"] for c in group)
            groups.append(group)
            used_ids.update(group_ids)
    
    rng.shuffle(groups)
    
    if max_groups:
        groups = groups[:max_groups]
    
    return groups


# ---------------------------------------------------------------------------
# Query generation
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


async def generate_multihop_queries(
    client: AsyncOpenAI,
    model: str,
    chunk_group: list[dict],
    config: MultiHopConfig,
    semaphore: asyncio.Semaphore,
    n: int = 2,
) -> list[MultiHopPair]:
    """Generate multi-hop queries for a group of 2-3 related chunks."""
    
    # Build chunk_3 block (empty if only 2 chunks)
    chunk_3_block = ""
    if len(chunk_group) >= 3:
        chunk_3_block = CHUNK_3_TEMPLATE.format(
            chunk_3_text=chunk_group[2]["text"],
            chunk_3_section_type=chunk_group[2]["section_type"],
            chunk_3_hierarchy_path=chunk_group[2]["hierarchy_path"],
        )
    
    user_msg = config.user_prompt_template.format(
        source_label=SOURCE_LABELS[config.name],
        chunk_1_text=chunk_group[0]["text"],
        chunk_1_section_type=chunk_group[0]["section_type"],
        chunk_1_hierarchy_path=chunk_group[0]["hierarchy_path"],
        chunk_2_text=chunk_group[1]["text"],
        chunk_2_section_type=chunk_group[1]["section_type"],
        chunk_2_hierarchy_path=chunk_group[1]["hierarchy_path"],
        chunk_3_block=chunk_3_block,
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
            chunk_ids = [c["chunk_id"] for c in chunk_group]
            print(f"  [ERROR] chunks {chunk_ids}: {e}")
            return []
    
    # Unroll: one training pair per (query, chunk) combination
    source_chunk_ids = [c["chunk_id"] for c in chunk_group]
    pairs = []
    for query in queries:
        for chunk in chunk_group:
            pairs.append(MultiHopPair(
                anchor=query,
                positive=chunk["text"],
                chunk_id=chunk["chunk_id"],
                section_type=chunk["section_type"],
                hierarchy_path=chunk["hierarchy_path"],
                is_multihop=True,
                hop_count=len(chunk_group),
                source_chunk_ids=source_chunk_ids,
            ))
    return pairs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(
    config: MultiHopConfig,
    base_url: str,
    model: str,
    api_key: str = "ollama",
    n_queries: int = 2,
    hops: int = 2,
    max_groups: int | None = None,
    max_concurrent: int = 10,
):
    """Generate multi-hop queries for chunk groups and save to JSONL."""
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load chunks
    with open(config.chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    
    # Group chunks by relatedness
    groups = group_chunks_by_relatedness(chunks, hops=hops, max_groups=max_groups)
    
    print(f"[{config.name}] {len(chunks)} chunks → {len(groups)} groups "
          f"({hops}-hop, {n_queries} queries each)")
    print(f"  Expected pairs: ~{len(groups) * n_queries * hops} "
          f"({len(groups)} groups × {n_queries} queries × {hops} unrolled)")
    print(f"  Model: {model}")
    
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = [
        generate_multihop_queries(
            client, model, group, config, semaphore, n_queries,
        )
        for group in groups
    ]
    
    all_pairs: list[MultiHopPair] = []
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
    unique_queries = len(set(p.anchor for p in all_pairs))
    print(f"\nDone. {len(all_pairs)} pairs ({unique_queries} unique queries) "
          f"saved to {config.output_path}")
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
    DOC = "eu_ai_act"
    
    # --- LLM endpoint (OpenAI-compatible: Ollama, vLLM, Azure, etc.) ---
    BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    MODEL = os.getenv("LLM_MODEL", "qwen3:30b-a3b")
    API_KEY = os.getenv("LLM_API_KEY", "ollama")
    
    # Number of multi-hop queries to generate per chunk group
    # Lower than single-hop since each query produces N unrolled pairs
    QUERIES_PER_GROUP = 2
    
    # Number of chunks per group (2 = 2-hop, 3 = 3-hop)
    HOPS = 2
    
    # Max number of chunk groups to process (None = all)
    MAX_GROUPS = None
    
    # Max concurrent API requests
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
            n_queries=QUERIES_PER_GROUP,
            hops=HOPS,
            max_groups=MAX_GROUPS,
            max_concurrent=MAX_CONCURRENT,
        ))
