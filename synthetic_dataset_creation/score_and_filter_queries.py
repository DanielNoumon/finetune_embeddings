"""
Quality scoring and filtering for synthetic query-chunk pairs.

Uses an LLM judge to score each (query, chunk) pair on multiple
quality dimensions, then filters out low-quality pairs before training.

Inspired by NVIDIA's NeMo Embed fine-tuning recipe which scores
each synthetic QA pair on relevance, accuracy, context support,
and clarity.

Usage:
  Set INPUT_PATH, endpoint config, and SCORE_THRESHOLD in __main__.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
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
# Pydantic output schema for quality scores
# ---------------------------------------------------------------------------

class QualityScore(BaseModel):
    """Quality assessment of a (query, chunk) pair."""
    relevance: int = Field(
        ge=1, le=10,
        description="How relevant is the chunk to answering the query? (1=irrelevant, 10=perfect answer)"
    )
    accuracy: int = Field(
        ge=1, le=10,
        description="Does the chunk actually contain the information needed to answer? (1=no, 10=fully)"
    )
    clarity: int = Field(
        ge=1, le=10,
        description="Is the query clear, unambiguous, and well-formed? (1=unclear, 10=crystal clear)"
    )
    specificity: int = Field(
        ge=1, le=10,
        description="Is the query specific enough that this chunk is a better answer than others? (1=too generic, 10=highly specific)"
    )
    overall: float = Field(
        ge=1.0, le=10.0,
        description="Overall quality score (weighted average)"
    )


# ---------------------------------------------------------------------------
# Scoring prompt
# ---------------------------------------------------------------------------

SCORING_SYSTEM_PROMPT = """\
Je bent een strenge kwaliteitsbeoordelaar voor zoekquery-tekstfragment paren \
die worden gebruikt om een embedding model te trainen voor informatie-opvraging.

Je taak: beoordeel of de gegeven query realistisch aansluit bij het \
tekstfragment. Score elk aspect op een schaal van 1-10.

Scoringscriteria:
- **relevance** (1-10): Hoe relevant is het fragment als antwoord op de query?
  * 1-3: Fragment gaat over een ander onderwerp
  * 4-6: Fragment is gerelateerd maar beantwoordt de query niet direct
  * 7-8: Fragment beantwoordt de query grotendeels
  * 9-10: Fragment is het perfecte antwoord op de query

- **accuracy** (1-10): Bevat het fragment daadwerkelijk de informatie die \
nodig is om de query te beantwoorden?
  * 1-3: Informatie ontbreekt of is onjuist
  * 4-6: Gedeeltelijk aanwezig
  * 7-10: Volledig aanwezig

- **clarity** (1-10): Is de query helder, eenduidig en goed geformuleerd?
  * 1-3: Onbegrijpelijk of zeer dubbelzinnig
  * 4-6: Enigszins onduidelijk
  * 7-10: Helder en eenduidig

- **specificity** (1-10): Is de query specifiek genoeg dat dit fragment \
een beter antwoord is dan willekeurige andere fragmenten?
  * 1-3: Te generiek — tientallen fragmenten zouden passen
  * 4-6: Redelijk specifiek
  * 7-10: Zeer specifiek — dit fragment is duidelijk het beste antwoord

- **overall** (1.0-10.0): Gewogen gemiddelde. Gebruik deze formule:
  overall = 0.35 * relevance + 0.25 * accuracy + 0.20 * clarity + 0.20 * specificity

Antwoord ALLEEN met een JSON-object. Geen uitleg.\
"""

SCORING_USER_PROMPT = """\
Query: {query}

Tekstfragment:
---
{chunk_text}
---

Beoordeel dit paar.\
"""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Extract a JSON object from LLM output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text.strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

async def score_pair(
    client: AsyncOpenAI,
    model: str,
    pair: dict,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Score a single (query, chunk) pair using the LLM judge.
    
    Returns the pair dict with quality scores added, or None on error.
    """
    user_msg = SCORING_USER_PROMPT.format(
        query=pair["anchor"],
        chunk_text=pair["positive"],
    )
    
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content or "{}"
            content = _extract_json(content)
            scores = QualityScore.model_validate_json(content)
            
            pair["quality_scores"] = {
                "relevance": scores.relevance,
                "accuracy": scores.accuracy,
                "clarity": scores.clarity,
                "specificity": scores.specificity,
                "overall": scores.overall,
            }
            return pair
            
        except Exception as e:
            print(f"  [ERROR] chunk {pair.get('chunk_id', '?')}: {e}")
            return None


async def score_all_pairs(
    pairs: list[dict],
    base_url: str,
    model: str,
    api_key: str = "ollama",
    max_concurrent: int = 10,
) -> list[dict]:
    """Score all pairs and return those with scores (skipping errors)."""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = [
        score_pair(client, model, pair, semaphore)
        for pair in pairs
    ]
    
    scored = []
    start = time.time()
    completed = 0
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            scored.append(result)
        completed += 1
        if completed % 100 == 0 or completed == len(tasks):
            elapsed = time.time() - start
            print(f"  {completed}/{len(tasks)} scored "
                  f"({len(scored)} successful, {elapsed:.0f}s)")
    
    return scored


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_by_threshold(
    pairs: list[dict],
    threshold: float = 7.0,
    score_key: str = "overall",
) -> tuple[list[dict], list[dict]]:
    """Split pairs into kept and rejected based on quality threshold.
    
    Args:
        pairs: List of scored pair dicts
        threshold: Minimum overall score to keep (inclusive)
        score_key: Which score to filter on
    
    Returns:
        (kept, rejected) tuple
    """
    kept = []
    rejected = []
    
    for pair in pairs:
        scores = pair.get("quality_scores", {})
        score = scores.get(score_key, 0)
        if score >= threshold:
            kept.append(pair)
        else:
            rejected.append(pair)
    
    return kept, rejected


def print_score_stats(pairs: list[dict]):
    """Print statistics about quality scores."""
    scores = [p["quality_scores"] for p in pairs if "quality_scores" in p]
    if not scores:
        print("No scored pairs.")
        return
    
    print(f"\n=== Quality Score Statistics ({len(scores)} pairs) ===")
    print()
    
    for key in ["relevance", "accuracy", "clarity", "specificity", "overall"]:
        values = [s[key] for s in scores]
        avg = sum(values) / len(values)
        minimum = min(values)
        maximum = max(values)
        
        # Distribution buckets
        low = sum(1 for v in values if v < 5)
        mid = sum(1 for v in values if 5 <= v < 7)
        high = sum(1 for v in values if v >= 7)
        
        print(f"  {key:12s}: avg={avg:.2f}  min={minimum}  max={maximum}  "
              f"[<5: {low}, 5-7: {mid}, ≥7: {high}]")
    
    # Show threshold impact
    print(f"\n  Pairs surviving at various thresholds:")
    for t in [5.0, 6.0, 6.5, 7.0, 7.5, 8.0]:
        surviving = sum(1 for s in scores if s["overall"] >= t)
        pct = surviving / len(scores) * 100
        print(f"    threshold={t:.1f}: {surviving}/{len(scores)} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    base_url: str,
    model: str,
    api_key: str = "ollama",
    threshold: float = 7.0,
    max_concurrent: int = 10,
    max_pairs: int | None = None,
):
    """Score and filter query-chunk pairs."""
    
    # Load pairs
    with open(input_path, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    
    if max_pairs:
        pairs = pairs[:max_pairs]
    
    print(f"Loaded {len(pairs)} pairs from {input_path.name}")
    print(f"Scoring with model: {model}")
    print(f"Threshold: {threshold}")
    
    # Score all pairs
    scored = await score_all_pairs(
        pairs, base_url, model, api_key, max_concurrent
    )
    
    # Print stats
    print_score_stats(scored)
    
    # Filter
    kept, rejected = filter_by_threshold(scored, threshold=threshold)
    
    print("\n=== Filtering Results ===")
    print(f"  Kept:     {len(kept)} ({len(kept)/len(scored)*100:.1f}%)")
    print(f"  Rejected: {len(rejected)} ({len(rejected)/len(scored)*100:.1f}%)")
    
    # Backup existing files
    for path in [output_path, rejected_path]:
        if path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = path.parent / f"{path.stem}_backup_{timestamp}{path.suffix}"
            shutil.copy2(path, backup)
    
    # Save kept pairs (without quality_scores to keep format clean)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in kept:
            # Remove quality_scores from training data output
            clean = {k: v for k, v in pair.items() if k != "quality_scores"}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    
    # Save rejected pairs (with scores for analysis)
    with open(rejected_path, "w", encoding="utf-8") as f:
        for pair in rejected:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    # Save full scored dataset for analysis
    scored_path = output_path.parent / f"{output_path.stem}_scored{output_path.suffix}"
    with open(scored_path, "w", encoding="utf-8") as f:
        for pair in scored:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    print(f"\n  Kept pairs saved to: {output_path}")
    print(f"  Rejected pairs saved to: {rejected_path}")
    print(f"  Full scored dataset saved to: {scored_path}")
    
    return kept, rejected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    # -----------------------------------------------------------------------
    # CONFIG: Adjust these parameters for your use case
    # -----------------------------------------------------------------------
    
    # Input: the JSONL file with (anchor, positive, ...) pairs to score
    DOC = "eu_ai_act"
    INPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / f"{DOC}_query_pairs.jsonl"
    
    # Output paths
    OUTPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / f"{DOC}_query_pairs_filtered.jsonl"
    REJECTED_PATH = PROJECT_ROOT / "data" / "synthetic" / f"{DOC}_query_pairs_rejected.jsonl"
    
    # --- LLM endpoint (OpenAI-compatible: Ollama, vLLM, Azure, etc.) ---
    BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    MODEL = os.getenv("LLM_MODEL", "qwen3:30b-a3b")
    API_KEY = os.getenv("LLM_API_KEY", "ollama")
    
    # Minimum overall quality score to keep a pair (1.0-10.0)
    # 7.0 is a good starting point — keeps ~80-90% of pairs typically
    # Lower (6.0) = keep more data, accept some noise
    # Higher (8.0) = stricter, cleaner data, but less of it
    SCORE_THRESHOLD = 7.0
    
    # Limit pairs to score (for testing). None = all
    MAX_PAIRS = None
    
    # Max concurrent API requests
    MAX_CONCURRENT = 4
    
    # -----------------------------------------------------------------------
    
    asyncio.run(run(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        rejected_path=REJECTED_PATH,
        base_url=BASE_URL,
        model=MODEL,
        api_key=API_KEY,
        threshold=SCORE_THRESHOLD,
        max_concurrent=MAX_CONCURRENT,
        max_pairs=MAX_PAIRS,
    ))
