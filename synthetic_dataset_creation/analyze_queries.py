"""Analyze generated query-chunk pairs for quality and statistics."""
import json
from pathlib import Path
from collections import Counter, defaultdict


def analyze_query_pairs(jsonl_path: str):
    """Analyze query pair dataset."""
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"=== Query Pair Analysis ===\n")
    print(f"Total pairs: {len(pairs)}")
    print(f"Expected (573 chunks × 4 queries): {573 * 4}")
    print(f"Success rate: {len(pairs) / (573 * 4) * 100:.1f}%\n")

    # Query length distribution
    query_lengths = [len(p["anchor"]) for p in pairs]
    print(f"Query length (chars):")
    print(f"  Min: {min(query_lengths)}")
    print(f"  Max: {max(query_lengths)}")
    print(f"  Mean: {sum(query_lengths) / len(query_lengths):.1f}")
    print(f"  Median: {sorted(query_lengths)[len(query_lengths) // 2]}\n")

    # Chunk length distribution
    chunk_lengths = [len(p["positive"]) for p in pairs]
    print(f"Chunk length (chars):")
    print(f"  Min: {min(chunk_lengths)}")
    print(f"  Max: {max(chunk_lengths)}")
    print(f"  Mean: {sum(chunk_lengths) / len(chunk_lengths):.1f}\n")

    # Section type distribution
    section_types = Counter(p["section_type"] for p in pairs)
    print(f"Pairs by section type:")
    for stype, count in section_types.most_common():
        print(f"  {stype}: {count} ({count / len(pairs) * 100:.1f}%)")
    print()

    # Query type heuristics
    question_words = {
        "wat": "definitional",
        "welke": "factual",
        "hoe": "procedural",
        "wanneer": "temporal",
        "waarom": "causal",
        "wie": "actor",
        "een bedrijf": "scenario",
        "een organisatie": "scenario",
        "moet": "obligation",
        "mag": "permission",
    }

    query_types = defaultdict(int)
    for pair in pairs:
        query_lower = pair["anchor"].lower()
        for word, qtype in question_words.items():
            if word in query_lower:
                query_types[qtype] += 1
                break

    print(f"Query type distribution (heuristic):")
    for qtype, count in sorted(
        query_types.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {qtype}: {count} ({count / len(pairs) * 100:.1f}%)")
    print()

    # Sample queries by section type
    print(f"=== Sample Queries ===\n")
    for stype in ["overweging", "artikel", "bijlage"]:
        samples = [p for p in pairs if p["section_type"] == stype][:2]
        if samples:
            print(f"{stype.upper()}:")
            for i, s in enumerate(samples, 1):
                print(f"  {i}. {s['anchor']}")
            print()

    # Check for duplicates
    unique_queries = len(set(p["anchor"] for p in pairs))
    print(f"Unique queries: {unique_queries} / {len(pairs)}")
    print(
        f"Duplicate rate: {(1 - unique_queries / len(pairs)) * 100:.2f}%\n"
    )

    # Check for article number references (should be minimal)
    article_refs = sum(
        1
        for p in pairs
        if any(
            term in p["anchor"].lower()
            for term in ["artikel", "art.", "art ", "bijlage"]
        )
    )
    print(f"Queries with article references: {article_refs}")
    print(
        f"  (Should be low - users don't know article numbers): "
        f"{article_refs / len(pairs) * 100:.1f}%\n"
    )

    # Quality flags
    print(f"=== Quality Checks ===\n")

    # Very short queries (likely low quality)
    short_queries = [p for p in pairs if len(p["anchor"]) < 30]
    print(f"Very short queries (<30 chars): {len(short_queries)}")
    if short_queries:
        print(f"  Examples:")
        for q in short_queries[:3]:
            print(f"    - {q['anchor']}")
    print()

    # Very long queries (might be too specific)
    long_queries = [p for p in pairs if len(p["anchor"]) > 200]
    print(f"Very long queries (>200 chars): {len(long_queries)}")
    if long_queries:
        print(f"  Examples:")
        for q in long_queries[:2]:
            print(f"    - {q['anchor'][:150]}...")
    print()


if __name__ == "__main__":
    jsonl_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "synthetic"
        / "query_pairs.jsonl"
    )
    analyze_query_pairs(str(jsonl_path))
