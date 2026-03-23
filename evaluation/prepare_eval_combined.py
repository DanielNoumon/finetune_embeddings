"""
Prepare evaluation data from hf_dataset.parquet for the combined
EU regulations retrieval benchmark.

Creates three eval sets under data/processed/eval_combined/:

  combined/   -- all 5,395 queries against all 912 corpus chunks
  eu_ai_act/  -- 3,166 EU AI Act queries against 535-chunk corpus
  gdpr/       -- 2,229 GDPR queries against 377-chunk GDPR corpus

Output files per set (InformationRetrievalEvaluator-compatible):
  queries.json        {"qid": "query_text", ...}
  corpus.json         {"cid": "chunk_text", ...}
  relevant_docs.json  {"qid": ["cid"], ...}

Corpus IDs are prefixed to avoid overlap between documents:
  EU AI Act (NL) -> "eu_ai_act_{chunk_id}"
  AVG/GDPR (NL)  -> "gdpr_{chunk_id}"

Run:
  python evaluation/prepare_eval_combined.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "hf_dataset.parquet"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "eval_combined"

DOC_PREFIX: dict[str, str] = {
    "EU AI Act (NL)": "eu_ai_act",
    "AVG/GDPR (NL)": "gdpr",
}


def corpus_id(doc_name: str, chunk_id: int) -> str:
    """Globally unique corpus key: '{doc_prefix}_{chunk_id}'."""
    return f"{DOC_PREFIX[doc_name]}_{chunk_id}"


def build_eval_set(
    df: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    """Build queries, corpus, relevant_docs from a DataFrame slice."""
    corpus: dict[str, str] = {}
    queries: dict[str, str] = {}
    relevant_docs: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        cid = corpus_id(row["document_name"], int(row["chunk_id"]))
        corpus.setdefault(cid, row["chunk"])

        qid = str(row["question_id"])
        queries[qid] = row["query"]
        relevant_docs[qid] = [cid]

    return queries, corpus, relevant_docs


def save_eval_set(
    name: str,
    queries: dict,
    corpus: dict,
    relevant_docs: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fname, obj in [
        ("queries.json", queries),
        ("corpus.json", corpus),
        ("relevant_docs.json", relevant_docs),
    ]:
        with open(output_dir / fname, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    print(
        f"  {name:<12}  {len(queries):>5} queries  "
        f"{len(corpus):>4} corpus chunks"
    )


if __name__ == "__main__":
    print(f"Loading {PARQUET_PATH.name} ...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} rows\n")

    splits = [
        ("combined", df),
        ("eu_ai_act", df[df["document_name"] == "EU AI Act (NL)"]),
        ("gdpr", df[df["document_name"] == "AVG/GDPR (NL)"]),
    ]

    print(f"Saving eval sets to {OUTPUT_ROOT}/")
    for name, subset in splits:
        queries, corpus, relevant_docs = build_eval_set(subset.copy())
        save_eval_set(name, queries, corpus, relevant_docs, OUTPUT_ROOT / name)

    print(
        "\nDone. Run evaluation/eval_combined.py to evaluate models."
    )
