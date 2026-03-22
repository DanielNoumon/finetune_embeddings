"""Test different instruction phrasings on the base instruct model.

Evaluates NDCG@10 at dim=1024 for various instruction prefixes to find
the optimal phrasing. Also tests the non-instruct "query: " / "passage: "
baseline for comparison.

Run on the remote GPU:
  python finetuning/stage_1_mnrl/diagnose_instructions.py
"""

import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "processed" / "eval"
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

INSTRUCTIONS = {
    # Our current instruction
    "current": (
        "Instruct: Given a question about EU AI regulation, "
        "retrieve the most relevant passage\nQuery: "
    ),
    # More standard patterns (from official eval code)
    "generic_retrieve": (
        "Instruct: Given a question, retrieve relevant passages "
        "that answer the question\nQuery: "
    ),
    "web_search": (
        "Instruct: Given a web search query, retrieve relevant "
        "passages that answer the query\nQuery: "
    ),
    # Mention Dutch explicitly
    "dutch_explicit": (
        "Instruct: Given a question in Dutch about EU AI regulation, "
        "retrieve the most relevant passage in Dutch\nQuery: "
    ),
    # Shorter / simpler
    "simple": (
        "Instruct: Retrieve the most relevant passage for "
        "this query\nQuery: "
    ),
    # Domain-specific but standard pattern
    "legal_retrieve": (
        "Instruct: Given a question about AI legislation, "
        "retrieve relevant legal passages that answer the "
        "question\nQuery: "
    ),
    # Non-instruct baseline (query: / passage:)
    "non_instruct_baseline": "query: ",
    # No prefix at all
    "no_prefix": "",
}

# Corpus prompt: "" for instruct variants, "passage: " for non-instruct
CORPUS_PROMPTS = {
    "non_instruct_baseline": "passage: ",
}


def load_eval_data(eval_dir):
    with open(eval_dir / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(eval_dir / "corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(eval_dir / "relevant_docs.json", "r", encoding="utf-8") as f:
        relevant_docs_raw = json.load(f)
    relevant_docs = {
        qid: set(cids) for qid, cids in relevant_docs_raw.items()
    }
    return queries, corpus, relevant_docs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {name} ({mem:.1f} GB)")

    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs={"attn_implementation": "eager"},
    )

    print(f"Loading eval data from: {EVAL_DIR}")
    queries, corpus, relevant_docs = load_eval_data(EVAL_DIR)
    print(f"  Queries: {len(queries)}, Corpus: {len(corpus)}")

    print(f"\n{'='*62}")
    print(f"{'Instruction':>25}  {'NDCG@10':>8}  {'MRR@10':>8}  {'Acc@1':>8}")
    print(f"{'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}")

    results = {}
    for name, query_prompt in INSTRUCTIONS.items():
        corpus_prompt = CORPUS_PROMPTS.get(name, "")

        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"test-{name}",
            truncate_dim=1024,
            query_prompt=query_prompt,
            corpus_prompt=corpus_prompt,
            show_progress_bar=False,
        )

        res = evaluator(model)
        ndcg = res[f"test-{name}_cosine_ndcg@10"]
        mrr = res[f"test-{name}_cosine_mrr@10"]
        acc1 = res[f"test-{name}_cosine_accuracy@1"]
        results[name] = (ndcg, mrr, acc1)

        print(f"{name:>25}  {ndcg:>8.4f}  {mrr:>8.4f}  {acc1:>8.4f}")

    print(f"{'='*62}")

    best = max(results, key=lambda k: results[k][0])
    print(f"\nBest instruction: '{best}'")
    print(f"  NDCG@10: {results[best][0]:.4f}")
    print(f"  Prompt: {INSTRUCTIONS[best]!r}")
