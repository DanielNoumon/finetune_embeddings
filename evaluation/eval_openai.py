"""
Evaluate OpenAI text-embedding-3-large on the EU AI Act NL retrieval task.

Compares a proprietary SOTA embedding model against our fine-tuned models
using the same eval set (340 queries, 85 corpus chunks).

text-embedding-3-large natively supports Matryoshka dimensions via the
`dimensions` parameter, so we can compare at the same dims as our models.

Setup:
  1. Deploy text-embedding-3-large on Azure OpenAI, OR
  2. Use an OpenAI API key directly

  Add to your .env:
    # Option A: Azure OpenAI
    EMBEDDINGS_AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    EMBEDDINGS_AZURE_OPENAI_KEY=your-key
    EMBEDDINGS_DEPLOYMENT_NAME=text-embedding-3-large
    EMBEDDINGS_API_VERSION=2024-06-01

    # Option B: OpenAI directly
    OPENAI_API_KEY=sk-...

Run:
  python evaluation/eval_openai.py
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "processed" / "eval"

# Matryoshka dims to evaluate — text-embedding-3-large max is 3072
MATRYOSHKA_DIMS = [3072, 1024, 768, 512, 256, 128]

# Batch size for API calls (Azure limit: 16 inputs per request for this model)
EMBED_BATCH_SIZE = 16

# Reference scores from our fine-tuned models (NDCG@10 at dim=1024)
REFERENCE_SCORES = {
    "e5-large Stage 2 (batch 8)": 0.9492,
    "Qwen3-0.6B Stage 2": 0.9467,
    "Qwen3-4B LoRA Stage 2": 0.9658,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_eval_data(eval_dir: Path):
    with open(eval_dir / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(eval_dir / "corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(eval_dir / "relevant_docs.json", "r", encoding="utf-8") as f:
        relevant_docs_raw = json.load(f)
    relevant_docs = {qid: set(cids) for qid, cids in relevant_docs_raw.items()}
    return queries, corpus, relevant_docs


def get_client():
    """Create OpenAI or Azure OpenAI client based on available env vars."""
    load_dotenv(PROJECT_ROOT / ".env")

    azure_endpoint = os.getenv("EMBEDDINGS_AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        print("Using Azure OpenAI embeddings")
        return (
            AzureOpenAI(
                api_key=os.getenv("EMBEDDINGS_AZURE_OPENAI_KEY"),
                azure_endpoint=azure_endpoint,
                api_version=os.getenv(
                    "EMBEDDINGS_API_VERSION", "2024-06-01"
                ),
            ),
            os.getenv(
                "EMBEDDINGS_DEPLOYMENT_NAME", "text-embedding-3-large"
            ),
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("Using OpenAI directly")
        return OpenAI(api_key=openai_key), "text-embedding-3-large"

    raise RuntimeError(
        "No embedding API credentials found. Add either:\n"
        "  AZURE_OPENAI_EMBEDDING_ENDPOINT + AZURE_OPENAI_API_KEY  (Azure)\n"
        "  OPENAI_API_KEY  (OpenAI direct)\n"
        "to your .env file."
    )


def embed_texts(client, model, texts: list[str], dimensions: int) -> np.ndarray:
    """Embed a list of texts in batches, returning a (N, dimensions) array."""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(
            model=model,
            input=batch,
            dimensions=dimensions,
        )
        batch_embs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embs)

        if i > 0 and i % (EMBED_BATCH_SIZE * 10) == 0:
            print(f"    Embedded {i + len(batch)}/{len(texts)}")

    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between all pairs (a[i], b[j])."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def compute_metrics(
    sim_matrix: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    relevant_docs: dict[str, set[str]],
    k_values: list[int] = [1, 3, 5, 10],
) -> dict[str, float]:
    """Compute IR metrics from a similarity matrix."""
    metrics = {}
    ndcg_scores = []
    mrr_scores = []
    recall_at_k = {k: [] for k in k_values}
    accuracy_at_k = {k: [] for k in k_values}
    precision_at_k = {k: [] for k in k_values}
    map_scores = []

    for i, qid in enumerate(query_ids):
        rel = relevant_docs.get(qid, set())
        if not rel:
            continue

        scores = sim_matrix[i]
        ranked_indices = np.argsort(-scores)
        ranked_ids = [corpus_ids[idx] for idx in ranked_indices]

        # NDCG@10
        dcg = 0.0
        for rank in range(min(10, len(ranked_ids))):
            if ranked_ids[rank] in rel:
                dcg += 1.0 / np.log2(rank + 2)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(10, len(rel))))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        # MRR@10
        mrr = 0.0
        for rank in range(min(10, len(ranked_ids))):
            if ranked_ids[rank] in rel:
                mrr = 1.0 / (rank + 1)
                break
        mrr_scores.append(mrr)

        # MAP@100
        ap = 0.0
        n_rel_found = 0
        for rank in range(min(100, len(ranked_ids))):
            if ranked_ids[rank] in rel:
                n_rel_found += 1
                ap += n_rel_found / (rank + 1)
        map_scores.append(ap / len(rel) if len(rel) > 0 else 0.0)

        # Recall, Accuracy, Precision @k
        for k in k_values:
            top_k = set(ranked_ids[:k])
            hits = len(top_k & rel)
            recall_at_k[k].append(hits / len(rel))
            accuracy_at_k[k].append(1.0 if hits > 0 else 0.0)
            precision_at_k[k].append(hits / k)

    metrics["ndcg@10"] = float(np.mean(ndcg_scores))
    metrics["mrr@10"] = float(np.mean(mrr_scores))
    metrics["map@100"] = float(np.mean(map_scores))
    for k in k_values:
        metrics[f"recall@{k}"] = float(np.mean(recall_at_k[k]))
        metrics[f"accuracy@{k}"] = float(np.mean(accuracy_at_k[k]))
        metrics[f"precision@{k}"] = float(np.mean(precision_at_k[k]))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    client, model_name = get_client()
    print(f"Model: {model_name}")

    queries, corpus, relevant_docs = load_eval_data(EVAL_DIR)
    print(f"Eval set: {len(queries)} queries, {len(corpus)} corpus chunks")

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid] for cid in corpus_ids]

    all_results = {}

    for dim in MATRYOSHKA_DIMS:
        print(f"\n--- Evaluating at dim={dim} ---")

        t0 = time.time()
        print(f"  Embedding {len(query_texts)} queries...")
        query_embs = embed_texts(client, model_name, query_texts, dim)

        print(f"  Embedding {len(corpus_texts)} corpus chunks...")
        corpus_embs = embed_texts(client, model_name, corpus_texts, dim)
        elapsed = time.time() - t0
        print(f"  Embeddings done in {elapsed:.1f}s")

        sim_matrix = cosine_similarity_matrix(query_embs, corpus_embs)
        metrics = compute_metrics(
            sim_matrix, query_ids, corpus_ids, relevant_docs
        )
        all_results[dim] = metrics
        print(f"  NDCG@10: {metrics['ndcg@10']:.4f}")

    # -------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------
    print(f"\n{'='*62}")
    print("text-embedding-3-large — NDCG@10 across dimensions")
    print(f"{'='*62}")
    print(f"{'Dim':>6}  {'NDCG@10':>10}  {'vs Qwen3-4B FT':>16}")
    print(f"{'-'*6}  {'-'*10}  {'-'*16}")
    qwen4b_ref = REFERENCE_SCORES["Qwen3-4B LoRA Stage 2"]
    for dim in MATRYOSHKA_DIMS:
        score = all_results[dim]["ndcg@10"]
        # Compare at dim=1024 since that's where we have reference scores
        if dim == 1024:
            delta = score - qwen4b_ref
            print(f"{dim:>6}  {score:>10.4f}  {delta:>+16.4f}")
        else:
            print(f"{dim:>6}  {score:>10.4f}  {'—':>16}")
    print(f"{'='*62}")

    # Comparison table at dim=1024
    dim_compare = 1024
    if dim_compare in all_results:
        oai_score = all_results[dim_compare]["ndcg@10"]
        print(f"\n{'='*52}")
        print(f"Comparison at dim={dim_compare} (NDCG@10)")
        print(f"{'='*52}")
        print(f"{'Model':<36}  {'NDCG@10':>10}")
        print(f"{'-'*36}  {'-'*10}")
        print(f"{'text-embedding-3-large (zero-shot)':<36}  {oai_score:>10.4f}")
        for name, score in REFERENCE_SCORES.items():
            print(f"{name:<36}  {score:>10.4f}")
        print(f"{'='*52}")

    # Full metrics at max dim
    max_dim = MATRYOSHKA_DIMS[0]
    print(f"\n{'='*42}")
    print(f"Full Metrics at dim={max_dim}")
    print(f"{'='*42}")
    print(f"{'Metric':<16}  {'Score':>10}")
    print(f"{'-'*16}  {'-'*10}")
    detail_metrics = [
        ("NDCG@10", "ndcg@10"),
        ("MRR@10", "mrr@10"),
        ("MAP@100", "map@100"),
        ("Accuracy@1", "accuracy@1"),
        ("Accuracy@3", "accuracy@3"),
        ("Accuracy@5", "accuracy@5"),
        ("Accuracy@10", "accuracy@10"),
        ("Precision@1", "precision@1"),
        ("Precision@3", "precision@3"),
        ("Precision@10", "precision@10"),
        ("Recall@1", "recall@1"),
        ("Recall@3", "recall@3"),
        ("Recall@5", "recall@5"),
        ("Recall@10", "recall@10"),
    ]
    for label, key in detail_metrics:
        score = all_results[max_dim].get(key, 0)
        print(f"{label:<16}  {score:>10.4f}")
    print(f"{'='*42}")
