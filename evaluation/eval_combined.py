"""
Unified evaluation of embedding models on the combined EU regulations dataset.

Evaluates three retrieval splits:
  combined   -- all 5,395 queries vs 912-chunk corpus (hardest, cross-doc)
  eu_ai_act  -- 3,166 EU AI Act queries vs 535-chunk EU AI Act corpus
  gdpr       -- 2,229 GDPR queries vs 377-chunk GDPR corpus

Supported model types:
  st      -- any SentenceTransformer model (local fine-tuned or HF Hub)
  openai  -- text-embedding-3-large via Azure OpenAI or OpenAI direct

Local fine-tuned models are silently skipped if the path does not exist.
OpenAI is skipped if no API credentials are found in .env.

Prerequisites:
  Run prepare_eval_combined.py once to build the eval JSON files, then
  copy data/processed/eval_combined/ to the GPU machine if needed.

Run:
  python evaluation/eval_combined.py
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = PROJECT_ROOT / "data" / "processed" / "eval_combined"

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Model configs — edit this section to add/remove models
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    model_path: str
    model_type: Literal["st", "openai"]
    dims: list[int]
    query_prompt: str = ""
    corpus_prompt: str = ""
    batch_size: int = 64
    model_kwargs: dict = field(default_factory=dict)
    tokenizer_kwargs: dict = field(default_factory=dict)


_EU_PROMPT = (
    "Instruct: Given a question about EU AI regulation, "
    "retrieve the most relevant passage\nQuery:"
)
_REG_PROMPT = (
    "Instruct: Given a question about EU regulations, "
    "retrieve the most relevant passage\nQuery:"
)

MODEL_CONFIGS: list[ModelConfig] = [
    # --- Zero-shot baselines ---
    ModelConfig(
        name="multilingual-e5-large (zero-shot)",
        model_path="intfloat/multilingual-e5-large-instruct",
        model_type="st",
        dims=[1024],
        query_prompt=_REG_PROMPT + " ",
        model_kwargs={
            "torch_dtype": torch.float32,
            "attn_implementation": "eager",
        },
    ),
    ModelConfig(
        name="Qwen3-0.6B (zero-shot)",
        model_path="Qwen/Qwen3-Embedding-0.6B",
        model_type="st",
        dims=[1024, 512, 256, 128],
        query_prompt=_REG_PROMPT,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
        tokenizer_kwargs={"padding_side": "left"},
    ),
    # --- Fine-tuned on EU AI Act (from HF Hub) ---
    ModelConfig(
        name="multilingual-e5-large (EU AI Act FT)",
        model_path="danielnoumon/multilingual-e5-large-ai-act-nl",
        model_type="st",
        dims=[1024],
        query_prompt=_EU_PROMPT + " ",
        model_kwargs={
            "torch_dtype": torch.float32,
            "attn_implementation": "eager",
        },
    ),
    ModelConfig(
        name="Qwen3-0.6B (EU AI Act FT)",
        model_path="danielnoumon/qwen3-embedding-0.6b-ai-act-nl",
        model_type="st",
        dims=[1024, 512, 256, 128],
        query_prompt=_EU_PROMPT,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
        tokenizer_kwargs={"padding_side": "left"},
    ),
    ModelConfig(
        name="Qwen3-4B (EU AI Act FT)",
        model_path="danielnoumon/qwen3-embedding-4b-ai-act-nl",
        model_type="st",
        dims=[1024, 512, 256, 128],
        query_prompt=_EU_PROMPT,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
        tokenizer_kwargs={"padding_side": "left"},
    ),
    # --- OpenAI (skipped if no credentials in .env) ---
    ModelConfig(
        name="text-embedding-3-large (OpenAI)",
        model_path="text-embedding-3-large",
        model_type="openai",
        dims=[3072, 1024],
        batch_size=16,
    ),
]

# Eval sets to run
EVAL_SETS: list[str] = ["combined", "eu_ai_act", "gdpr"]

# Dim used in the final comparison table
PRIMARY_DIM = 1024


# ---------------------------------------------------------------------------
# Eval data loading
# ---------------------------------------------------------------------------

def load_eval_set(set_name: str) -> tuple[
    dict[str, str], dict[str, str], dict[str, set[str]],
    list[str], list[str], list[str], list[str],
]:
    """Load queries, corpus, relevant_docs for a named eval set."""
    d = EVAL_ROOT / set_name
    with open(d / "queries.json", encoding="utf-8") as f:
        queries: dict[str, str] = json.load(f)
    with open(d / "corpus.json", encoding="utf-8") as f:
        corpus: dict[str, str] = json.load(f)
    with open(d / "relevant_docs.json", encoding="utf-8") as f:
        rel_raw: dict[str, list[str]] = json.load(f)

    relevant_docs = {qid: set(cids) for qid, cids in rel_raw.items()}
    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    return (
        queries, corpus, relevant_docs,
        query_ids, corpus_ids, query_texts, corpus_texts,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(
    a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-9)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-9)
    return a_norm @ b_norm.T


def compute_metrics(
    sim_matrix: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    relevant_docs: dict[str, set[str]],
    k_values: list[int] = [1, 3, 5, 10],
) -> dict[str, float]:
    ndcg_scores, mrr_scores, map_scores = [], [], []
    recall: dict[int, list[float]] = {k: [] for k in k_values}
    accuracy: dict[int, list[float]] = {k: [] for k in k_values}

    for i, qid in enumerate(query_ids):
        rel = relevant_docs.get(qid, set())
        if not rel:
            continue
        ranked_ids = [
            corpus_ids[idx] for idx in np.argsort(-sim_matrix[i])
        ]

        # NDCG@10
        dcg = sum(
            1.0 / np.log2(r + 2)
            for r, rid in enumerate(ranked_ids[:10])
            if rid in rel
        )
        idcg = sum(
            1.0 / np.log2(r + 2) for r in range(min(10, len(rel)))
        )
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        # MRR@10
        mrr = next(
            (1.0 / (r + 1)
             for r, rid in enumerate(ranked_ids[:10]) if rid in rel),
            0.0,
        )
        mrr_scores.append(mrr)

        # MAP@100
        n_rel, ap = 0, 0.0
        for r, rid in enumerate(ranked_ids[:100]):
            if rid in rel:
                n_rel += 1
                ap += n_rel / (r + 1)
        map_scores.append(ap / len(rel))

        # Recall / Accuracy @k
        for k in k_values:
            hits = len(set(ranked_ids[:k]) & rel)
            recall[k].append(hits / len(rel))
            accuracy[k].append(1.0 if hits > 0 else 0.0)

    out: dict[str, float] = {
        "ndcg@10": float(np.mean(ndcg_scores)),
        "mrr@10": float(np.mean(mrr_scores)),
        "map@100": float(np.mean(map_scores)),
    }
    for k in k_values:
        out[f"recall@{k}"] = float(np.mean(recall[k]))
        out[f"accuracy@{k}"] = float(np.mean(accuracy[k]))
    return out


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def get_openai_client():
    """Return (client, model_name) or None if no credentials found."""
    from openai import AzureOpenAI, OpenAI

    azure_endpoint = os.getenv("EMBEDDINGS_AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        print("  OpenAI backend: Azure OpenAI")
        client = AzureOpenAI(
            api_key=os.getenv("EMBEDDINGS_AZURE_OPENAI_KEY"),
            azure_endpoint=azure_endpoint,
            api_version=os.getenv(
                "EMBEDDINGS_API_VERSION", "2024-06-01"
            ),
        )
        model = os.getenv(
            "EMBEDDINGS_DEPLOYMENT_NAME", "text-embedding-3-large"
        )
        return client, model

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("  OpenAI backend: OpenAI direct")
        return OpenAI(api_key=openai_key), "text-embedding-3-large"

    return None


def embed_openai(
    client, model: str, texts: list[str], dim: int, batch_size: int
) -> np.ndarray:
    all_embs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        resp = client.embeddings.create(
            model=model, input=batch, dimensions=dim
        )
        all_embs.extend(item.embedding for item in resp.data)
        if i > 0 and i % (batch_size * 20) == 0:
            print(f"    {i + len(batch)}/{len(texts)} embedded")
    return np.array(all_embs, dtype=np.float32)


# ---------------------------------------------------------------------------
# ST model runner
# ---------------------------------------------------------------------------

def run_st_model(
    config: ModelConfig,
    eval_sets: list[str],
) -> dict[str, dict[int, dict[str, float]]]:
    """
    Evaluate a SentenceTransformer model on all eval sets.
    Returns results[eval_set][dim] = metrics_dict.
    """
    from sentence_transformers import SentenceTransformer

    model_path = PROJECT_ROOT / config.model_path
    resolved = (
        str(model_path) if model_path.exists() else config.model_path
    )

    print(f"\nLoading {resolved} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        resolved,
        device=device,
        model_kwargs=config.model_kwargs or None,
        tokenizer_kwargs=config.tokenizer_kwargs or None,
    )
    dtype = next(model.parameters()).dtype
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded ({dtype}, {n_params:.0f}M params) on {device}")

    results: dict[str, dict[int, dict[str, float]]] = {}

    for set_name in eval_sets:
        set_dir = EVAL_ROOT / set_name
        if not set_dir.exists():
            print(f"  [SKIP] eval set '{set_name}' not found — "
                  "run prepare_eval_combined.py first")
            continue

        _, _, relevant_docs, q_ids, c_ids, q_texts, c_texts = (
            load_eval_set(set_name)
        )
        print(
            f"  [{set_name}] {len(q_ids)} queries, "
            f"{len(c_ids)} corpus chunks"
        )

        t0 = time.time()
        q_embs = model.encode(
            q_texts,
            prompt=config.query_prompt or None,
            batch_size=config.batch_size,
            normalize_embeddings=False,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        c_embs = model.encode(
            c_texts,
            prompt=config.corpus_prompt or None,
            batch_size=config.batch_size,
            normalize_embeddings=False,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        print(f"  Encoding done in {time.time() - t0:.1f}s")

        results[set_name] = {}
        for dim in config.dims:
            sim = cosine_similarity_matrix(
                q_embs[:, :dim], c_embs[:, :dim]
            )
            metrics = compute_metrics(sim, q_ids, c_ids, relevant_docs)
            results[set_name][dim] = metrics
            print(
                f"    dim={dim:<5}  ndcg@10={metrics['ndcg@10']:.4f}  "
                f"mrr@10={metrics['mrr@10']:.4f}"
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# OpenAI model runner
# ---------------------------------------------------------------------------

def run_openai_model(
    config: ModelConfig,
    eval_sets: list[str],
) -> dict[str, dict[int, dict[str, float]]]:
    """
    Evaluate an OpenAI embedding model on all eval sets.
    Returns results[eval_set][dim] = metrics_dict.
    """
    client_info = get_openai_client()
    if client_info is None:
        print(
            "  [SKIP] No OpenAI credentials found. "
            "Set EMBEDDINGS_AZURE_OPENAI_ENDPOINT + "
            "EMBEDDINGS_AZURE_OPENAI_KEY or OPENAI_API_KEY in .env"
        )
        return {}

    client, model_name = client_info
    print(f"  Model: {model_name}")
    results: dict[str, dict[int, dict[str, float]]] = {}

    for set_name in eval_sets:
        set_dir = EVAL_ROOT / set_name
        if not set_dir.exists():
            print(f"  [SKIP] eval set '{set_name}' not found")
            continue

        _, _, relevant_docs, q_ids, c_ids, q_texts, c_texts = (
            load_eval_set(set_name)
        )
        print(
            f"  [{set_name}] {len(q_ids)} queries, "
            f"{len(c_ids)} corpus chunks"
        )
        results[set_name] = {}

        for dim in config.dims:
            print(f"    dim={dim} — embedding queries ...")
            t0 = time.time()
            q_embs = embed_openai(
                client, model_name, q_texts, dim, config.batch_size
            )
            print(f"    dim={dim} — embedding corpus ...")
            c_embs = embed_openai(
                client, model_name, c_texts, dim, config.batch_size
            )
            print(f"    Embedding done in {time.time() - t0:.1f}s")

            sim = cosine_similarity_matrix(q_embs, c_embs)
            metrics = compute_metrics(sim, q_ids, c_ids, relevant_docs)
            results[set_name][dim] = metrics
            print(
                f"    dim={dim:<5}  ndcg@10={metrics['ndcg@10']:.4f}  "
                f"mrr@10={metrics['mrr@10']:.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def best_dim_metrics(
    model_results: dict[str, dict[int, dict[str, float]]],
    set_name: str,
    primary_dim: int,
    metric: str = "ndcg@10",
) -> float | None:
    set_data = model_results.get(set_name, {})
    if not set_data:
        return None
    if primary_dim in set_data:
        return set_data[primary_dim].get(metric)
    closest = min(set_data.keys(), key=lambda d: abs(d - primary_dim))
    return set_data[closest].get(metric)


def print_results_table(
    all_results: dict[str, dict],
    eval_sets: list[str],
    primary_dim: int,
) -> None:
    metric = "ndcg@10"
    col_w = 12
    name_w = 44

    print(f"\n{'=' * (name_w + col_w * len(eval_sets) + 4)}")
    print(
        f"RESULTS: {metric.upper()} @ dim={primary_dim}  "
        f"(* closest available dim used if {primary_dim} unavailable)"
    )
    print(f"{'=' * (name_w + col_w * len(eval_sets) + 4)}")
    header = f"{'Model':<{name_w}}" + "".join(
        f"{s:>{col_w}}" for s in eval_sets
    )
    print(header)
    print("-" * len(header))

    for model_name, model_results in all_results.items():
        row = f"{model_name:<{name_w}}"
        for set_name in eval_sets:
            val = best_dim_metrics(
                model_results, set_name, primary_dim, metric
            )
            row += f"{val:>{col_w}.4f}" if val is not None else f"{'—':>{col_w}}"
        print(row)

    print("=" * len(header))

    # Detail block per model at primary dim
    print(f"\n{'=' * 62}")
    print(f"FULL METRICS @ dim={primary_dim}  (combined eval set)")
    print(f"{'=' * 62}")
    detail_keys = [
        ("NDCG@10", "ndcg@10"),
        ("MRR@10", "mrr@10"),
        ("MAP@100", "map@100"),
        ("Accuracy@1", "accuracy@1"),
        ("Accuracy@3", "accuracy@3"),
        ("Accuracy@5", "accuracy@5"),
        ("Accuracy@10", "accuracy@10"),
        ("Recall@1", "recall@1"),
        ("Recall@5", "recall@5"),
        ("Recall@10", "recall@10"),
    ]
    for model_name, model_results in all_results.items():
        combined = model_results.get("combined", {})
        if not combined:
            continue
        dim_data = combined.get(
            primary_dim,
            combined.get(
                min(combined.keys(), key=lambda d: abs(d - primary_dim)),
                {},
            ),
        )
        if not dim_data:
            continue
        print(f"\n  {model_name}")
        for label, key in detail_keys:
            val = dim_data.get(key)
            if val is not None:
                print(f"    {label:<14}  {val:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("Device: CPU (no CUDA — ST models will be slow)")

    all_results: dict[str, dict] = {}

    for config in MODEL_CONFIGS:
        print(f"\n{'─' * 62}")
        print(f"Model: {config.name}")

        # Gracefully skip fine-tuned models if not present locally
        if config.model_type == "st":
            local = PROJECT_ROOT / config.model_path
            if (
                not local.exists()
                and "/" not in config.model_path
                and "\\" not in config.model_path
            ):
                print(f"  [SKIP] Local path not found: {local}")
                continue

        t_model = time.time()

        if config.model_type == "st":
            results = run_st_model(config, EVAL_SETS)
        else:
            results = run_openai_model(config, EVAL_SETS)

        if results:
            all_results[config.name] = results
            elapsed = time.time() - t_model
            print(f"  Total time: {elapsed:.0f}s")

    if not all_results:
        print("\nNo models evaluated. Check model paths and .env credentials.")
    else:
        print_results_table(all_results, EVAL_SETS, PRIMARY_DIM)

        # Save JSON results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = (
            PROJECT_ROOT / "evaluation" / f"results_combined_{ts}.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {out_path.relative_to(PROJECT_ROOT)}")
