"""Evaluate the reranker in isolation with ranking metrics (Recall@k, MRR).

Run:  python -m training.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.reranker import NeuralReranker
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from src.utils import ScoredPassage, load_config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_corpus() -> list[dict]:
    path = DATA_DIR / "corpus.jsonl"
    corpus = []
    with open(path) as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus


def _load_val_samples() -> list[dict]:
    """Load val triplets and group by query -> set of positive doc_ids."""
    path = DATA_DIR / "val.jsonl"
    query_positives: dict[str, set[str]] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["label"] == 1:
                query_positives.setdefault(row["query"], set()).add(row["doc_id"])
    return [{"query": q, "positives": pids} for q, pids in query_positives.items()]


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    top = set(ranked_ids[:k])
    return 1.0 if top & gold_ids else 0.0


def reciprocal_rank(ranked_ids: list[str], gold_ids: set[str]) -> float:
    for i, did in enumerate(ranked_ids):
        if did in gold_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retriever(
    retriever_name: str,
    retrieve_fn,
    val_samples: list[dict],
    ks: list[int],
    reranker: NeuralReranker | None = None,
):
    recalls = {k: [] for k in ks}
    mrrs = []

    for sample in tqdm(val_samples, desc=retriever_name):
        results: list[ScoredPassage] = retrieve_fn(sample["query"])
        if reranker is not None:
            results = reranker.rerank(sample["query"], results, top_k=max(ks))
        ranked_ids = [r.doc_id for r in results]
        gold = sample["positives"]

        for k in ks:
            recalls[k].append(recall_at_k(ranked_ids, gold, k))
        mrrs.append(reciprocal_rank(ranked_ids, gold))

    metrics = {}
    for k in ks:
        metrics[f"Recall@{k}"] = np.mean(recalls[k])
    metrics["MRR"] = np.mean(mrrs)
    return metrics


def main():
    cfg = load_config()
    ks = cfg["evaluation"]["recall_at_k"]

    print("Loading corpus...")
    corpus = _load_corpus()
    print(f"  {len(corpus)} passages")

    print("Loading validation samples...")
    val_samples = _load_val_samples()
    print(f"  {len(val_samples)} queries with gold positives")

    print("Building retrievers...")
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)
    hybrid = HybridRetriever(bm25, dense)

    print("Loading reranker...")
    reranker = NeuralReranker()

    configs = [
        ("BM25-only", lambda q: bm25.retrieve(q), None),
        ("Dense-only", lambda q: dense.retrieve(q), None),
        ("Hybrid (no rerank)", lambda q: hybrid.retrieve(q), None),
        ("Hybrid + Reranker", lambda q: hybrid.retrieve(q), reranker),
    ]

    print("\n" + "=" * 70)
    print("Reranker Evaluation — Ranking Metrics")
    print("=" * 70)

    all_results = {}
    for name, retrieve_fn, rr in configs:
        metrics = evaluate_retriever(name, retrieve_fn, val_samples, ks, rr)
        all_results[name] = metrics
        parts = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"  {name:25s}  {parts}")

    print("=" * 70)

    results_path = DATA_DIR / "reranker_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
