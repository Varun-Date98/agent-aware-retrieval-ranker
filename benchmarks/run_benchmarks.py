"""Run all 4 methods on the evaluation set and print the result table.

Run:  python -m benchmarks.run_benchmarks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from benchmarks.baselines import build_baselines
from benchmarks.metrics import SemanticMatcher, compute_all_metrics
from src.pipeline import AgentAwareRankerPipeline
from src.reasoner import LLMReasoner
from src.utils import PipelineResult, load_config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent


def _load_eval_set(cfg: dict) -> list[dict]:
    """Load HotpotQA eval samples (question + gold answer)."""
    ds = load_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["split"],
        split="validation",
    )
    n = cfg["dataset"]["eval_samples"]
    samples = []
    for row in ds:
        if len(samples) >= n:
            break
        samples.append({"question": row["question"], "answer": row["answer"]})
    return samples


def _run_method(name: str, pipeline, eval_samples: list[dict]):
    """Run a pipeline on all eval samples, collecting results."""
    questions, predictions, golds = [], [], []
    token_counts = []
    lat_retrieval, lat_rerank, lat_reasoning, lat_e2e = [], [], [], []

    for sample in tqdm(eval_samples, desc=name):
        result: PipelineResult = pipeline.run(sample["question"])
        questions.append(sample["question"])
        predictions.append(result.answer)
        golds.append(sample["answer"])
        token_counts.append(result.token_count)
        lat_retrieval.append(result.latency_retrieval_ms)
        lat_rerank.append(result.latency_reranking_ms)
        lat_reasoning.append(result.latency_reasoning_ms)
        lat_e2e.append(result.latency_e2e_ms)

    return {
        "questions": questions,
        "predictions": predictions,
        "golds": golds,
        "avg_tokens": float(np.mean(token_counts)),
        "avg_retrieval_ms": float(np.mean(lat_retrieval)),
        "avg_rerank_ms": float(np.mean(lat_rerank)),
        "avg_reasoning_ms": float(np.mean(lat_reasoning)),
        "avg_e2e_ms": float(np.mean(lat_e2e)),
    }


def _print_table(all_results: dict):
    header = (
        f"{'Method':<14} | {'EM':>5} | {'F1':>5} | {'Sem':>5} | {'Judge':>5} "
        f"| {'Tokens':>6} | {'Retrieval':>9} | {'Rerank':>8} | {'Reasoning':>9} | {'E2E':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for name, r in all_results.items():
        m = r["metrics"]
        print(
            f"{name:<14} | {m['EM']:>5.1%} | {m['F1']:>5.1%} | {m['Semantic Match']:>5.1%} "
            f"| {m.get('LLM Judge', 0):>5.1%} | {r['avg_tokens']:>6.0f} "
            f"| {r['avg_retrieval_ms']:>7.0f}ms | {r['avg_rerank_ms']:>6.0f}ms "
            f"| {r['avg_reasoning_ms']:>7.0f}ms | {r['avg_e2e_ms']:>6.0f}ms"
        )
    print("=" * len(header))


def main():
    cfg = load_config()

    print("Loading evaluation set...")
    eval_samples = _load_eval_set(cfg)
    print(f"  {len(eval_samples)} samples")

    print("Building pipelines...")
    agent_pipeline = AgentAwareRankerPipeline()
    reasoner = LLMReasoner()
    baselines = build_baselines(
        agent_pipeline.corpus, reasoner, agent_pipeline.dense
    )

    methods = {**baselines, "Agent-Aware": agent_pipeline}
    semantic_matcher = SemanticMatcher(model=agent_pipeline.dense.model)

    all_results = {}
    for name, pipeline in methods.items():
        print(f"\nRunning {name}...")
        raw = _run_method(name, pipeline, eval_samples)
        metrics = compute_all_metrics(
            raw["questions"], raw["predictions"], raw["golds"],
            semantic_matcher=semantic_matcher,
            run_judge=True,
        )
        all_results[name] = {
            "metrics": metrics,
            "avg_tokens": raw["avg_tokens"],
            "avg_retrieval_ms": raw["avg_retrieval_ms"],
            "avg_rerank_ms": raw["avg_rerank_ms"],
            "avg_reasoning_ms": raw["avg_reasoning_ms"],
            "avg_e2e_ms": raw["avg_e2e_ms"],
        }

    _print_table(all_results)

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed results to {results_path}")


if __name__ == "__main__":
    main()
