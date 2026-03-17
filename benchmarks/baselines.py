"""Baseline pipelines: BM25-only, embedding-only, hybrid (no reranking)."""
from __future__ import annotations

import logging
from typing import List

from src.reasoner import LLMReasoner
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from src.utils import PipelineResult, ScoredPassage, format_passages, timer


class BaselinePipeline:
    """Retrieval -> Reasoning (no decomposition, no reranking)."""

    def __init__(
        self,
        name: str,
        retriever,
        reasoner: LLMReasoner,
        top_k: int = 5,
    ):
        self.name = name
        self.retriever = retriever
        self.reasoner = reasoner
        self.top_k = top_k

    def run(self, query: str, log_context: dict | None = None) -> PipelineResult:
        result = PipelineResult()

        with timer() as e2e:
            with timer() as t_ret:
                passages = self.retriever.retrieve(query, top_k=self.top_k)
            result.latency_retrieval_ms = t_ret.elapsed_ms

            if log_context:
                logger = logging.getLogger("benchmarks")
                logger.debug(
                    "[%s] sample %s: Question: %s",
                    log_context.get("method"),
                    log_context.get("sample_idx"),
                    query,
                )
                logger.debug("  Retrieved (%d): %s", len(passages), format_passages(passages))

            result.latency_reranking_ms = 0.0

            with timer() as t_reason:
                answer, token_count = self.reasoner.reason(query, passages, log_context=log_context)
            result.latency_reasoning_ms = t_reason.elapsed_ms

            result.answer = answer
            result.token_count = token_count

        result.latency_e2e_ms = e2e.elapsed_ms
        return result


def build_baselines(
    corpus: list[dict],
    reasoner: LLMReasoner,
    dense_retriever: DenseRetriever | None = None,
) -> dict[str, BaselinePipeline]:
    """Build all three baseline pipelines sharing the same corpus."""
    bm25 = BM25Retriever(corpus)
    dense = dense_retriever or DenseRetriever(corpus)
    hybrid = HybridRetriever(bm25, dense)

    return {
        "BM25": BaselinePipeline("BM25", bm25, reasoner),
        "Embeddings": BaselinePipeline("Embeddings", dense, reasoner),
        "Hybrid": BaselinePipeline("Hybrid", hybrid, reasoner),
    }
