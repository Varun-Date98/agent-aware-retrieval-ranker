"""Pipeline orchestrator wiring all components with per-stage timing."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from src.query_decomposer import QueryDecomposer
from src.reasoner import LLMReasoner
from src.reranker import NeuralReranker
from src.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)
from src.utils import PipelineResult, ScoredPassage, format_passages, load_config, timer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class AgentAwareRankerPipeline:
    def __init__(self, corpus: list[dict] | None = None):
        self.cfg = load_config()

        if corpus is None:
            corpus = self._load_corpus()
        self.corpus = corpus

        self.bm25 = BM25Retriever(corpus)
        self.dense = DenseRetriever(corpus)
        self.hybrid = HybridRetriever(self.bm25, self.dense)
        self.reranker = NeuralReranker()
        self.decomposer = QueryDecomposer()
        self.reasoner = LLMReasoner()

    def _load_corpus(self) -> list[dict]:
        path = DATA_DIR / "corpus.jsonl"
        corpus = []
        with open(path) as f:
            for line in f:
                corpus.append(json.loads(line))
        return corpus

    def run(self, query: str, log_context: dict | None = None) -> PipelineResult:
        result = PipelineResult()

        with timer() as e2e:
            # Optional decomposition
            sub_queries = self.decomposer.decompose(query)

            # Retrieval
            with timer() as t_ret:
                rcfg = self.cfg["retriever"]
                fetch_k = rcfg.get("agent_fetch_k")
                rrf_top = rcfg.get("agent_rrf_top_k")
                all_passages: dict[str, ScoredPassage] = {}
                for sq in sub_queries:
                    if fetch_k is not None and rrf_top is not None:
                        for p in self.hybrid.retrieve(sq, top_k=rrf_top, fetch_per_retriever=fetch_k):
                            if p.doc_id not in all_passages or p.score > all_passages[p.doc_id].score:
                                all_passages[p.doc_id] = p
                    else:
                        for p in self.hybrid.retrieve(sq):
                            if p.doc_id not in all_passages or p.score > all_passages[p.doc_id].score:
                                all_passages[p.doc_id] = p
                candidates = list(all_passages.values())
            result.latency_retrieval_ms = t_ret.elapsed_ms

            if log_context:
                logger = logging.getLogger("benchmarks")
                logger.debug(
                    "[%s] sample %s: Question: %s",
                    log_context.get("method"),
                    log_context.get("sample_idx"),
                    query,
                )
                logger.debug("  Retrieved (%d): %s", len(candidates), format_passages(candidates))

            # Reranking
            with timer() as t_rerank:
                reranked = self.reranker.rerank(query, candidates)
            result.latency_reranking_ms = t_rerank.elapsed_ms

            if log_context:
                logger = logging.getLogger("benchmarks")
                logger.debug("  Reranked (%d): %s", len(reranked), format_passages(reranked))

            # Reasoning
            with timer() as t_reason:
                answer, token_count = self.reasoner.reason(query, reranked, log_context=log_context)
            result.latency_reasoning_ms = t_reason.elapsed_ms

            result.answer = answer
            result.token_count = token_count

        result.latency_e2e_ms = e2e.elapsed_ms
        return result
