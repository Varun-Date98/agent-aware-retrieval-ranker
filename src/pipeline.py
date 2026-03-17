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
        print(f"  {len(corpus)} passages loaded")

        self.bm25 = BM25Retriever(corpus)
        print(f"  BM25Retriever loaded")
        self.dense = DenseRetriever(corpus)
        print(f"  DenseRetriever loaded")
        self.hybrid = HybridRetriever(self.bm25, self.dense)
        print(f"  HybridRetriever loaded")
        self.reranker = NeuralReranker()
        print(f"  NeuralReranker loaded")
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

            if log_context:
                logger = logging.getLogger("benchmarks")
                logger.debug(
                    "[%s] sample %s: Decomposed queries (%d): %s",
                    log_context.get("method"),
                    log_context.get("sample_idx"),
                    len(sub_queries),
                    sub_queries,
                )

            # Retrieval
            with timer() as t_ret:
                rcfg = self.cfg["retriever"]
                fetch_k = rcfg.get("agent_fetch_k")
                rrf_top = rcfg.get("agent_rrf_top_k")
                rrf_k = rcfg.get("rrf_k", 60)
                
                # Collect ranked lists per sub-query
                ranked_lists: list[list[ScoredPassage]] = []
                for sq in sub_queries:
                    if fetch_k is not None and rrf_top is not None:
                        ranked_lists.append(self.hybrid.retrieve(sq, top_k=rrf_top, fetch_per_retriever=fetch_k))
                    else:
                        ranked_lists.append(self.hybrid.retrieve(sq))
                
                # Apply RRF fusion across all sub-query results
                if len(ranked_lists) == 1:
                    candidates = ranked_lists[0]
                else:
                    rrf_scores: dict[str, float] = {}
                    passage_map: dict[str, ScoredPassage] = {}
                    
                    for rank_list in ranked_lists:
                        for rank, sp in enumerate(rank_list):
                            rrf_scores[sp.doc_id] = rrf_scores.get(sp.doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
                            if sp.doc_id not in passage_map:
                                passage_map[sp.doc_id] = sp
                    
                    sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)
                    candidates = [
                        ScoredPassage(
                            doc_id=did,
                            text=passage_map[did].text,
                            title=passage_map[did].title,
                            score=rrf_scores[did],
                        )
                        for did in sorted_ids
                    ]
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
