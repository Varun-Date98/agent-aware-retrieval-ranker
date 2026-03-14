from __future__ import annotations

import abc
import hashlib
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.utils import ScoredPassage, load_config


class Retriever(abc.ABC):
    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> List[ScoredPassage]:
        ...


class BM25Retriever(Retriever):
    def __init__(self, passages: List[dict]):
        """passages: list of dicts with keys 'doc_id', 'title', 'text'."""
        self.passages = passages
        tokenized = [p["text"].lower().split() for p in passages]
        self.bm25 = BM25Okapi(tokenized)
        cfg = load_config()
        self._default_k = cfg["retriever"]["bm25_top_k"]

    def retrieve(self, query: str, top_k: int | None = None) -> List[ScoredPassage]:
        k = top_k or self._default_k
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            ScoredPassage(
                doc_id=self.passages[i]["doc_id"],
                text=self.passages[i]["text"],
                title=self.passages[i].get("title", ""),
                score=float(scores[i]),
            )
            for i in top_idx
            if scores[i] > 0
        ]


class DenseRetriever(Retriever):
    def __init__(self, passages: List[dict], model: SentenceTransformer | None = None):
        cfg = load_config()
        self.passages = passages
        self.model = model or SentenceTransformer(cfg["retriever"]["dense_model"])
        self._default_k = cfg["retriever"]["dense_top_k"]
        texts = [p["text"] for p in passages]
        self.embeddings = self.model.encode(texts, show_progress_bar=True,
                                            convert_to_numpy=True, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int | None = None) -> List[ScoredPassage]:
        k = top_k or self._default_k
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores = (self.embeddings @ q_emb.T).squeeze()
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            ScoredPassage(
                doc_id=self.passages[i]["doc_id"],
                text=self.passages[i]["text"],
                title=self.passages[i].get("title", ""),
                score=float(scores[i]),
            )
            for i in top_idx
        ]


class HybridRetriever(Retriever):
    """Combines BM25 and dense retrieval via Reciprocal Rank Fusion."""

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever):
        self.bm25 = bm25
        self.dense = dense
        cfg = load_config()
        self.rrf_k = cfg["retriever"]["rrf_k"]

    def retrieve(self, query: str, top_k: int | None = None) -> List[ScoredPassage]:
        k = top_k or max(self.bm25._default_k, self.dense._default_k)
        fetch_k = k * 2

        bm25_results = self.bm25.retrieve(query, top_k=fetch_k)
        dense_results = self.dense.retrieve(query, top_k=fetch_k)

        rrf_scores: dict[str, float] = {}
        passage_map: dict[str, ScoredPassage] = {}

        for rank, sp in enumerate(bm25_results):
            rrf_scores[sp.doc_id] = rrf_scores.get(sp.doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            passage_map[sp.doc_id] = sp

        for rank, sp in enumerate(dense_results):
            rrf_scores[sp.doc_id] = rrf_scores.get(sp.doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if sp.doc_id not in passage_map:
                passage_map[sp.doc_id] = sp

        sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[:k]
        return [
            ScoredPassage(
                doc_id=did,
                text=passage_map[did].text,
                title=passage_map[did].title,
                score=rrf_scores[did],
            )
            for did in sorted_ids
        ]


def build_passage_id(title: str, sent_idx: int) -> str:
    return hashlib.md5(f"{title}::{sent_idx}".encode()).hexdigest()[:12]
