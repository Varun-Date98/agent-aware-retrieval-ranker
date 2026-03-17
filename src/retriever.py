from __future__ import annotations

import abc
import hashlib
import json
from pathlib import Path
from typing import List

import bm25s
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.utils import ScoredPassage, load_config


def _chunk_passage(
    text: str,
    tokenizer: AutoTokenizer,
    chunk_size: int,
    overlap: int,
    max_tokens: int | None = None,
) -> list[str]:
    """Split text into overlapping token chunks. Returns list of chunk text strings.
    max_tokens: if set, ensures no chunk exceeds this; tokenizer.encode will truncate at model max."""
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    effective_size = min(chunk_size, max_tokens) if max_tokens else chunk_size
    if len(tokens) <= effective_size:
        chunk = tokens[:effective_size] if max_tokens and len(tokens) > max_tokens else tokens
        return [tokenizer.decode(chunk, skip_special_tokens=True)] if chunk else []
    step = effective_size - overlap
    if step < 1:
        step = 1
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + effective_size]
        if not chunk_tokens:
            break
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
    return chunks


def _corpus_hash(passages: list[dict]) -> str:
    """Stable hash of corpus for cache invalidation."""
    contents = json.dumps(
        [(p.get("doc_id"), p.get("title", ""), p.get("text", "")) for p in passages],
        sort_keys=True,
    )
    return hashlib.sha256(contents.encode()).hexdigest()[:16]


class Retriever(abc.ABC):
    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> List[ScoredPassage]:
        ...


class BM25Retriever(Retriever):
    def __init__(self, passages: List[dict]):
        """passages: list of dicts with keys 'doc_id', 'title', 'text'."""
        self.passages = passages
        corpus_texts = [p["text"] for p in passages]
        corpus_tokens = bm25s.tokenize(corpus_texts, lower=True)
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)
        cfg = load_config()
        self._default_k = cfg["retriever"]["bm25_top_k"]

    def retrieve(self, query: str, top_k: int | None = None) -> List[ScoredPassage]:
        k = min(top_k or self._default_k, len(self.passages))
        query_tokens = bm25s.tokenize([query], lower=True)
        indices, scores = self.bm25.retrieve(query_tokens, k=k)
        row_indices = indices[0]
        row_scores = scores[0]
        return [
            ScoredPassage(
                doc_id=self.passages[i]["doc_id"],
                text=self.passages[i]["text"],
                title=self.passages[i].get("title", ""),
                score=float(row_scores[rank]),
            )
            for rank, i in enumerate(row_indices)
        ]


class DenseRetriever(Retriever):
    _CACHE_DIR = Path(__file__).resolve().parent.parent / "data"

    def __init__(self, passages: List[dict], model: SentenceTransformer | None = None):
        cfg = load_config()
        rcfg = cfg["retriever"]
        self.passages = passages
        
        # Initialize model with GPU if available
        if model is not None:
            self.model = model
        else:
            device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
            print(f"Dense retriever running on {device}")
            self.model = SentenceTransformer(rcfg["dense_model"], device=device)
        
        self._default_k = rcfg["dense_top_k"]

        chunk_size = rcfg.get("dense_chunk_size", 0)
        overlap = rcfg.get("dense_chunk_overlap", 0)
        use_chunked = chunk_size > 0 and 0 <= overlap < chunk_size

        if not use_chunked:
            # Legacy: one vector per passage, no title prefix
            model_name = _safe_model_name(rcfg["dense_model"])
            corpus_hash = _corpus_hash(passages)
            cache_path = self._CACHE_DIR / f"dense_{model_name}_{corpus_hash}.npz"
            if cache_path.exists():
                data = np.load(cache_path)
                self.embeddings = data["embeddings"]
                self.chunk_to_passage = None
            else:
                texts = [p["text"] for p in passages]
                encode_batch_size = rcfg.get("dense_encode_batch_size", 256)
                self.embeddings = self.model.encode(
                    texts,
                    batch_size=encode_batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                self.chunk_to_passage = None
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path, embeddings=self.embeddings)
            return

        # Chunked encoding (text only; no title prefix to stay under max_seq_length)
        model_name = _safe_model_name(rcfg["dense_model"])
        corpus_hash = _corpus_hash(passages)
        cache_path = self._CACHE_DIR / f"dense_{model_name}_{corpus_hash}_c{chunk_size}_o{overlap}_textonly.npz"

        if cache_path.exists():
            data = np.load(cache_path)
            self.embeddings = data["embeddings"]
            self.chunk_to_passage = data["chunk_to_passage"]
        else:
            tokenizer = self.model.tokenizer
            max_tokens = 254  # 256 - 2 for [CLS]/[SEP]
            chunk_texts: list[str] = []
            chunk_to_passage: list[int] = []

            for i, p in enumerate(passages):
                text = p.get("text", "") or ""
                chunks = _chunk_passage(text, tokenizer, chunk_size, overlap, max_tokens=max_tokens)
                for chunk in chunks:
                    chunk_texts.append(chunk)
                    chunk_to_passage.append(i)

            if not chunk_texts:
                self.embeddings = np.zeros((len(passages), self.model.get_sentence_embedding_dimension()))
                self.chunk_to_passage = np.arange(len(passages))
            else:
                encode_batch_size = rcfg.get("dense_encode_batch_size", 256)
                self.embeddings = self.model.encode(
                    chunk_texts,
                    batch_size=encode_batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                self.chunk_to_passage = np.array(chunk_to_passage, dtype=np.int64)

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, embeddings=self.embeddings, chunk_to_passage=self.chunk_to_passage)

    def retrieve(self, query: str, top_k: int | None = None) -> List[ScoredPassage]:
        k = top_k or self._default_k
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        if self.chunk_to_passage is None:
            scores = (self.embeddings @ q_emb.T).squeeze()
            top_idx = np.argsort(scores)[::-1][:k]
        else:
            chunk_scores = (self.embeddings @ q_emb.T).squeeze()
            passage_scores = np.full(len(self.passages), -np.inf)
            for j, pidx in enumerate(self.chunk_to_passage):
                passage_scores[pidx] = max(passage_scores[pidx], float(chunk_scores[j]))
            top_idx = np.argsort(passage_scores)[::-1][:k]
            scores = passage_scores
        return [
            ScoredPassage(
                doc_id=self.passages[i]["doc_id"],
                text=self.passages[i]["text"],
                title=self.passages[i].get("title", ""),
                score=float(scores[i]),
            )
            for i in top_idx
        ]


def _safe_model_name(model_name: str) -> str:
    """Filesystem-safe model name (e.g. replace / with _)."""
    return model_name.replace("/", "_")


class HybridRetriever(Retriever):
    """Combines BM25 and dense retrieval via Reciprocal Rank Fusion."""

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever):
        self.bm25 = bm25
        self.dense = dense
        cfg = load_config()
        self.rrf_k = cfg["retriever"]["rrf_k"]

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        fetch_per_retriever: int | None = None,
    ) -> List[ScoredPassage]:
        k = top_k or max(self.bm25._default_k, self.dense._default_k)
        fetch_k = fetch_per_retriever if fetch_per_retriever is not None else k * 2

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
