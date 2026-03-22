"""Dependency injection for FastAPI endpoints."""
from __future__ import annotations

from src.reranker import NeuralReranker

_reranker_instance: NeuralReranker | None = None


def get_reranker() -> NeuralReranker:
    """Return singleton NeuralReranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = NeuralReranker()
    return _reranker_instance
