"""Pydantic models for API request/response schemas."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Passage(BaseModel):
    """A single passage with metadata."""
    doc_id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Passage text content")
    title: str = Field(default="", description="Optional passage title")
    score: float = Field(default=0.0, description="Relevance score")


class RerankRequest(BaseModel):
    """Request schema for /rerank endpoint."""
    question: str = Field(..., description="Query text to rerank against")
    passages: List[Passage] = Field(..., description="Retrieved passages to rerank")
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of top results to return"
    )


class RerankResponse(BaseModel):
    """Response schema for /rerank endpoint."""
    reranked_passages: List[Passage] = Field(..., description="Passages sorted by relevance score")
    latency_ms: float = Field(..., description="Reranking latency in milliseconds")
