"""FastAPI application for Agent-Aware Reranker API."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_reranker
from api.models import Passage, RerankRequest, RerankResponse
from src.reranker import NeuralReranker
from src.utils import ScoredPassage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup: warm up the reranker
    reranker = get_reranker()
    print(f"Reranker loaded on {reranker.device}")
    yield
    # Shutdown: cleanup if needed (currently none)


app = FastAPI(
    title="Agent-Aware Reranker API",
    version="1.0.0",
    description="Neural reranker endpoint for agent-optimized search",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/rerank", response_model=RerankResponse)
async def rerank_passages(
    request: RerankRequest,
    reranker: NeuralReranker = Depends(get_reranker),
):
    """Rerank passages for a given question using the trained neural reranker."""
    start = time.perf_counter()

    # Convert Pydantic models to ScoredPassage
    passages = [
        ScoredPassage(
            doc_id=p.doc_id,
            text=p.text,
            title=p.title,
            score=p.score,
        )
        for p in request.passages
    ]

    # Rerank
    reranked = reranker.rerank(request.question, passages, top_k=request.top_k)

    # Convert back to Pydantic
    result_passages = [
        Passage(
            doc_id=p.doc_id,
            text=p.text,
            title=p.title,
            score=p.score,
        )
        for p in reranked
    ]

    latency_ms = (time.perf_counter() - start) * 1000

    return RerankResponse(
        reranked_passages=result_passages,
        latency_ms=latency_ms,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
