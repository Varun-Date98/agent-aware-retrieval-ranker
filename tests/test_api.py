"""Tests for FastAPI reranker endpoint."""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rerank_endpoint():
    """Test the /rerank endpoint with sample passages."""
    response = client.post(
        "/rerank",
        json={
            "question": "test query",
            "passages": [
                {"doc_id": "1", "text": "relevant passage", "title": "Title"},
                {"doc_id": "2", "text": "less relevant", "title": ""},
            ],
            "top_k": 2,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["reranked_passages"]) <= 2
    assert data["latency_ms"] > 0
    assert all(
        "doc_id" in p and "text" in p and "title" in p and "score" in p
        for p in data["reranked_passages"]
    )


def test_rerank_top_k_filtering():
    """Test that top_k correctly limits the number of results."""
    passages = [
        {"doc_id": str(i), "text": f"passage {i}", "title": ""}
        for i in range(10)
    ]
    response = client.post(
        "/rerank",
        json={
            "question": "test",
            "passages": passages,
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["reranked_passages"]) == 3


def test_rerank_validation():
    """Test request validation for invalid top_k."""
    response = client.post(
        "/rerank",
        json={
            "question": "test",
            "passages": [{"doc_id": "1", "text": "test"}],
            "top_k": 0,  # Invalid, should be >= 1
        },
    )
    assert response.status_code == 422  # Unprocessable Entity


def test_rerank_with_titles():
    """Test reranking with passages that have titles."""
    response = client.post(
        "/rerank",
        json={
            "question": "What is the capital of France?",
            "passages": [
                {
                    "doc_id": "1",
                    "text": "Paris is the capital of France.",
                    "title": "France",
                },
                {
                    "doc_id": "2",
                    "text": "Berlin is the capital of Germany.",
                    "title": "Germany",
                },
            ],
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["reranked_passages"]) == 2
    # First result should be about Paris (more relevant)
    assert "Paris" in data["reranked_passages"][0]["text"]
