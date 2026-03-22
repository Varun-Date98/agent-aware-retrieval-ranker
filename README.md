# AgentAwareRetrivalRanker

A smart AI layer that optimizes web search results for agents instead of humans.

## Pipeline

```
Agent Query → Retrieval (BM25 + Dense) → Neural Reranking → LLM Reasoning → Answer
```

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

## Usage

### 1. Prepare training data
```bash
python -m training.prepare_data
```

### 2. Train the reranker
```bash
python -m training.train_reranker
```

### 3. Evaluate the reranker (Recall@k, MRR)
```bash
python -m training.evaluate
```

### 4. Run benchmarks
```bash
python -m benchmarks.run_benchmarks
```

### 5. Run the API server

**Development server:**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Production (with multiple workers):**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 6. API Example Usage

```bash
curl -X POST "http://localhost:8000/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "passages": [
      {"doc_id": "1", "text": "Paris is the capital of France.", "title": "France"},
      {"doc_id": "2", "text": "Berlin is the capital of Germany.", "title": "Germany"}
    ],
    "top_k": 5
  }'
```

**Response:**
```json
{
  "reranked_passages": [
    {
      "doc_id": "1",
      "text": "Paris is the capital of France.",
      "title": "France",
      "score": 0.95
    },
    {
      "doc_id": "2",
      "text": "Berlin is the capital of Germany.",
      "title": "Germany",
      "score": 0.12
    }
  ],
  "latency_ms": 45.2
}
```

## Configuration

All settings live in `config/default.yaml`. Model names, hyperparameters, and feature
flags are centralized there — no hardcoded values in source code.

## Results

### End-to-End Performance (HotpotQA, 200 samples)

| Method | EM | F1 | Semantic Match | Judge | Latency (E2E) |
|--------|-----|-----|----------------|-------|---------------|
| BM25 | 15.0% | 19.7% | 19.5% | 21.5% | 1204ms |
| Embeddings (BGE-base) | 18.0% | 22.8% | 22.5% | 26.0% | 1452ms |
| Hybrid (BM25 + Dense) | 16.0% | 22.1% | 23.0% | 25.5% | 1605ms |
| **Agent-Aware** | **24.0%** | **28.4%** | **29.0%** | **31.0%** | 4053ms |

**Agent-Aware pipeline:**
- Query decomposition (GPT-4o) for multi-hop questions
- Hybrid retrieval with RRF fusion across sub-queries
- Neural reranker (512-token, title-aware, trained on HotpotQA)
- GPT-4o reasoning over top-20 reranked passages
- The decomposer adds significant latency to the Agent-Aware pipeline (GPT-4o API call before retrieval)

### Reranker Performance (Isolated Evaluation)

| Method | Recall@5 | Recall@10 | Recall@20 | MRR |
|--------|----------|-----------|-----------|-----|
| Dense-only | 79.8% | 85.0% | 89.3% | 0.611 |
| Hybrid (no rerank) | 80.3% | 86.9% | 90.4% | 0.597 |
| **Hybrid + Reranker** | **87.7%** | **89.8%** | **90.4%** | **0.667** |

The reranker adds **+7.4pp to Recall@5** and **+11.7% to MRR**, demonstrating strong ranking improvements.

## Project Structure

```
config/default.yaml        — single source of truth for all settings
src/                       — core pipeline components
training/                  — data prep, reranker training, reranker eval
benchmarks/                — end-to-end benchmark harness
data/                      — auto-downloaded HotpotQA data
checkpoints/               — saved model weights
```
