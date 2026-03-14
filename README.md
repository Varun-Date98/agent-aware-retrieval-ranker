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

## Configuration

All settings live in `config/default.yaml`. Model names, hyperparameters, and feature
flags are centralized there — no hardcoded values in source code.

## Project Structure

```
config/default.yaml        — single source of truth for all settings
src/                       — core pipeline components
training/                  — data prep, reranker training, reranker eval
benchmarks/                — end-to-end benchmark harness
data/                      — auto-downloaded HotpotQA data
checkpoints/               — saved model weights
```
