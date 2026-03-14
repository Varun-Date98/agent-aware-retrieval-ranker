"""Evaluation metrics: EM, F1, semantic match, LLM judge, tokens, latency."""
from __future__ import annotations

import re
import string
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import get_openai_client, load_config


# ---------------------------------------------------------------------------
# Text normalization (HotpotQA standard)
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def exact_match(predicted: str, gold: str) -> float:
    return 1.0 if _normalize(predicted) == _normalize(gold) else 0.0


def token_f1(predicted: str, gold: str) -> float:
    pred_tokens = _normalize(predicted).split()
    gold_tokens = _normalize(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------
class SemanticMatcher:
    def __init__(self, model: SentenceTransformer | None = None):
        cfg = load_config()
        self.model = model or SentenceTransformer(cfg["retriever"]["dense_model"])
        self.threshold = cfg["evaluation"]["semantic_similarity_threshold"]

    def score(self, predicted: str, gold: str) -> float:
        embs = self.model.encode([predicted, gold], convert_to_numpy=True,
                                 normalize_embeddings=True)
        sim = float(np.dot(embs[0], embs[1]))
        return sim

    def match(self, predicted: str, gold: str) -> float:
        return 1.0 if self.score(predicted, gold) >= self.threshold else 0.0


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------
JUDGE_PROMPT = (
    "Given the question and the reference answer, is the predicted answer correct?\n"
    "Respond with ONLY 'correct' or 'incorrect'.\n\n"
    "Question: {question}\n"
    "Reference answer: {gold}\n"
    "Predicted answer: {predicted}"
)


def llm_judge(question: str, predicted: str, gold: str) -> float:
    cfg = load_config()
    client = get_openai_client()
    response = client.chat.completions.create(
        model=cfg["openai"]["judge_model"],
        messages=[
            {"role": "user",
             "content": JUDGE_PROMPT.format(
                 question=question, gold=gold, predicted=predicted)},
        ],
        temperature=0.0,
        max_tokens=16,
    )
    verdict = response.choices[0].message.content.strip().lower()
    return 1.0 if "correct" in verdict else 0.0


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------
def compute_all_metrics(
    questions: List[str],
    predictions: List[str],
    golds: List[str],
    semantic_matcher: SemanticMatcher | None = None,
    run_judge: bool = True,
) -> dict[str, float]:
    """Compute all accuracy metrics over a list of samples."""
    ems, f1s, sems, judges = [], [], [], []
    matcher = semantic_matcher or SemanticMatcher()

    for q, pred, gold in zip(questions, predictions, golds):
        ems.append(exact_match(pred, gold))
        f1s.append(token_f1(pred, gold))
        sems.append(matcher.match(pred, gold))
        if run_judge:
            judges.append(llm_judge(q, pred, gold))

    results = {
        "EM": float(np.mean(ems)),
        "F1": float(np.mean(f1s)),
        "Semantic Match": float(np.mean(sems)),
    }
    if run_judge:
        results["LLM Judge"] = float(np.mean(judges))
    return results
