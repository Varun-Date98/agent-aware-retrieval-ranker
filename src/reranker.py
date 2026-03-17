from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import ScoredPassage, load_config


def _format_passage(p: ScoredPassage) -> str:
    """Format passage with title prefix for reranker input."""
    title = (p.title or "").strip()
    text = (p.text or "").strip()
    if title:
        return f"{title}. {text}"
    return text


class NeuralReranker:
    def __init__(self, checkpoint_path: str | None = None):
        cfg = load_config()
        rcfg = cfg["reranker"]
        self.model_name = rcfg["model"]
        self.top_k = rcfg["top_k"]
        self.max_seq_length = rcfg.get(
            "max_seq_length", cfg["training"].get("max_seq_length", 512)
        )
        self.device = torch.device(
            cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1
        )

        ckpt = checkpoint_path or rcfg["checkpoint_path"]
        if Path(ckpt).exists():
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank(
        self, query: str, passages: List[ScoredPassage], top_k: int | None = None
    ) -> List[ScoredPassage]:
        k = top_k or self.top_k
        if not passages:
            return []

        pairs = [[query, _format_passage(p)] for p in passages]
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encoded).logits.squeeze(-1).cpu().float()
        scores = torch.sigmoid(logits).tolist()

        reranked = []
        for sp, score in zip(passages, scores):
            reranked.append(
                ScoredPassage(
                    doc_id=sp.doc_id,
                    text=sp.text,
                    title=sp.title,
                    score=score,
                )
            )
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:k]
