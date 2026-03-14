"""LLM reasoner that generates a final answer from retrieved passages."""
from __future__ import annotations

from typing import List

from src.utils import ScoredPassage, count_tokens, get_openai_client, load_config

SYSTEM_PROMPT = (
    "You are a precise question-answering agent. Given context passages and a "
    "question, provide a short, factual answer. Think step by step, then give "
    "your final answer on the last line prefixed with 'Answer: '."
)


def _format_context(passages: List[ScoredPassage]) -> str:
    parts = []
    for i, p in enumerate(passages, 1):
        header = f"[{i}]"
        if p.title:
            header += f" {p.title}"
        parts.append(f"{header}\n{p.text}")
    return "\n\n".join(parts)


class LLMReasoner:
    def __init__(self):
        cfg = load_config()
        self.model = cfg["openai"]["reasoner_model"]

    def reason(self, query: str, passages: List[ScoredPassage]) -> tuple[str, int]:
        """Return (answer, token_count) where token_count is prompt tokens sent."""
        context = _format_context(passages)
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"
        token_count = count_tokens(SYSTEM_PROMPT + user_msg, self.model)

        client = get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        full_answer = response.choices[0].message.content.strip()

        # Extract the part after "Answer: " if present
        answer = full_answer
        for line in reversed(full_answer.split("\n")):
            line = line.strip()
            if line.lower().startswith("answer:"):
                answer = line[len("answer:"):].strip()
                break

        return answer, token_count
