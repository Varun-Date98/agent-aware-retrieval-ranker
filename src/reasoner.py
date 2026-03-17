"""LLM reasoner that generates a final answer from retrieved passages."""
from __future__ import annotations

import logging
from typing import List

from src.utils import ScoredPassage, count_tokens, get_openai_client, load_config

logger = logging.getLogger("benchmarks")

SYSTEM_PROMPT = (
    "You are a precise question-answering system."
    "\nYou must answer using only the provided context."
    "Do not use outside knowledge."
    "If the context does not contain enough information to answer confidently, output exactly:"
    "Answer: INSUFFICIENT CONTEXT"
    "\nRules:"
    "- Use only facts stated in the context."
    "- Prefer short, exact answers over full sentences."
    "- For yes/no questions, answer only 'yes' or 'no'."
    "- For entity questions, answer with just the entity name."
    "- For date/year questions, answer with just the date or year."
    "- Do not explain your reasoning."
    "- Output exactly one final line in this format:"
    "Answer: <answer>"
    "Context: {context}"
    "Question: {question}"
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

    def reason(self, query: str, passages: List[ScoredPassage], log_context: dict | None = None) -> tuple[str, int]:
        """Return (answer, token_count) where token_count is prompt tokens sent."""
        context = _format_context(passages)
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"
        token_count = count_tokens(SYSTEM_PROMPT.format(context=context, question=query), self.model)

        if log_context:
            logger = logging.getLogger("benchmarks")
            logger.debug("  Prompt (system): %s", SYSTEM_PROMPT)
            logger.debug("  Prompt (user): %s", user_msg)

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

        if log_context:
            logger = logging.getLogger("benchmarks")
            logger.debug("  LLM raw response: %s", full_answer)

        # Extract the part after "Answer: " if present
        answer = full_answer
        for line in reversed(full_answer.split("\n")):
            line = line.strip()
            if line.lower().startswith("answer:"):
                answer = line[len("answer:"):].strip()
                break

        return answer, token_count
