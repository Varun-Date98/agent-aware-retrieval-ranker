"""LLM reasoner that generates a final answer from retrieved passages."""
from __future__ import annotations

import logging
from typing import List

from src.utils import ScoredPassage, count_tokens, get_openai_client, load_config

logger = logging.getLogger("benchmarks")

SYSTEM_PROMPT = (
    "You are a precise question-answering system that extracts answers from provided context passages."
    "\n\nIMPORTANT INSTRUCTIONS:"
    "\n1. ONLY use information from the context passages below. Do not use outside knowledge."
    "\n2. Pay close attention to passage titles - they often contain key entity names."
    "\n3. For multi-hop questions requiring multiple facts, combine information from different passages."
    "\n4. If you find partial information, provide your best answer based on what is available."
    "\n5. ONLY respond with 'INSUFFICIENT CONTEXT' if the context contains absolutely no relevant information."
    "\n\nANSWER FORMAT RULES:"
    "\n- For yes/no questions: answer only 'yes' or 'no'"
    "\n- For entity/name questions: provide just the entity name"
    "\n- For date/number questions: provide just the date or number"
    "\n- Keep answers short and exact - no explanations"
    "\n- Always end with exactly: Answer: <your answer>"
    "\n\nEXAMPLES:"
    "\nQ: What is the capital of France?"
    "\nContext: [1] France - France is a country in Europe. Paris is its capital city."
    "\nAnswer: Paris"
    "\n"
    "\nQ: Are both X and Y musicians?"
    "\nContext: [1] X is a musician. [2] Y is an actor."
    "\nAnswer: no"
    "\n"
    "\nQ: When was the director of film X born?"
    "\nContext: [1] Film X was directed by John Smith. [2] John Smith was born in 1965."
    "\nAnswer: 1965"
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
