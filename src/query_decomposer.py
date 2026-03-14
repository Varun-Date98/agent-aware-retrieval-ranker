"""Optional query decomposer (disabled by default for v1 / HotpotQA)."""
from __future__ import annotations

from typing import List

from src.utils import get_openai_client, load_config

SYSTEM_PROMPT = (
    "You are a query decomposition assistant. Given a complex question, "
    "break it into 1-3 simple, atomic sub-questions that together answer the "
    "original. If the question is already simple, return it unchanged.\n\n"
    "Return ONLY the sub-questions, one per line, no numbering or bullets."
)


class QueryDecomposer:
    def __init__(self):
        cfg = load_config()
        self.enabled = cfg["pipeline"]["use_decomposer"]
        self.model = cfg["openai"]["decomposer_model"]

    def decompose(self, query: str) -> List[str]:
        if not self.enabled:
            return [query]

        client = get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text = response.choices[0].message.content.strip()
        sub_queries = [line.strip() for line in text.split("\n") if line.strip()]
        return sub_queries if sub_queries else [query]
