"""Optional query decomposer (disabled by default for v1 / HotpotQA)."""
from __future__ import annotations

from typing import List

from src.utils import get_openai_client, load_config

SYSTEM_PROMPT = (
    "You are a query decomposition assistant for multi-hop question answering."
    "Given a question, rewrite it as 1 to 3 short, factual sub-questions that are easier to retrieve documents for."
    "Rules:"
    "- Decompose only if the question requires multiple facts or entities to answer."
    "- If decomposition is needed, each sub-question must be independently searchable."
    "- Preserve the original entity names and key phrases from the question whenever possible."
    "- Do not use pronouns like 'he', 'she', 'it', 'they', 'this person', or 'that film'."
    "- Make each sub-question explicit and self-contained."
    "- Prefer bridge-style decomposition:"
    "  1. identify the intermediate entity or fact"
    "  2. ask for the needed attribute, relation, or comparison"
    "- For comparison questions, produce sub-questions for each entity and then one final comparison question only if necessary."
    "- Do not add outside knowledge."
    "- If the question truly requires only one fact, return it unchanged."
    "Return ONLY the sub-questions, one per line."
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
