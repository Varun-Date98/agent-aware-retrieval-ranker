from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken
import yaml

_CONFIG_CACHE: dict | None = None
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE
    if path is None:
        path = _PROJECT_ROOT / "config" / "default.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if path is None or Path(path) == _PROJECT_ROOT / "config" / "default.yaml":
        _CONFIG_CACHE = cfg
    return cfg


def get_openai_client():
    """Return an OpenAI client using the API key from config."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    cfg = load_config()
    api_key = os.environ.get(cfg["openai"]["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"Set the {cfg['openai']['api_key_env']} environment variable"
        )
    return OpenAI(api_key=api_key)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


@dataclass
class TimingResult:
    elapsed_ms: float = 0.0


@contextmanager
def timer():
    """Context manager that records wall-clock milliseconds."""
    result = TimingResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_ms = (time.perf_counter() - start) * 1000


@dataclass
class PipelineResult:
    answer: str = ""
    token_count: int = 0
    latency_retrieval_ms: float = 0.0
    latency_reranking_ms: float = 0.0
    latency_reasoning_ms: float = 0.0
    latency_e2e_ms: float = 0.0


@dataclass
class ScoredPassage:
    doc_id: str
    text: str
    score: float
    title: str = ""
