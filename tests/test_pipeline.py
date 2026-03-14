"""Smoke tests for core components (no API keys or GPU required)."""
from __future__ import annotations

import unittest

import numpy as np

from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever, build_passage_id
from src.utils import PipelineResult, ScoredPassage, count_tokens, load_config, timer
from benchmarks.metrics import exact_match, token_f1, _normalize

SAMPLE_CORPUS = [
    {"doc_id": "a1", "title": "Python", "text": "Python is a programming language created by Guido van Rossum."},
    {"doc_id": "a2", "title": "Java", "text": "Java is a programming language developed by Sun Microsystems."},
    {"doc_id": "a3", "title": "Paris", "text": "Paris is the capital of France and is known for the Eiffel Tower."},
    {"doc_id": "a4", "title": "Berlin", "text": "Berlin is the capital city of Germany."},
    {"doc_id": "a5", "title": "Mars", "text": "Mars is the fourth planet from the Sun in the solar system."},
]


class TestUtils(unittest.TestCase):
    def test_load_config(self):
        cfg = load_config()
        self.assertIn("openai", cfg)
        self.assertIn("retriever", cfg)
        self.assertIn("reranker", cfg)
        self.assertIn("training", cfg)

    def test_timer(self):
        import time
        with timer() as t:
            time.sleep(0.05)
        self.assertGreater(t.elapsed_ms, 40)

    def test_count_tokens(self):
        n = count_tokens("Hello world")
        self.assertGreater(n, 0)

    def test_pipeline_result_defaults(self):
        r = PipelineResult()
        self.assertEqual(r.answer, "")
        self.assertEqual(r.token_count, 0)

    def test_build_passage_id(self):
        pid = build_passage_id("Python", 0)
        self.assertEqual(len(pid), 12)
        self.assertEqual(pid, build_passage_id("Python", 0))


class TestBM25Retriever(unittest.TestCase):
    def setUp(self):
        self.retriever = BM25Retriever(SAMPLE_CORPUS)

    def test_retrieve_returns_results(self):
        results = self.retriever.retrieve("programming language", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], ScoredPassage)

    def test_retrieve_relevance(self):
        results = self.retriever.retrieve("capital of France", top_k=2)
        doc_ids = [r.doc_id for r in results]
        self.assertIn("a3", doc_ids)

    def test_retrieve_empty_query(self):
        results = self.retriever.retrieve("xyznonexistent", top_k=3)
        self.assertEqual(len(results), 0)


class TestDenseRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = DenseRetriever(SAMPLE_CORPUS)

    def test_retrieve_returns_results(self):
        results = self.retriever.retrieve("what is Python", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], ScoredPassage)

    def test_retrieve_relevance(self):
        results = self.retriever.retrieve("planet in solar system", top_k=1)
        self.assertEqual(results[0].doc_id, "a5")


class TestHybridRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bm25 = BM25Retriever(SAMPLE_CORPUS)
        dense = DenseRetriever(SAMPLE_CORPUS)
        cls.retriever = HybridRetriever(bm25, dense)

    def test_hybrid_returns_results(self):
        results = self.retriever.retrieve("programming language", top_k=3)
        self.assertGreater(len(results), 0)

    def test_hybrid_deduplicates(self):
        results = self.retriever.retrieve("capital of Germany", top_k=5)
        doc_ids = [r.doc_id for r in results]
        self.assertEqual(len(doc_ids), len(set(doc_ids)))


class TestMetrics(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(_normalize("The Capital"), "capital")

    def test_exact_match_positive(self):
        self.assertEqual(exact_match("Paris", "paris"), 1.0)

    def test_exact_match_negative(self):
        self.assertEqual(exact_match("London", "Paris"), 0.0)

    def test_token_f1_perfect(self):
        self.assertAlmostEqual(token_f1("the quick brown fox", "quick brown fox"), 1.0)

    def test_token_f1_partial(self):
        f1 = token_f1("quick red fox", "quick brown fox")
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

    def test_token_f1_no_overlap(self):
        self.assertEqual(token_f1("alpha beta", "gamma delta"), 0.0)


if __name__ == "__main__":
    unittest.main()
