"""Small dependency-light tests for data validation and search utilities."""

from __future__ import annotations

import unittest

import numpy as np

from search import keyword_search
from utils import cosine_similarity, load_dataset


class CoreTests(unittest.TestCase):
    def test_dataset_has_40_valid_unique_records(self) -> None:
        records = load_dataset()
        self.assertEqual(len(records), 40)
        self.assertEqual(len({record["id"] for record in records}), 40)
        self.assertTrue(all(record["question"] and record["answer"] for record in records))

    def test_cosine_similarity_identical_vector_is_one(self) -> None:
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        score = cosine_similarity(vector, vector.reshape(1, -1))[0]
        self.assertAlmostEqual(float(score), 1.0, places=6)

    def test_cosine_similarity_unrelated_axis_vector_is_zero(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        matrix = np.array([[0.0, 1.0]], dtype=np.float32)
        score = cosine_similarity(query, matrix)[0]
        self.assertAlmostEqual(float(score), 0.0, places=6)

    def test_keyword_search_returns_shared_terms(self) -> None:
        records = load_dataset()
        results = keyword_search("reset forgotten account password", records, top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "FAQ001")
        self.assertIn("password", results[0]["matched_terms"])

    def test_keyword_search_empty_query_returns_no_results(self) -> None:
        self.assertEqual(keyword_search("   ", load_dataset()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
