"""
Unit tests for KDU 3.25 Embedding Fundamentals & Vector Representation (src/embedding_demo.py)
"""

import unittest
import os
import sys

# Ensure project root directory is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.embedding_demo import (
    embed_texts,
    cosine_similarity,
    explain_vector_representation,
    run_embedding_demonstration
)


class TestEmbeddingDemo(unittest.TestCase):

    def test_embed_texts_returns_valid_vectors(self):
        texts = ["Password recovery steps", "Laptop import tariff"]
        embeddings = embed_texts(texts)

        self.assertEqual(len(embeddings), len(texts))
        self.assertIsInstance(embeddings[0], list)
        self.assertIsInstance(embeddings[1], list)
        self.assertEqual(len(embeddings[0]), len(embeddings[1]))
        self.assertGreater(len(embeddings[0]), 0)
        self.assertTrue(all(isinstance(val, float) for val in embeddings[0]))

    def test_cosine_similarity_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_cosine_similarity_orthogonal_vectors(self):
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        sim = cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(sim, 0.0, places=4)

    def test_embedding_demonstration_execution(self):
        results = run_embedding_demonstration()

        self.assertGreaterEqual(len(results["sample_texts"]), 3)
        self.assertTrue(results["all_dimensions_equal"])
        self.assertGreater(results["vector_dimension"], 0)
        self.assertEqual(len(results["sample_vector_preview"]["text_0_first_8_values"]), 8)

        # Similar pair score > Dissimilar pair score
        sim_similar = results["similarity_comparison"]["similar_pair"]["cosine_similarity"]
        sim_dissimilar = results["similarity_comparison"]["dissimilar_pair"]["cosine_similarity"]

        self.assertGreater(sim_similar, sim_dissimilar)
        self.assertTrue(results["similarity_comparison"]["ranking_test_passed"])

    def test_explanation_content(self):
        explanations = explain_vector_representation()

        self.assertIn("what_is_embedding", explanations)
        self.assertIn("what_is_dimension", explanations)
        self.assertIn("why_semantic_search", explanations)

        self.assertIn("vector", explanations["what_is_embedding"].lower())
        self.assertIn("dimension", explanations["what_is_dimension"].lower())
        self.assertIn("semantic", explanations["why_semantic_search"].lower())


if __name__ == "__main__":
    unittest.main()
