"""
Unit Tests for Embedding Similarity & Distance Metrics Module
==============================================================
Tests cosine similarity calculation, zero-vector safety, query embedding generation,
chunk similarity ranking, top_k filtering, empty query handling, missing embeddings,
and structured results schema preservation.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.similarity import (
    cosine_similarity,
    generate_query_embedding,
    rank_chunks_by_similarity,
    search_similar_chunks,
)


class TestEmbeddingSimilarity(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            {
                "embedding_id": "chunk_001",
                "chunk_id": "ship_rule_01",
                "source": "shipping_rules.txt",
                "chunk_index": 1,
                "page": "1",
                "chunk_text": "Commercial invoices are mandatory for all international shipments.",
                "embedding": [1.0, 0.0, 0.0]
            },
            {
                "embedding_id": "chunk_002",
                "chunk_id": "customs_req_01",
                "source": "customs_requirements.txt",
                "chunk_index": 2,
                "page": "1",
                "chunk_text": "Customs duties for laptop computers under HS code 8471.30 require BIS registration.",
                "embedding": [0.7071, 0.7071, 0.0]
            },
            {
                "embedding_id": "chunk_003",
                "chunk_id": "office_menu_01",
                "source": "cafeteria_menu.txt",
                "chunk_index": 1,
                "page": "1",
                "chunk_text": "Cafeteria lunch menu includes pasta and salad.",
                "embedding": [0.0, 1.0, 0.0]
            }
        ]

    def test_cosine_similarity_identical_vectors(self):
        """1. Test that identical vectors return cosine similarity score of 1.0."""
        vec = [0.123, -0.456, 0.789, 0.321]
        score = cosine_similarity(vec, vec)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_cosine_similarity_unrelated_orthogonal_vectors(self):
        """2. Test that orthogonal (unrelated) vectors return similarity score of 0.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        score = cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_cosine_similarity_zero_vectors_handled_safely(self):
        """3. Test zero-norm vectors return 0.0 without causing ZeroDivisionError."""
        zero_vec = [0.0, 0.0, 0.0]
        normal_vec = [1.0, 2.0, 3.0]

        self.assertEqual(cosine_similarity(zero_vec, normal_vec), 0.0)
        self.assertEqual(cosine_similarity(normal_vec, zero_vec), 0.0)
        self.assertEqual(cosine_similarity(zero_vec, zero_vec), 0.0)

    def test_ranking_multiple_chunks(self):
        """4. Test ranking chunks by similarity sorts scores in descending order."""
        query_vec = [1.0, 0.0, 0.0]
        results = rank_chunks_by_similarity(query_vec, self.sample_chunks, top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["metadata"]["document_id"], "chunk_001")
        self.assertEqual(results[0]["score"], 1.0)

        self.assertEqual(results[1]["metadata"]["document_id"], "chunk_002")
        self.assertAlmostEqual(results[1]["score"], 0.7071, places=3)

        self.assertEqual(results[2]["metadata"]["document_id"], "chunk_003")
        self.assertEqual(results[2]["score"], 0.0)

    def test_top_k_parameter(self):
        """5. Test top_k returns exact top_k count, handles zero and large top_k."""
        query_vec = [1.0, 0.0, 0.0]

        # top_k = 2
        top_2 = rank_chunks_by_similarity(query_vec, self.sample_chunks, top_k=2)
        self.assertEqual(len(top_2), 2)

        # top_k = 0 -> empty list
        top_0 = rank_chunks_by_similarity(query_vec, self.sample_chunks, top_k=0)
        self.assertEqual(top_0, [])

        # top_k negative -> empty list
        top_neg = rank_chunks_by_similarity(query_vec, self.sample_chunks, top_k=-5)
        self.assertEqual(top_neg, [])

        # top_k > candidate count -> returns all candidate chunks
        top_large = rank_chunks_by_similarity(query_vec, self.sample_chunks, top_k=100)
        self.assertEqual(len(top_large), 3)

    def test_empty_query_handling(self):
        """6. Test searching with empty or whitespace-only query returns [] without crashing."""
        self.assertEqual(search_similar_chunks("", self.sample_chunks), [])
        self.assertEqual(search_similar_chunks("   \n\t  ", self.sample_chunks), [])

    def test_missing_or_invalid_embeddings_handled_gracefully(self):
        """7. Test candidate chunks missing embeddings or with None are handled safely."""
        corrupted_chunks = [
            {
                "embedding_id": "good_chunk",
                "chunk_text": "Valid text.",
                "embedding": [1.0, 0.0, 0.0]
            },
            {
                "embedding_id": "missing_embed_chunk",
                "chunk_text": "Missing embedding vector.",
                "embedding": None
            },
            {
                "embedding_id": "malformed_chunk",
                "chunk_text": "Non-list embedding vector.",
                "embedding": "invalid_string_vector"
            }
        ]

        query_vec = [1.0, 0.0, 0.0]
        results = rank_chunks_by_similarity(query_vec, corrupted_chunks, top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["metadata"]["document_id"], "good_chunk")
        self.assertEqual(results[0]["score"], 1.0)
        self.assertEqual(results[1]["score"], 0.0)
        self.assertEqual(results[2]["score"], 0.0)

    def test_structured_results_schema(self):
        """8. Test that returned results match the required JSON schema structure."""
        query_vec = [1.0, 0.0, 0.0]
        results = rank_chunks_by_similarity(query_vec, self.sample_chunks[:1], top_k=1)

        self.assertEqual(len(results), 1)
        item = results[0]

        # Verify top-level keys
        self.assertIn("text", item)
        self.assertIn("score", item)
        self.assertIn("metadata", item)

        # Verify metadata dictionary keys
        meta = item["metadata"]
        self.assertIn("source", meta)
        self.assertIn("chunk_index", meta)
        self.assertIn("page", meta)
        self.assertIn("document_id", meta)

        self.assertEqual(meta["source"], "shipping_rules.txt")
        self.assertEqual(meta["page"], "1")

    @patch("src.embeddings.generate_embedding")
    @patch("src.embeddings.create_embedding_client")
    def test_generate_query_embedding_uses_same_model(self, mock_create_client, mock_gen_embed):
        """9. Test that generate_query_embedding uses the exact same model and client setup."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_gen_embed.return_value = [0.1, 0.2, 0.3]

        vec = generate_query_embedding("What are import rules?", client=mock_client, model="text-embedding-3-small")
        self.assertEqual(vec, [0.1, 0.2, 0.3])
        mock_gen_embed.assert_called_once_with(mock_client, "What are import rules?", model="text-embedding-3-small")


if __name__ == "__main__":
    unittest.main()
