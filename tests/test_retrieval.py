"""
Unit Tests for Task 3.32: Similarity Search & Top-K Retrieval Module
======================================================================
Tests query validation, k parameter validation, embedding model reuse,
vector similarity search ranking, formatted output rendering, empty collection safety,
missing metadata/text fallbacks, embedding API error handling, and top-k retrieval demonstrations.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.indexing import VectorCollection
from src.retrieval import (
    retrieve,
    format_retrieval_output,
    load_indexed_vector_collection,
    run_retrieval_demonstration,
)


def get_sample_vector_collection() -> VectorCollection:
    """Helper to construct a pre-populated mock VectorCollection with known embeddings."""
    collection = VectorCollection(name="test_retrieval_col")
    records = [
        {
            "id": "CDLP-ACC-01",
            "vector": [1.0, 0.0, 0.0, 0.0],
            "text": "To reset your learner account password, navigate to the login portal and click 'Forgot Password'.",
            "metadata": {
                "source": "account-guide.md",
                "chunk_index": 1,
                "section": "Authentication",
                "page": "1"
            }
        },
        {
            "id": "CDLP-IN-8471-01",
            "vector": [0.0, 1.0, 0.0, 0.0],
            "text": "Customs rules for shipping laptops (HS Code 8471.30) to India require BIS registration and commercial invoice.",
            "metadata": {
                "source": "customs_reg_india.json",
                "chunk_index": 1,
                "section": "Electronics",
                "country": "India",
                "hs_code": "8471.30"
            }
        },
        {
            "id": "CDLP-DOC-01",
            "vector": [0.0, 0.7071, 0.7071, 0.0],
            "text": "Mandatory shipping documents for international trade include Commercial Invoice, Packing List, and Bill of Lading.",
            "metadata": {
                "source": "shipping_rules.txt",
                "chunk_index": 2,
                "section": "Documentation",
                "page": "3"
            }
        },
        {
            "id": "CDLP-UNREL-01",
            "vector": [0.0, 0.0, 0.0, 1.0],
            "text": "Jupiter is the largest planet in our Solar System and has 95 officially recognized moons.",
            "metadata": {
                "source": "astronomy_guide.txt",
                "chunk_index": 1,
                "section": "Planets",
                "page": "12"
            }
        }
    ]
    collection.upsert(records)
    return collection


class TestRetrievalPipeline(unittest.TestCase):

    def setUp(self):
        self.collection = get_sample_vector_collection()

    # --------------------------------------------------------------------------
    # 1. QUERY VALIDATION TESTS
    # --------------------------------------------------------------------------

    def test_retrieve_empty_query_raises_value_error(self):
        """Verifies that empty string query raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            retrieve("", k=3, collection=self.collection)
        self.assertIn("Query must be a non-empty string", str(ctx.exception))

    def test_retrieve_whitespace_query_raises_value_error(self):
        """Verifies that whitespace-only query raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            retrieve("   \n\t  ", k=3, collection=self.collection)
        self.assertIn("Query must be a non-empty string", str(ctx.exception))

    def test_retrieve_invalid_query_type_raises_value_error(self):
        """Verifies that non-string query type raises ValueError."""
        with self.assertRaises(ValueError):
            retrieve(None, k=3, collection=self.collection)

        with self.assertRaises(ValueError):
            retrieve(12345, k=3, collection=self.collection)

    # --------------------------------------------------------------------------
    # 2. K PARAMETER VALIDATION TESTS
    # --------------------------------------------------------------------------

    def test_retrieve_zero_or_negative_k_raises_value_error(self):
        """Verifies that k <= 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            retrieve("How to reset password?", k=0, collection=self.collection)
        self.assertIn("must be a positive integer", str(ctx.exception))

        with self.assertRaises(ValueError):
            retrieve("How to reset password?", k=-5, collection=self.collection)

    def test_retrieve_non_integer_k_raises_value_error(self):
        """Verifies that boolean or float k parameter raises ValueError."""
        with self.assertRaises(ValueError):
            retrieve("How to reset password?", k=True, collection=self.collection)

        with self.assertRaises(ValueError):
            retrieve("How to reset password?", k=3.5, collection=self.collection)

        with self.assertRaises(ValueError):
            retrieve("How to reset password?", k="3", collection=self.collection)

    def test_retrieve_k_greater_than_collection_count(self):
        """Verifies that when k > indexed chunks count, all available chunks are returned."""
        with patch("src.retrieval.generate_query_embedding") as mock_embed:
            mock_embed.return_value = [1.0, 0.0, 0.0, 0.0]
            # Collection has 4 chunks, requested k=100
            results = retrieve("How to reset password?", k=100, collection=self.collection)
            self.assertEqual(len(results), 4)

    # --------------------------------------------------------------------------
    # 3. VECTOR SIMILARITY SEARCH & RANKING TESTS
    # --------------------------------------------------------------------------

    @patch("src.retrieval.generate_query_embedding")
    def test_retrieve_password_reset_query_ranking(self, mock_embed):
        """Verifies exact password reset query returns account-guide chunk at rank 1."""
        # Query vector matches account password reset record [1.0, 0.0, 0.0, 0.0]
        mock_embed.return_value = [1.0, 0.0, 0.0, 0.0]

        results = retrieve("How can a learner reset their password?", k=3, collection=self.collection)

        self.assertEqual(len(results), 3)
        # Check rank 1
        self.assertEqual(results[0]["rank"], 1)
        self.assertAlmostEqual(results[0]["similarity_score"], 1.0, places=3)
        self.assertEqual(results[0]["source"], "account-guide.md")
        self.assertIn("reset your learner account password", results[0]["chunk_text"])

        # Verify ordering is strictly descending by similarity score
        scores = [res["similarity_score"] for res in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    @patch("src.retrieval.generate_query_embedding")
    def test_retrieve_returns_all_required_result_fields(self, mock_embed):
        """Verifies result dict contains rank, similarity score, chunk text, source, chunk index, and metadata."""
        mock_embed.return_value = [0.0, 1.0, 0.0, 0.0]

        results = retrieve("What are the shipping rules in India?", k=1, collection=self.collection)
        self.assertEqual(len(results), 1)

        item = results[0]
        self.assertIn("rank", item)
        self.assertIn("similarity_score", item)
        self.assertIn("chunk_text", item)
        self.assertIn("source", item)
        self.assertIn("chunk_index", item)
        self.assertIn("metadata", item)

        self.assertEqual(item["rank"], 1)
        self.assertEqual(item["source"], "customs_reg_india.json")
        self.assertEqual(item["chunk_index"], 1)

    # --------------------------------------------------------------------------
    # 4. TOP-K PARAMETER DEMONSTRATIONS (k=1, k=3, k=5)
    # --------------------------------------------------------------------------

    @patch("src.retrieval.generate_query_embedding")
    def test_retrieve_top_k_demonstrations(self, mock_embed):
        """Verifies that k=1 returns 1 chunk, k=3 returns 3 chunks, and k=5 returns 4 (capped to total)."""
        mock_embed.return_value = [0.0, 1.0, 0.0, 0.0]
        query = "What are the shipping rules in India?"

        res_k1 = retrieve(query, k=1, collection=self.collection)
        self.assertEqual(len(res_k1), 1)

        res_k3 = retrieve(query, k=3, collection=self.collection)
        self.assertEqual(len(res_k3), 3)

        res_k5 = retrieve(query, k=5, collection=self.collection)
        self.assertEqual(len(res_k5), 4)  # Capped at collection total of 4

    # --------------------------------------------------------------------------
    # 5. ERROR HANDLING & FALLBACK TESTS
    # --------------------------------------------------------------------------

    def test_retrieve_empty_collection_raises_value_error(self):
        """Verifies that querying an empty collection raises ValueError."""
        empty_col = VectorCollection(name="empty_collection")
        with self.assertRaises(ValueError) as ctx:
            retrieve("What are the shipping rules?", k=3, collection=empty_col)
        self.assertIn("Vector collection is empty", str(ctx.exception))

    @patch("src.retrieval.generate_query_embedding")
    def test_retrieve_api_error_raises_runtime_error(self, mock_embed):
        """Verifies embedding API exception is converted to RuntimeError."""
        mock_embed.side_effect = RuntimeError("API key invalid")

        with self.assertRaises(RuntimeError) as ctx:
            retrieve("Test query", k=3, collection=self.collection)
        self.assertIn("Embedding API error", str(ctx.exception))

    def test_retrieve_missing_metadata_or_chunk_text_handled_safely(self):
        """Verifies missing text or missing metadata fields use safe fallback values."""
        sparse_col = VectorCollection(name="sparse_col")
        sparse_col.upsert([
            {
                "id": "SPARSE_01",
                "vector": [1.0, 0.0, 0.0, 0.0],
                "text": "",
                "metadata": {}
            }
        ])

        with patch("src.retrieval.generate_query_embedding") as mock_embed:
            mock_embed.return_value = [1.0, 0.0, 0.0, 0.0]
            results = retrieve("Test query", k=1, collection=sparse_col)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["chunk_text"], "[Missing chunk text]")
            self.assertEqual(results[0]["source"], "unknown")
            self.assertEqual(results[0]["chunk_index"], 1)

    # --------------------------------------------------------------------------
    # 6. FORMATTED OUTPUT RENDERER TESTS
    # --------------------------------------------------------------------------

    def test_format_retrieval_output_schema(self):
        """Verifies that format_retrieval_output produces specified ASCII format."""
        sample_results = [
            {
                "rank": 1,
                "similarity_score": 0.9521,
                "source": "account-guide.md",
                "chunk_index": 1,
                "chunk_text": "To reset password click forgot password."
            },
            {
                "rank": 2,
                "similarity_score": 0.4120,
                "source": "shipping_rules.txt",
                "chunk_index": 2,
                "chunk_text": "Commercial invoices required for customs."
            }
        ]

        formatted = format_retrieval_output(
            query="How can a learner reset their password?",
            results=sample_results,
            model_name="text-embedding-3-small",
            k=3
        )

        self.assertIn("Top-K Retrieval", formatted)
        self.assertIn("Query: How can a learner reset their password?", formatted)
        self.assertIn("Embedding Model: text-embedding-3-small", formatted)
        self.assertIn("Top-K: 3", formatted)
        self.assertIn("--- Rank 1 ---", formatted)
        self.assertIn("Similarity Score: 0.9521", formatted)
        self.assertIn("Source: account-guide.md", formatted)
        self.assertIn("Chunk Index: 1", formatted)
        self.assertIn("Text: To reset password click forgot password.", formatted)
        self.assertIn("--- Rank 2 ---", formatted)


if __name__ == "__main__":
    unittest.main()
