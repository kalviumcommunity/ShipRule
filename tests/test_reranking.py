"""
Unit Tests for Task 3.35: Chunk Re-Ranking for Precision Module
================================================================
Tests two-stage retrieval, re-ranking scoring logic, parameter validation
(specifically candidate_k > final_k enforcement), rank movement tracking,
metadata filter preservation, timing metrics, and output formatters.
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
from src.retrieval import retrieve
from src.reranker import (
    rerank_score,
    rerank,
    retrieve_and_rerank,
    format_before_after_comparison,
    format_rank_movement_report,
)


def get_mock_vector_collection() -> VectorCollection:
    """Helper to build a populated VectorCollection with diverse test chunks."""
    collection = VectorCollection(name="test_reranking_col")
    records = [
        {
            "id": "chunk_01",
            "vector": [1.0, 0.0, 0.0],
            "text": "General shipping guide mentioning container sizes and vessel schedules.",
            "metadata": {
                "source": "international_shipping_guide.pdf",
                "chunk_index": 1,
                "section": "General"
            }
        },
        {
            "id": "chunk_02",
            "vector": [0.8, 0.6, 0.0],
            "text": "Statutory registrations for importing electronics into India require BIS registration certificate.",
            "metadata": {
                "source": "customs_requirements.txt",
                "chunk_index": 1,
                "section": "Statutory Registrations"
            }
        },
        {
            "id": "chunk_03",
            "vector": [0.0, 1.0, 0.0],
            "text": "Commercial invoice and packing list are mandatory documents for customs clearance.",
            "metadata": {
                "source": "shipping_rules.txt",
                "chunk_index": 2,
                "section": "Documentation"
            }
        },
        {
            "id": "chunk_04",
            "vector": [0.5, 0.5, 0.707],
            "text": "Detailed BIS CRS registration rules for telecommunications and IT hardware importers in India.",
            "metadata": {
                "source": "customs_requirements.txt",
                "chunk_index": 2,
                "section": "Statutory Registrations"
            }
        }
    ]
    collection.upsert(records)
    return collection


class TestChunkReRanking(unittest.TestCase):

    def setUp(self):
        self.collection = get_mock_vector_collection()

    # --------------------------------------------------------------------------
    # 1. PARAMETER VALIDATION & ERROR HANDLING TESTS
    # --------------------------------------------------------------------------
    def test_candidate_k_greater_than_final_k_validation(self):
        """Verifies ValueError is raised when candidate_k <= final_k."""
        # candidate_k == final_k
        with self.assertRaises(ValueError) as ctx:
            retrieve_and_rerank("electronics import India", candidate_k=3, final_k=3, collection=self.collection)
        self.assertIn("must be strictly greater than final_k", str(ctx.exception))

        # candidate_k < final_k
        with self.assertRaises(ValueError) as ctx:
            retrieve_and_rerank("electronics import India", candidate_k=2, final_k=5, collection=self.collection)
        self.assertIn("must be strictly greater than final_k", str(ctx.exception))

    def test_invalid_query_validation(self):
        """Verifies ValueError is raised for empty or non-string query."""
        with self.assertRaises(ValueError):
            retrieve_and_rerank("", candidate_k=5, final_k=2, collection=self.collection)

        with self.assertRaises(ValueError):
            retrieve_and_rerank(None, candidate_k=5, final_k=2, collection=self.collection)

    def test_invalid_k_type_or_value_validation(self):
        """Verifies ValueError for non-positive or boolean k values."""
        with self.assertRaises(ValueError):
            retrieve_and_rerank("query", candidate_k=-1, final_k=3, collection=self.collection)

        with self.assertRaises(ValueError):
            retrieve_and_rerank("query", candidate_k=5, final_k=0, collection=self.collection)

    # --------------------------------------------------------------------------
    # 2. RE-RANKER SCORING & RANKING TESTS
    # --------------------------------------------------------------------------
    def test_rerank_score_term_coverage(self):
        """Verifies chunk with exact query terms receives higher rerank_score."""
        query = "BIS registration electronics India"
        relevant_text = "BIS registration certificate required for electronics import in India."
        irrelevant_text = "Standard shipping vessel schedule and container sizes."

        score_rel = rerank_score(query, relevant_text, vector_score=0.5)
        score_irrel = rerank_score(query, irrelevant_text, vector_score=0.5)

        self.assertGreater(score_rel, score_irrel)

    def test_rerank_sorting_and_rank_assignment(self):
        """Verifies rerank sorts candidates descending by rerank_score and assigns new ranks."""
        query = "BIS registration certificate"
        candidates = [
            {
                "rank": 1,
                "similarity_score": 0.9,
                "chunk_text": "General shipping vessel schedule.",
                "source": "shipping_guide.pdf",
                "chunk_index": 1
            },
            {
                "rank": 2,
                "similarity_score": 0.7,
                "chunk_text": "Mandatory BIS registration certificate for IT hardware.",
                "source": "customs_requirements.txt",
                "chunk_index": 2
            }
        ]

        reranked = rerank(query=query, candidates=candidates, final_k=2)

        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["rank"], 1)
        self.assertEqual(reranked[0]["original_rank"], 2)  # Originally rank 2 moved to rank 1
        self.assertEqual(reranked[0]["source"], "customs_requirements.txt")
        self.assertGreater(reranked[0]["rerank_score"], reranked[1]["rerank_score"])

    # --------------------------------------------------------------------------
    # 3. TWO-STAGE RETRIEVAL PIPELINE TESTS
    # --------------------------------------------------------------------------
    @patch("src.retrieval.generate_query_embedding")
    def test_retrieve_and_rerank_pipeline(self, mock_embed):
        """Verifies end-to-end retrieve_and_rerank pipeline execution and timing metrics."""
        mock_embed.return_value = [0.8, 0.6, 0.0]

        candidates, top_k, metrics = retrieve_and_rerank(
            query="BIS registration electronics India",
            candidate_k=4,
            final_k=2,
            collection=self.collection
        )

        self.assertEqual(len(candidates), 4)
        self.assertEqual(len(top_k), 2)

        self.assertIn("candidate_k", metrics)
        self.assertIn("final_k", metrics)
        self.assertIn("vector_retrieval_time_ms", metrics)
        self.assertIn("reranking_time_ms", metrics)
        self.assertIn("total_pipeline_time_ms", metrics)

    # --------------------------------------------------------------------------
    # 4. METADATA FILTER PRESERVATION TESTS
    # --------------------------------------------------------------------------
    @patch("src.retrieval.generate_query_embedding")
    def test_metadata_filter_preservation(self, mock_embed):
        """Verifies metadata filter passed to stage 1 is preserved and applied."""
        mock_embed.return_value = [0.8, 0.6, 0.0]

        filter_criteria = {"source": "customs_requirements.txt"}
        candidates, top_k, metrics = retrieve_and_rerank(
            query="BIS registration",
            candidate_k=3,
            final_k=2,
            metadata_filter=filter_criteria,
            collection=self.collection
        )

        for item in top_k:
            self.assertEqual(item["source"], "customs_requirements.txt")

    # --------------------------------------------------------------------------
    # 5. OUTPUT FORMATTER TESTS
    # --------------------------------------------------------------------------
    def test_format_before_after_comparison(self):
        """Verifies BEFORE / AFTER ASCII report string formatting."""
        candidates = [
            {"rank": 1, "similarity_score": 0.8, "source": "a.txt", "chunk_index": 1, "chunk_text": "text A"}
        ]
        reranked = [
            {"rank": 1, "vector_score": 0.8, "rerank_score": 8.5, "source": "a.txt", "chunk_index": 1, "chunk_text": "text A"}
        ]

        report = format_before_after_comparison("query text", candidates, reranked)
        self.assertIn("BEFORE RE-RANKING", report)
        self.assertIn("AFTER RE-RANKING", report)
        self.assertIn("query text", report)

    def test_format_rank_movement_report(self):
        """Verifies candidate rank movement report formatting."""
        reranked = [
            {"rank": 1, "original_rank": 3, "source": "customs.txt", "rerank_score": 9.2}
        ]

        report = format_rank_movement_report(reranked)
        self.assertIn("CANDIDATE RANK MOVEMENT TRACKING", report)
        self.assertIn("Rank 3", report)
        self.assertIn("Rank 1", report)
        self.assertIn("customs.txt", report)


if __name__ == "__main__":
    unittest.main()
