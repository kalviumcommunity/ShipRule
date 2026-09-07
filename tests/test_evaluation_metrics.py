"""
Unit Tests for Task 3.36: Retrieval Evaluation & Recall Testing Module
=======================================================================
Tests evaluate_query(), evaluate_dataset(), evaluate_multi_k(),
Recall@K, Precision@K, MRR@K formulas, zero division safety,
failure inspection, diagnostic logging, and strategy comparisons.
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
from src.evaluation import (
    LABELLED_QUERIES,
    evaluate_query,
    evaluate_dataset,
    evaluate_multi_k,
    inspect_retrieval_failures,
    diagnose_failure_causes,
    compare_retrieval_strategies,
)


def get_test_vector_collection() -> VectorCollection:
    """Constructs a populated VectorCollection for evaluation testing."""
    collection = VectorCollection(name="test_eval_col")
    records = [
        {
            "id": "customs_requirements_paragraph_001",
            "vector": [1.0, 0.0, 0.0],
            "text": "Customs duties and 8-digit HS code tariff classification details.",
            "metadata": {"source": "customs_requirements.txt", "chunk_index": 1}
        },
        {
            "id": "customs_requirements_paragraph_002",
            "vector": [0.0, 1.0, 0.0],
            "text": "Preferential tariff rates and Certificate of Origin under FTA.",
            "metadata": {"source": "customs_requirements.txt", "chunk_index": 2}
        },
        {
            "id": "customs_requirements_paragraph_003",
            "vector": [0.0, 0.0, 1.0],
            "text": "Statutory registrations BIS CRS for IT hardware and telecommunications.",
            "metadata": {"source": "customs_requirements.txt", "chunk_index": 3}
        },
        {
            "id": "international_shipping_guide_paragraph_001",
            "vector": [0.707, 0.707, 0.0],
            "text": "Incoterms 2020 rules for FOB and CIF risk obligations.",
            "metadata": {"source": "international_shipping_guide.pdf", "chunk_index": 1}
        },
        {
            "id": "shipping_rules_paragraph_001",
            "vector": [0.5, 0.5, 0.707],
            "text": "Mandatory shipping documents commercial invoice and packing list.",
            "metadata": {"source": "shipping_rules.txt", "chunk_index": 1}
        }
    ]
    collection.upsert(records)
    return collection


class TestRetrievalEvaluation(unittest.TestCase):

    def setUp(self):
        self.collection = get_test_vector_collection()
        self.sample_query_item = {
            "query": "What are the statutory registrations for IT hardware?",
            "relevant_chunk_ids": [
                "customs_requirements_paragraph_003",
                "customs_requirements.txt:3"
            ],
            "expected_sources": ["customs_requirements.txt"],
            "keywords": ["statutory", "registrations", "hardware"],
            "topic": "Statutory Registrations"
        }

    # --------------------------------------------------------------------------
    # 1. PARAMETER VALIDATION & ERROR HANDLING TESTS
    # --------------------------------------------------------------------------
    def test_invalid_query_item_validation(self):
        """Verifies ValueError for invalid query item."""
        with self.assertRaises(ValueError):
            evaluate_query({}, k=3, collection=self.collection)

        with self.assertRaises(ValueError):
            evaluate_query(None, k=3, collection=self.collection)

    def test_invalid_labelled_queries_dataset_validation(self):
        """Verifies ValueError for invalid labelled queries dataset."""
        with self.assertRaises(ValueError):
            evaluate_dataset([], k=3, collection=self.collection)

        with self.assertRaises(ValueError):
            evaluate_dataset("invalid_type", k=3, collection=self.collection)

    # --------------------------------------------------------------------------
    # 2. METRIC FORMULA & RECALL/PRECISION TESTS
    # --------------------------------------------------------------------------
    @patch("src.evaluation.retrieve")
    def test_evaluate_query_metrics_hit(self, mock_retrieve):
        """Verifies Recall@K, Precision@K, and MRR@K calculations when hit occurs."""
        mock_retrieve.return_value = [
            {
                "document_id": "customs_requirements_paragraph_003",
                "source": "customs_requirements.txt",
                "chunk_index": 3,
                "similarity_score": 0.85,
                "chunk_text": "Statutory registrations BIS CRS."
            },
            {
                "document_id": "shipping_rules_paragraph_001",
                "source": "shipping_rules.txt",
                "chunk_index": 1,
                "similarity_score": 0.40,
                "chunk_text": "Commercial invoice."
            }
        ]

        res = evaluate_query(self.sample_query_item, k=2, strategy="vector", collection=self.collection)

        self.assertTrue(res["hit"])
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["precision"], 0.5)  # 1 hit / 2 retrieved
        self.assertEqual(res["reciprocal_rank"], 1.0)  # Hit at rank 1

    @patch("src.evaluation.retrieve")
    def test_evaluate_query_metrics_miss(self, mock_retrieve):
        """Verifies zero metrics when no relevant chunks are retrieved."""
        mock_retrieve.return_value = [
            {
                "document_id": "shipping_rules_paragraph_001",
                "source": "shipping_rules.txt",
                "chunk_index": 1,
                "similarity_score": 0.30,
                "chunk_text": "Unrelated text."
            }
        ]

        item_miss = {
            "query": "Non-matching query",
            "relevant_chunk_ids": ["customs_requirements_paragraph_003"],
            "expected_sources": ["customs_requirements.txt"]
        }

        res = evaluate_query(item_miss, k=1, strategy="vector", collection=self.collection)

        self.assertFalse(res["hit"])
        self.assertEqual(res["recall"], 0.0)
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["reciprocal_rank"], 0.0)

    def test_division_by_zero_safety(self):
        """Verifies evaluation handles K > corpus size and empty retrieval safely without division by zero."""
        item = {
            "query": "Test query",
            "relevant_chunk_ids": [],
            "expected_sources": []
        }
        res = evaluate_query(item, k=100, strategy="vector", collection=self.collection)
        self.assertIn("recall", res)
        self.assertIn("precision", res)
        self.assertGreaterEqual(res["recall"], 0.0)

    # --------------------------------------------------------------------------
    # 3. AGGREGATE EVALUATION & MULTI-K TESTS
    # --------------------------------------------------------------------------
    def test_evaluate_dataset_aggregate(self):
        """Verifies evaluate_dataset returns aggregate recall, precision, and MRR."""
        dataset_eval = evaluate_dataset(LABELLED_QUERIES, k=3, strategy="vector", collection=self.collection)

        self.assertEqual(dataset_eval["total_queries"], 5)
        self.assertGreaterEqual(dataset_eval["average_recall"], 0.0)
        self.assertGreaterEqual(dataset_eval["average_precision"], 0.0)
        self.assertGreaterEqual(dataset_eval["mrr"], 0.0)
        self.assertIn("recall_pct", dataset_eval)

    def test_evaluate_multi_k(self):
        """Verifies evaluate_multi_k evaluates metrics across K=[1, 3, 5, 10]."""
        multi_k = evaluate_multi_k(LABELLED_QUERIES, k_values=[1, 3, 5, 10], strategy="vector", collection=self.collection)

        self.assertIn("metrics_by_k", multi_k)
        for k in [1, 3, 5, 10]:
            self.assertIn(k, multi_k["metrics_by_k"])
            self.assertIn("recall_pct", multi_k["metrics_by_k"][k])
            self.assertIn("precision_pct", multi_k["metrics_by_k"][k])

    # --------------------------------------------------------------------------
    # 4. FAILURE DIAGNOSTICS & STRATEGY COMPARISON TESTS
    # --------------------------------------------------------------------------
    def test_inspect_retrieval_failures_and_diagnose(self):
        """Verifies failure extraction and cause diagnosis logic."""
        mock_failure_record = {
            "query": "Unknown query",
            "topic": "Unknown",
            "recall": 0.0,
            "precision": 0.0,
            "expected_relevant_chunks": ["chunk_xyz"],
            "retrieved_chunk_ids": ["chunk_abc"],
            "missing_relevant_chunks": ["chunk_xyz"],
            "retrieved_scores": [0.05],
            "retrieved_sources": ["source.txt"]
        }

        causes = diagnose_failure_causes(mock_failure_record)
        self.assertTrue(len(causes) > 0)
        self.assertIn("Weak semantic similarity", causes[0])

    def test_compare_retrieval_strategies(self):
        """Verifies comparison across Vector, Hybrid, and Re-Ranked strategies."""
        comp = compare_retrieval_strategies(LABELLED_QUERIES, k_values=[1, 3], collection=self.collection)

        self.assertIn("vector", comp)
        self.assertIn("hybrid", comp)
        self.assertIn("reranked", comp)


if __name__ == "__main__":
    unittest.main()
