"""
Unit Tests for Task 3.34: Retrieval Relevance Tuning & Evaluation Module
==========================================================================
Tests evaluation dataset validity, single configuration evaluation calculations,
Hit Rate %, MRR, Recall@K, Average Rank, score threshold filtering effects,
metadata filter evaluation, configuration comparison, and auto-recommendation logic.
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
    EVALUATION_DATASET,
    EVALUATION_CONFIGURATIONS,
    evaluate_retrieval,
    compare_configurations,
    format_manual_inspection_view,
)


def get_sample_vector_collection() -> VectorCollection:
    """Helper to construct a pre-populated mock VectorCollection with known chunks and vectors."""
    collection = VectorCollection(name="test_eval_col")
    records = [
        {
            "id": "CDLP-REG-01",
            "vector": [1.0, 0.0, 0.0, 0.0],
            "text": "Mandatory statutory registrations for IT electronics include Bureau of Indian Standards (BIS) CRS compliance.",
            "metadata": {
                "source": "customs_requirements.txt",
                "chunk_index": 3,
                "section": "Statutory Registrations",
                "document_type": "txt"
            }
        },
        {
            "id": "CDLP-GUIDE-01",
            "vector": [0.0, 1.0, 0.0, 0.0],
            "text": "Standard Incoterms 2020 define division of costs and risks between buyer and seller: FOB, CIF, DDP.",
            "metadata": {
                "source": "international_shipping_guide.pdf",
                "chunk_index": 1,
                "section": "Incoterms 2020",
                "document_type": "pdf"
            }
        },
        {
            "id": "CDLP-RULES-01",
            "vector": [0.0, 0.0, 1.0, 0.0],
            "text": "All international shipments require accurate shipping documentation including commercial invoice and packing list.",
            "metadata": {
                "source": "shipping_rules.txt",
                "chunk_index": 1,
                "section": "Documentation",
                "document_type": "txt"
            }
        }
    ]
    collection.upsert(records)
    return collection


class TestRetrievalEvaluation(unittest.TestCase):

    def setUp(self):
        self.collection = get_sample_vector_collection()

    # --------------------------------------------------------------------------
    # 1. EVALUATION DATASET STRUCTURE TESTS
    # --------------------------------------------------------------------------

    def test_evaluation_dataset_contains_at_least_5_real_queries(self):
        """Verifies dataset has at least 5 realistic test cases with real corpus sources."""
        self.assertGreaterEqual(len(EVALUATION_DATASET), 5)
        known_sources = {"customs_requirements.txt", "international_shipping_guide.pdf", "shipping_rules.txt"}

        for item in EVALUATION_DATASET:
            self.assertIn("query", item)
            self.assertIn("expected_source", item)
            self.assertIn(item["expected_source"], known_sources)

    # --------------------------------------------------------------------------
    # 2. SINGLE CONFIGURATION EVALUATION TESTS
    # --------------------------------------------------------------------------

    @patch("src.retrieval.generate_query_embedding")
    def test_evaluate_retrieval_baseline_metrics(self, mock_embed):
        """Verifies evaluate_retrieval calculates Hits, Misses, Hit Rate %, MRR, and Avg Rank."""
        mock_embed.return_value = [1.0, 0.0, 0.0, 0.0]

        setting = {
            "name": "baseline_k3",
            "k": 3,
            "filter": None,
            "min_score": 0.0,
            "use_hybrid": False
        }

        test_cases = [
            {
                "query": "Statutory registrations?",
                "expected_source": "customs_requirements.txt"
            }
        ]

        eval_res = evaluate_retrieval(setting, test_dataset=test_cases, collection=self.collection)

        self.assertEqual(eval_res["total_queries"], 1)
        self.assertEqual(eval_res["hits"], 1)
        self.assertEqual(eval_res["misses"], 0)
        self.assertEqual(eval_res["hit_rate_pct"], 100.0)
        self.assertEqual(eval_res["mrr"], 1.0)
        self.assertEqual(eval_res["avg_expected_rank"], 1.0)

    @patch("src.retrieval.generate_query_embedding")
    def test_evaluate_retrieval_score_threshold_effect(self, mock_embed):
        """Verifies high score threshold filters out lower similarity chunks."""
        mock_embed.return_value = [1.0, 0.0, 0.0, 0.0]

        high_thresh_setting = {
            "name": "high_threshold",
            "k": 3,
            "filter": None,
            "min_score": 0.99,  # Only exact match [1, 0, 0, 0] will pass
            "use_hybrid": False
        }

        eval_res = evaluate_retrieval(high_thresh_setting, test_dataset=EVALUATION_DATASET[:1], collection=self.collection)
        q_eval = eval_res["query_evaluations"][0]

        # Only 1 chunk should pass min_score=0.99
        self.assertEqual(q_eval["retrieved_count"], 1)
        self.assertEqual(q_eval["returned_sources"], ["customs_requirements.txt"])

    # --------------------------------------------------------------------------
    # 3. CONFIGURATION COMPARISON & RECOMMENDATION ENGINE TESTS
    # --------------------------------------------------------------------------

    @patch("src.retrieval.generate_query_embedding")
    def test_compare_configurations_compiles_summary_and_recommendation(self, mock_embed):
        """Verifies compare_configurations evaluates all settings and produces valid recommendation."""
        mock_embed.return_value = [1.0, 0.0, 0.0, 0.0]

        test_configs = [
            {"name": "baseline_k3", "k": 3, "filter": None, "min_score": 0.0, "use_hybrid": False},
            {"name": "baseline_k5", "k": 5, "filter": None, "min_score": 0.0, "use_hybrid": False}
        ]

        all_evals, summary_table, rec = compare_configurations(
            configurations=test_configs,
            test_dataset=EVALUATION_DATASET[:2],
            collection=self.collection
        )

        self.assertEqual(len(all_evals), 2)
        self.assertEqual(len(summary_table["rows"]), 2)
        self.assertIn("recommended_configuration", rec)
        self.assertIn("reasons", rec)
        self.assertIn("trade_offs", rec)

    # --------------------------------------------------------------------------
    # 4. ERROR HANDLING & FORMATTER TESTS
    # --------------------------------------------------------------------------

    def test_invalid_setting_or_dataset_raises_value_error(self):
        """Verifies invalid setting or dataset raises ValueError."""
        with self.assertRaises(ValueError):
            evaluate_retrieval(None, collection=self.collection)

        with self.assertRaises(ValueError):
            evaluate_retrieval({"name": "invalid"}, test_dataset=[], collection=self.collection)

    def test_format_manual_inspection_view(self):
        """Verifies format_manual_inspection_view produces readable ASCII inspection view."""
        sample_eval = {
            "label": "Baseline Top-K=3",
            "query_evaluations": [
                {
                    "query": "Test query?",
                    "topic": "Testing",
                    "expected_source": "customs_requirements.txt",
                    "hit": True,
                    "inspection_chunks": [
                        {
                            "rank": 1,
                            "score": 0.95,
                            "source": "customs_requirements.txt",
                            "section": "General",
                            "chunk_index": 1,
                            "text": "Text preview",
                            "judgement_hint": "Excellent"
                        }
                    ]
                }
            ]
        }

        view = format_manual_inspection_view(sample_eval)
        self.assertIn("MANUAL RELEVANCE INSPECTION", view)
        self.assertIn("Query: Test query?", view)
        self.assertIn("Status: [HIT]", view)
        self.assertIn("Judgement: [Excellent]", view)


if __name__ == "__main__":
    unittest.main()
