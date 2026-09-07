"""
Unit Tests for Embedding Quality Checks & Sanity Tests Module
===============================================================
Tests retrieval test case evaluation, top-k ranking calculation, expected source
rank tracking, pipeline sanity checks (dimension consistency, metadata validity,
duplicate chunk detection, NaN value safety), report formatting, and surprising case warnings.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sanity_checker import (
    run_pipeline_sanity_checks,
    evaluate_retrieval_test_case,
    run_embedding_sanity_tests,
    format_sanity_report,
    POSSIBLE_FAILURE_CAUSES,
)


class TestSanityChecker(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            {
                "embedding_id": "emb_001",
                "chunk_id": "account_01",
                "source": "account-guide.md",
                "chunk_index": 1,
                "chunk_text": "To reset your password, click on Forgot Password on the login screen.",
                "embedding_model": "text-embedding-3-small",
                "embedding": [1.0, 0.0, 0.0]
            },
            {
                "embedding_id": "emb_002",
                "chunk_id": "campus_01",
                "source": "campus-guide.md",
                "chunk_index": 1,
                "chunk_text": "The cafeteria menu changes weekly on Monday morning.",
                "embedding_model": "text-embedding-3-small",
                "embedding": [0.0, 1.0, 0.0]
            },
            {
                "embedding_id": "emb_003",
                "chunk_id": "shipping_01",
                "source": "shipping_rules.txt",
                "chunk_index": 1,
                "chunk_text": "Commercial invoices are required for international customs clearance.",
                "embedding_model": "text-embedding-3-small",
                "embedding": [0.7071, 0.7071, 0.0]
            }
        ]

    @patch("src.sanity_checker.generate_query_embedding")
    def test_evaluate_retrieval_test_case_pass(self, mock_gen_query_embed):
        """1. Test that when expected source is top result, status is PASS and rank is 1."""
        # Query matches account-guide.md vector [1.0, 0.0, 0.0]
        mock_gen_query_embed.return_value = [1.0, 0.0, 0.0]

        test_case = {
            "query": "How can a learner reset their password?",
            "expected_source": "account-guide.md"
        }

        res = evaluate_retrieval_test_case(
            test_case=test_case,
            candidate_chunks=self.sample_chunks,
            model="text-embedding-3-small",
            top_k=3
        )

        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["expected_rank"], 1)
        self.assertEqual(res["top_source"], "account-guide.md")
        self.assertAlmostEqual(res["top_score"], 1.0, places=3)
        self.assertTrue(res["in_top_k"])

    @patch("src.sanity_checker.generate_query_embedding")
    def test_evaluate_retrieval_test_case_fail_and_top_k_ranking(self, mock_gen_query_embed):
        """2. Test that when expected source is not top result, status is FAIL and expected_rank is accurate."""
        # Query vector closer to campus-guide.md [0.0, 1.0, 0.0]
        mock_gen_query_embed.return_value = [0.1, 0.9, 0.0]

        test_case = {
            "query": "How can a learner reset their password?",
            "expected_source": "account-guide.md"
        }

        res = evaluate_retrieval_test_case(
            test_case=test_case,
            candidate_chunks=self.sample_chunks,
            model="text-embedding-3-small",
            top_k=3
        )

        self.assertEqual(res["status"], "FAIL")
        self.assertEqual(res["top_source"], "campus-guide.md")
        self.assertIn(res["expected_rank"], [2, 3])
        self.assertTrue(res["in_top_k"])

    def test_pipeline_sanity_checks_clean_data(self):
        """3. Test pipeline sanity checks pass on complete and valid chunks."""
        query_vec = [1.0, 0.0, 0.0]
        res = run_pipeline_sanity_checks(
            candidate_chunks=self.sample_chunks,
            query_vector=query_vec,
            expected_model="text-embedding-3-small"
        )

        self.assertEqual(res["status"], "PASSED")
        self.assertEqual(len(res["issues"]), 0)
        self.assertEqual(res["metrics"]["chunk_count"], 3)
        self.assertEqual(res["metrics"]["vector_dimension"], 3)

    def test_pipeline_sanity_checks_dimension_mismatch(self):
        """4. Test detection of dimension mismatch between query vector and document vectors."""
        query_vec_5d = [1.0, 0.0, 0.0, 0.0, 0.0]  # 5-dim query vs 3-dim doc
        res = run_pipeline_sanity_checks(
            candidate_chunks=self.sample_chunks,
            query_vector=query_vec_5d,
            expected_model="text-embedding-3-small"
        )

        self.assertEqual(res["status"], "FAILED")
        self.assertTrue(any("dimension mismatch" in issue.lower() for issue in res["issues"]))

    def test_pipeline_sanity_checks_missing_source_metadata(self):
        """5. Test detection of missing source metadata."""
        corrupted_chunks = [
            {
                "embedding_id": "bad_01",
                "chunk_text": "Missing source metadata chunk",
                "source": "",
                "embedding": [1.0, 0.0, 0.0]
            }
        ]

        res = run_pipeline_sanity_checks(candidate_chunks=corrupted_chunks)
        self.assertEqual(res["status"], "FAILED")
        self.assertTrue(any("missing valid source metadata" in issue.lower() for issue in res["issues"]))

    def test_pipeline_sanity_checks_duplicate_chunk_detection(self):
        """6. Test detection of duplicate chunk text and duplicate vectors."""
        duplicate_chunks = [
            {
                "embedding_id": "orig_01",
                "source": "doc1.txt",
                "chunk_text": "Duplicate content line.",
                "embedding": [1.0, 0.0, 0.0]
            },
            {
                "embedding_id": "dup_01",
                "source": "doc2.txt",
                "chunk_text": "Duplicate content line.",
                "embedding": [1.0, 0.0, 0.0]
            }
        ]

        res = run_pipeline_sanity_checks(candidate_chunks=duplicate_chunks)
        self.assertEqual(len(res["warnings"]), 2)
        self.assertTrue(any("duplicate chunk text" in w.lower() for w in res["warnings"]))
        self.assertTrue(any("duplicate embedding vector" in w.lower() for w in res["warnings"]))

    def test_format_sanity_report_structure(self):
        """7. Test formatted report string output contains required headers, stats, and test details."""
        report_data = {
            "summary": {
                "total_tests": 2,
                "passed": 2,
                "failed": 0,
                "top_k_configured": 3,
                "embedding_model": "text-embedding-3-small"
            },
            "pipeline_check": {"status": "PASSED", "issues": [], "warnings": []},
            "test_results": [
                {
                    "query": "How can a learner reset their password?",
                    "expected_source": "account-guide.md",
                    "top_source": "account-guide.md",
                    "top_score": 0.82,
                    "expected_rank": 1,
                    "status": "PASS"
                },
                {
                    "query": "When does the cafeteria menu change?",
                    "expected_source": "campus-guide.md",
                    "top_source": "campus-guide.md",
                    "top_score": 0.79,
                    "expected_rank": 1,
                    "status": "PASS"
                }
            ]
        }

        report_text = format_sanity_report(report_data)

        self.assertIn("Embedding Sanity Report", report_text)
        self.assertIn("Tests: 2", report_text)
        self.assertIn("Passed: 2", report_text)
        self.assertIn("Failed: 0", report_text)
        self.assertIn("Expected source: account-guide.md", report_text)
        self.assertIn("Top score: 0.82", report_text)
        self.assertIn("Status: PASS", report_text)

    def test_format_sanity_report_failing_surprising_case(self):
        """8. Test that failing tests display the FAIL / SURPRISING CASE section with possible causes."""
        report_data = {
            "summary": {
                "total_tests": 1,
                "passed": 0,
                "failed": 1,
                "top_k_configured": 3,
                "embedding_model": "text-embedding-3-small"
            },
            "pipeline_check": {"status": "PASSED", "issues": [], "warnings": []},
            "test_results": [
                {
                    "query": "When does cafeteria menu change?",
                    "expected_source": "campus-guide.md",
                    "top_source": "account-guide.md",
                    "top_score": 0.45,
                    "expected_rank": 2,
                    "status": "FAIL"
                }
            ]
        }

        report_text = format_sanity_report(report_data)

        self.assertIn("FAIL / SURPRISING CASE", report_text)
        self.assertIn("Expected: campus-guide.md", report_text)
        self.assertIn("Retrieved: account-guide.md", report_text)
        self.assertIn("Possible causes:", report_text)
        for cause in POSSIBLE_FAILURE_CAUSES:
            self.assertIn(f"- {cause}", report_text)


if __name__ == "__main__":
    unittest.main()
