"""
Unit Tests for Task 3.43: RAG Evaluation & Answer Quality Scoring
===================================================================
Tests score_answer(), judge_expected_points(), judge_grounding(),
check_citations(), evaluate_rag_quality(), failure detection,
and summary report formatting.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag_evaluation import (
    TEST_SET,
    answer_with_citations,
    judge_expected_points,
    judge_grounding,
    check_citations,
    score_answer,
    evaluate_rag_quality,
    format_evaluation_summary
)


class TestRAGEvaluation(unittest.TestCase):

    # --------------------------------------------------------------------------
    # 1. CORRECTNESS SCORING TESTS
    # --------------------------------------------------------------------------
    def test_judge_expected_points_full_match(self):
        """Verifies 1.0 score when all expected points are matched."""
        answer = "Project submission requires a PR link, sample output, and a video explanation."
        expected = ["PR link", "sample output", "video explanation"]
        score = judge_expected_points(answer, expected)
        self.assertEqual(score, 1.0)

    def test_judge_expected_points_partial_match(self):
        """Verifies proportional score when only some points are present."""
        answer = "The submission must include a PR link."
        expected = ["PR link", "sample output", "video explanation"]
        score = judge_expected_points(answer, expected)
        self.assertAlmostEqual(score, 0.3333, places=3)

    def test_judge_expected_points_empty_expected(self):
        """Verifies 1.0 score when no expected points are defined."""
        answer = "Any answer."
        score = judge_expected_points(answer, [])
        self.assertEqual(score, 1.0)

    def test_judge_expected_points_refusal_match(self):
        """Verifies refusal keyphrase matching."""
        answer = "The provided context does not contain enough information to answer this query."
        expected = ["refuse", "not enough information"]
        score = judge_expected_points(answer, expected)
        self.assertEqual(score, 1.0)

    # --------------------------------------------------------------------------
    # 2. GROUNDING SCORING TESTS
    # --------------------------------------------------------------------------
    def test_judge_grounding_refusal_no_context(self):
        """Verifies refusal with missing context yields 1.0 grounding score (anti-hallucination)."""
        answer = "I refuse to answer as there is not enough information in the context."
        score = judge_grounding(answer, retrieved_sources=set(), retrieved_chunks=[])
        self.assertEqual(score, 1.0)

    def test_judge_grounding_hallucination_no_context(self):
        """Verifies fabricated answer without retrieved context yields 0.0 grounding score."""
        answer = "Passcode reset requires visiting account security settings."
        score = judge_grounding(answer, retrieved_sources=set(), retrieved_chunks=[])
        self.assertEqual(score, 0.0)

    def test_judge_grounding_supported_context(self):
        """Verifies high grounding score when answer terms match retrieved chunks."""
        answer = "Commercial invoice and packing list are required shipping documents."
        chunks = [{
            "chunk_text": "All international shipments must include accurate shipping documentation. Required documents may include a commercial invoice and packing list.",
            "source": "shipping_rules.txt"
        }]
        score = judge_grounding(answer, retrieved_sources={"shipping_rules.txt"}, retrieved_chunks=chunks)
        self.assertGreaterEqual(score, 0.7)

    # --------------------------------------------------------------------------
    # 3. CITATION ACCURACY TESTS
    # --------------------------------------------------------------------------
    def test_check_citations_exact_match(self):
        """Verifies 1.0 score when citations match expected sources."""
        citations = {"customs_requirements.txt"}
        expected = {"customs_requirements.txt"}
        score = check_citations(citations, expected)
        self.assertEqual(score, 1.0)

    def test_check_citations_mismatch(self):
        """Verifies 0.0 score when cited source is wrong."""
        citations = {"shipping_rules.txt"}
        expected = {"customs_requirements.txt"}
        score = check_citations(citations, expected)
        self.assertEqual(score, 0.0)

    def test_check_citations_empty_expected_and_citations(self):
        """Verifies 1.0 score when out-of-scope query has no citations."""
        score = check_citations(set(), set())
        self.assertEqual(score, 1.0)

    # --------------------------------------------------------------------------
    # 4. SINGLE EXAMPLE SCORING & PIPELINE MOCK TESTS
    # --------------------------------------------------------------------------
    @patch("src.rag_evaluation.answer_query")
    def test_score_answer_mocked(self, mock_answer_query):
        """Verifies score_answer returns structured evaluation dict."""
        mock_answer_query.return_value = {
            "answer": "IT electronics require Bureau of Indian Standards (BIS) Compulsory Registration Scheme (CRS) compliance.",
            "sources": [{"source": "customs_requirements.txt"}],
            "retrieved_chunks": [{
                "chunk_text": "In particular, IT electronics require Bureau of Indian Standards (BIS) Compulsory Registration Scheme (CRS) compliance.",
                "source": "customs_requirements.txt"
            }],
            "citation_registry": {
                "[Source: customs_requirements.txt]": {"source": "customs_requirements.txt"}
            }
        }

        example = {
            "question": "What mandatory statutory registrations are required for importing IT hardware?",
            "expected_points": ["Bureau of Indian Standards", "BIS", "CRS"],
            "expected_sources": {"customs_requirements.txt"}
        }

        res = score_answer(example)
        self.assertEqual(res["question"], example["question"])
        self.assertEqual(res["correctness"], 1.0)
        self.assertEqual(res["grounding"], 1.0)
        self.assertEqual(res["citation_accuracy"], 1.0)

    # --------------------------------------------------------------------------
    # 5. FULL TEST SET EVALUATION & SUMMARY TESTS
    # --------------------------------------------------------------------------
    @patch("src.rag_evaluation.answer_query")
    def test_evaluate_rag_quality_and_summary(self, mock_answer_query):
        """Verifies evaluate_rag_quality processes test set and generates formatted summary."""
        mock_answer_query.return_value = {
            "answer": "Required documents are commercial invoice and packing list [Source: shipping_rules.txt].",
            "sources": [{"source": "shipping_rules.txt"}],
            "retrieved_chunks": [{
                "chunk_text": "Required documents include commercial invoice and packing list.",
                "source": "shipping_rules.txt"
            }],
            "citation_registry": {}
        }

        mini_test_set = [
            {
                "question": "What shipping documentation is required?",
                "expected_points": ["commercial invoice", "packing list"],
                "expected_sources": {"shipping_rules.txt"}
            }
        ]

        summary = evaluate_rag_quality(mini_test_set)
        self.assertEqual(summary["questions"], 1)
        self.assertIn("avg_correctness", summary)
        self.assertIn("avg_grounding", summary)
        self.assertIn("avg_citation_accuracy", summary)

        report = format_evaluation_summary(summary)
        self.assertIn("SHIPRULE RAG EVALUATION & QUALITY REPORT", report)
        self.assertIn("Total Test Questions Evaluated", report)

    def test_validation_invalid_input(self):
        """Verifies error handling for invalid input parameters."""
        with self.assertRaises(ValueError):
            score_answer({})
        with self.assertRaises(ValueError):
            answer_with_citations("")
        with self.assertRaises(ValueError):
            evaluate_rag_quality([])


if __name__ == "__main__":
    unittest.main()
