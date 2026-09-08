"""
Unit Tests for Retrieval Quality Guardrail Module
==================================================
Tests:
TEST 1: Empty retrieval returns is_sufficient == False (reason: 'no_results').
TEST 2: All chunks below relevance threshold returns is_sufficient == False (reason: 'no_relevant_context').
TEST 3: Too few relevant chunks returns is_sufficient == False (reason: 'too_few_relevant_chunks').
TEST 4: Strong retrieval returns is_sufficient == True (reason: 'strong_retrieval').
TEST 5: Correct score/distance semantics (lower distance is better, distance <= 1.35).
TEST 6: Weak retrieval prevents LLM call (answer generation is intercepted).
TEST 7: Strong retrieval still calls existing grounded answer generation.
TEST 8: Refusal response contains no fabricated citations ([1], [2], etc.).
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import re

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.retrieval_guardrail import (
    check_retrieval_quality,
    is_chunk_relevant,
    get_safe_refusal_response,
    format_guardrail_report,
    DEFAULT_MAX_DISTANCE_THRESHOLD,
    DEFAULT_MIN_RELEVANT_CHUNKS
)
from src.rag_pipeline import answer_query
from src.indexing import VectorCollection


class TestRetrievalGuardrail(unittest.TestCase):

    def setUp(self):
        # Sample mock chunks with varying distances/scores
        self.strong_chunk_1 = {
            "chunk_text": "Customs duties apply based on HS code 8703 in Italy.",
            "source": "customs_italy.txt",
            "chunk_index": 1,
            "distance": 0.45,
            "similarity_score": 0.85
        }
        self.strong_chunk_2 = {
            "chunk_text": "Required documents include EUR.1 and commercial invoice.",
            "source": "customs_italy.txt",
            "chunk_index": 2,
            "distance": 0.75,
            "similarity_score": 0.72
        }
        self.weak_chunk_1 = {
            "chunk_text": "Unrelated topic regarding warehouse management system.",
            "source": "general_guide.txt",
            "chunk_index": 5,
            "distance": 1.85,
            "similarity_score": 0.10
        }
        self.weak_chunk_2 = {
            "chunk_text": "Random logistics overview without specific customs rules.",
            "source": "general_guide.txt",
            "chunk_index": 6,
            "distance": 1.95,
            "similarity_score": 0.05
        }

    # --------------------------------------------------------------------------
    # TEST 1: EMPTY RETRIEVAL
    # --------------------------------------------------------------------------
    def test_empty_retrieval(self):
        """TEST 1: Empty retrieval [] classifies retrieval as weak ('no_results')."""
        result = check_retrieval_quality(retrieved_chunks=[])
        self.assertFalse(result["is_sufficient"])
        self.assertEqual(result["reason"], "no_results")
        self.assertEqual(result["retrieved_count"], 0)
        self.assertEqual(result["relevant_count"], 0)
        self.assertEqual(result["decision"], "REFUSE")

    # --------------------------------------------------------------------------
    # TEST 2: ALL CHUNKS BELOW RELEVANCE THRESHOLD
    # --------------------------------------------------------------------------
    def test_all_chunks_below_relevance_threshold(self):
        """TEST 2: All chunks with distance > 1.35 classify as weak ('no_relevant_context')."""
        weak_chunks = [self.weak_chunk_1, self.weak_chunk_2]
        result = check_retrieval_quality(
            retrieved_chunks=weak_chunks,
            max_distance_threshold=1.35,
            min_relevant_chunks=1
        )
        self.assertFalse(result["is_sufficient"])
        self.assertEqual(result["reason"], "no_relevant_context")
        self.assertEqual(result["retrieved_count"], 2)
        self.assertEqual(result["relevant_count"], 0)
        self.assertEqual(result["decision"], "REFUSE")

    # --------------------------------------------------------------------------
    # TEST 3: TOO FEW RELEVANT CHUNKS
    # --------------------------------------------------------------------------
    def test_too_few_relevant_chunks(self):
        """TEST 3: When MIN_RELEVANT_CHUNKS=2 and only 1 chunk passes, classifies as weak."""
        mixed_chunks = [self.strong_chunk_1, self.weak_chunk_1]
        result = check_retrieval_quality(
            retrieved_chunks=mixed_chunks,
            max_distance_threshold=1.35,
            min_relevant_chunks=2
        )
        self.assertFalse(result["is_sufficient"])
        self.assertEqual(result["reason"], "too_few_relevant_chunks")
        self.assertEqual(result["retrieved_count"], 2)
        self.assertEqual(result["relevant_count"], 1)
        self.assertEqual(result["decision"], "REFUSE")

    # --------------------------------------------------------------------------
    # TEST 4: STRONG RETRIEVAL
    # --------------------------------------------------------------------------
    def test_strong_retrieval(self):
        """TEST 4: Enough chunks pass the threshold -> is_sufficient == True ('strong_retrieval')."""
        strong_chunks = [self.strong_chunk_1, self.strong_chunk_2]
        result = check_retrieval_quality(
            retrieved_chunks=strong_chunks,
            max_distance_threshold=1.35,
            min_relevant_chunks=1
        )
        self.assertTrue(result["is_sufficient"])
        self.assertEqual(result["reason"], "strong_retrieval")
        self.assertEqual(result["retrieved_count"], 2)
        self.assertEqual(result["relevant_count"], 2)
        self.assertEqual(result["decision"], "ALLOW")

    # --------------------------------------------------------------------------
    # TEST 5: CORRECT SCORE / DISTANCE SEMANTICS
    # --------------------------------------------------------------------------
    def test_correct_score_distance_semantics(self):
        """TEST 5: Confirms distance <= 1.35 is relevant and distance > 1.35 is rejected."""
        # Distance 0.45 <= 1.35 -> True
        self.assertTrue(is_chunk_relevant({"distance": 0.45}, max_distance_threshold=1.35))
        # Distance 1.35 <= 1.35 -> True (boundary)
        self.assertTrue(is_chunk_relevant({"distance": 1.35}, max_distance_threshold=1.35))
        # Distance 1.36 > 1.35 -> False
        self.assertFalse(is_chunk_relevant({"distance": 1.36}, max_distance_threshold=1.35))
        # Distance 1.85 > 1.35 -> False
        self.assertFalse(is_chunk_relevant({"distance": 1.85}, max_distance_threshold=1.35))

    # --------------------------------------------------------------------------
    # TEST 6: WEAK RETRIEVAL PREVENTS LLM CALL
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.generate_answer")
    @patch("src.rag_pipeline.retrieve_context")
    def test_weak_retrieval_prevents_llm_call(self, mock_retrieve, mock_gen_answer):
        """TEST 6: Mocked LLM answer generation is NOT called when retrieval is weak."""
        # Empty retrieval
        mock_retrieve.return_value = ([], {"retrieval_ms": 1.0, "reranking_ms": 1.0})

        response = answer_query("What are the dolphin regulations in Antarctica?")

        # Assert LLM generation was never called
        mock_gen_answer.assert_not_called()
        self.assertFalse(response.get("llm_called", True))
        self.assertEqual(response.get("guardrail_decision"), "REFUSE")
        self.assertFalse(response["guardrail"]["is_sufficient"])
        self.assertIn("insufficient", response["answer"].lower())

    # --------------------------------------------------------------------------
    # TEST 7: STRONG RETRIEVAL STILL CALLS EXISTING ANSWER GENERATION
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.generate_answer")
    @patch("src.rag_pipeline.retrieve_context")
    def test_strong_retrieval_calls_answer_generation(self, mock_retrieve, mock_gen_answer):
        """TEST 7: Strong retrieval passes guardrail and proceeds to grounded answer generation."""
        mock_retrieve.return_value = (
            [self.strong_chunk_1, self.strong_chunk_2],
            {"retrieval_ms": 1.0, "reranking_ms": 1.0}
        )
        mock_gen_answer.return_value = {
            "answer": "According to customs regulations, duty applies based on HS code 8703 in Italy. [1]",
            "sources": ["customs_italy.txt"]
        }

        response = answer_query("What is the HS code for cars in Italy?")

        # Assert LLM generation was called
        mock_gen_answer.assert_called_once()
        self.assertTrue(response.get("llm_called", False))
        self.assertEqual(response.get("guardrail_decision"), "ALLOW")
        self.assertTrue(response["guardrail"]["is_sufficient"])
        self.assertIn("Italy", response["answer"])

    # --------------------------------------------------------------------------
    # TEST 8: REFUSAL CONTAINS NO FABRICATED CITATION
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.retrieve_context")
    def test_refusal_contains_no_fabricated_citation(self, mock_retrieve):
        """TEST 8: Refusal response contains no citation markers like [1], [2] or fake sources."""
        mock_retrieve.return_value = (
            [self.weak_chunk_1],
            {"retrieval_ms": 1.0, "reranking_ms": 1.0}
        )

        response = answer_query("Dolphin import in Antarctica")

        # Refusal text should have NO bracketed citation markers [1], [2]
        citation_matches = re.findall(r"\[\d+\]", response["answer"])
        self.assertEqual(citation_matches, [])
        self.assertEqual(response["sources"], [])
        self.assertEqual(response["retrieved_chunks"], [])
        self.assertFalse(response["guardrail"]["is_sufficient"])
        self.assertFalse(response["llm_called"])


if __name__ == "__main__":
    unittest.main()
