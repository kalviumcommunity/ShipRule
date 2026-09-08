"""
Unit Tests for Answer Grounding Validation & Retrieval Comparison
==================================================================
Tests:
TEST 1: Answer is generated from injected context.
TEST 2: Generated answer contains valid source markers.
TEST 3: Unsupported source marker is detected.
TEST 4: No retrieved context produces the fallback.
TEST 5: Unsupported question does not result in a fabricated answer.
TEST 6: With-retrieval execution returns retrieved sources.
TEST 7: Without-retrieval execution does not access the vector DB.
TEST 8: Same question can be executed in both modes (with and without retrieval).
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
from src.rag_pipeline import (
    generate_answer,
    answer_query,
    answer_query_without_retrieval
)
from src.answer_validator import (
    extract_citation_markers,
    validate_answer_sources,
    format_validation_report
)


def get_mock_collection() -> VectorCollection:
    """Creates a mock VectorCollection populated with test documents."""
    col = VectorCollection(name="test_validator_col")
    records = [
        {
            "id": "c1",
            "vector": [1.0, 0.0, 0.0],
            "text": "Commercial invoice and packing list are required for clearance.",
            "metadata": {"source": "shipping_rules.txt", "chunk_index": 1, "section": "Documentation"}
        },
        {
            "id": "c2",
            "vector": [0.0, 1.0, 0.0],
            "text": "Import duty rates are determined based on the 8-digit HS Code classification.",
            "metadata": {"source": "customs_requirements.txt", "chunk_index": 1, "section": "Customs Tariff"}
        }
    ]
    col.upsert(records)
    return col


class TestAnswerGroundingAndValidation(unittest.TestCase):

    def setUp(self):
        self.collection = get_mock_collection()

    # --------------------------------------------------------------------------
    # TEST 1: ANSWER IS GENERATED FROM INJECTED CONTEXT
    # --------------------------------------------------------------------------
    def test_answer_is_generated_from_injected_context(self):
        """TEST 1: Verifies answer text is synthesized strictly from supplied context chunks."""
        context = "[1] Source: shipping_rules.txt\nCommercial invoice and packing list are required."
        sources_info = [{"citation_id": "[1]", "source": "shipping_rules.txt"}]

        res = generate_answer(
            query="What shipping documents are needed?",
            context=context,
            sources_info=sources_info
        )

        self.assertIn("answer", res)
        self.assertIn("commercial invoice", res["answer"].lower())
        self.assertEqual(res["sources"], ["shipping_rules.txt"])

    # --------------------------------------------------------------------------
    # TEST 2: GENERATED ANSWER CONTAINS VALID SOURCE MARKERS
    # --------------------------------------------------------------------------
    def test_generated_answer_contains_valid_source_markers(self):
        """TEST 2: Verifies answer containing valid citation markers passes validation."""
        answer = "Commercial invoice and packing list are required for clearance [1]. Duty is based on HS codes [2]."
        available_markers = ["[1]", "[2]", "[3]"]

        val = validate_answer_sources(answer, available_sources=available_markers)

        self.assertTrue(val["is_valid"])
        self.assertEqual(val["validation_status"], "PASS")
        self.assertEqual(val["grounding_status"], "GROUNDED")
        self.assertEqual(val["used_markers"], ["[1]", ["[2]"][0]])
        self.assertEqual(val["unsupported_markers"], [])

    # --------------------------------------------------------------------------
    # TEST 3: UNSUPPORTED SOURCE MARKER IS DETECTED
    # --------------------------------------------------------------------------
    def test_unsupported_source_marker_is_detected(self):
        """TEST 3: Flags hallucinated or unsupported source markers as validation failures."""
        # Answer references [7] and [9] which are not in available sources [1], [2]
        answer = "According to customs policy [7], all items need approval [9]."
        available_markers = ["[1]", "[2]"]

        val = validate_answer_sources(answer, available_sources=available_markers)

        self.assertFalse(val["is_valid"])
        self.assertEqual(val["validation_status"], "FAIL")
        self.assertEqual(val["grounding_status"], "REVIEW REQUIRED")
        self.assertIn("[7]", val["unsupported_markers"])
        self.assertIn("[9]", val["unsupported_markers"])
        self.assertIn("Unsupported Markers    : [7], [9]", val["report"])

    # --------------------------------------------------------------------------
    # TEST 4: NO RETRIEVED CONTEXT PRODUCES FALLBACK
    # --------------------------------------------------------------------------
    def test_no_retrieved_context_produces_fallback(self):
        """TEST 4: Empty context triggers the deterministic insufficient-context fallback."""
        res = generate_answer(query="What is the speed of sound in aluminum?", context="")

        self.assertIn("insufficient to answer this question", res["answer"].lower())
        self.assertEqual(res["sources"], [])

    # --------------------------------------------------------------------------
    # TEST 5: UNSUPPORTED QUESTION DOES NOT RESULT IN FABRICATED ANSWER
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.generate_query_embedding")
    def test_unsupported_question_does_not_fabricate_answer(self, mock_embed):
        """TEST 5: Out-of-domain query with no matches returns safe insufficient-context fallback."""
        mock_embed.return_value = [0.0, 0.0, 0.0]

        res = answer_query(
            query="What is the customs duty for a product called XYZ-9000 in India?",
            metadata_filter={"source": "non_existent_file.json"},
            collection=self.collection
        )

        self.assertIn("insufficient", res["answer"].lower())
        self.assertEqual(len(res["retrieved_chunks"]), 0)
        self.assertEqual(res["source_validation"]["validation_status"], "PASS")

    # --------------------------------------------------------------------------
    # TEST 6: WITH-RETRIEVAL EXECUTION RETURNS RETRIEVED SOURCES
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.generate_query_embedding")
    def test_with_retrieval_returns_retrieved_sources(self, mock_embed):
        """TEST 6: With-retrieval pipeline queries vector database and returns attached sources."""
        mock_embed.return_value = [1.0, 0.0, 0.0]

        res = answer_query(
            query="What documents are required?",
            candidate_k=2,
            final_k=1,
            collection=self.collection
        )

        self.assertTrue(len(res["retrieved_chunks"]) > 0)
        self.assertIn("shipping_rules.txt", res["sources"])
        self.assertIn("token_usage", res)
        self.assertEqual(res["source_validation"]["validation_status"], "PASS")

    # --------------------------------------------------------------------------
    # TEST 7: WITHOUT-RETRIEVAL EXECUTION DOES NOT ACCESS VECTOR DB
    # --------------------------------------------------------------------------
    def test_without_retrieval_does_not_access_vector_db(self):
        """TEST 7: Baseline non-retrieval execution skips vector search completely."""
        res = answer_query_without_retrieval(query="shipment rules in india")

        self.assertEqual(res["retrieved_chunks_count"], 0)
        self.assertEqual(res["retrieved_chunks"], [])
        self.assertEqual(res["context"], "NONE")
        self.assertIn("insufficient", res["answer"].lower())
        self.assertEqual(res["source_validation"]["grounding_status"], "NO RETRIEVAL CONTEXT")

    # --------------------------------------------------------------------------
    # TEST 8: SAME QUESTION EXECUTABLE IN BOTH MODES
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.generate_query_embedding")
    def test_same_question_executed_in_both_modes(self, mock_embed):
        """TEST 8: Executes same question with and without retrieval and compares results."""
        mock_embed.return_value = [1.0, 0.0, 0.0]
        question = "shipment rules in india"

        # Run A: With Retrieval
        run_a = answer_query(query=question, collection=self.collection)

        # Run B: Without Retrieval
        run_b = answer_query_without_retrieval(query=question)

        # Comparisons
        self.assertTrue(len(run_a["retrieved_chunks"]) > 0)
        self.assertEqual(len(run_b["retrieved_chunks"]), 0)
        self.assertEqual(run_a["source_validation"]["grounding_status"], "GROUNDED")
        self.assertEqual(run_b["source_validation"]["grounding_status"], "NO RETRIEVAL CONTEXT")


if __name__ == "__main__":
    unittest.main()
