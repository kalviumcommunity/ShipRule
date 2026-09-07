"""
Unit and Integration Tests for Phase 3.37: End-to-End RAG Pipeline Module
==========================================================================
Tests modular pipeline stages (embed_query, retrieve_context, assemble_context,
generate_answer, answer_query), metadata filtering, hybrid search integration,
re-ranking integration, empty context fallback, source citation matching,
timing metrics, and grounding / anti-hallucination safeguards.
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
    embed_query,
    retrieve_context,
    assemble_context,
    generate_answer,
    answer_query,
)


def get_mock_vector_collection() -> VectorCollection:
    """Helper to construct a populated mock VectorCollection for pipeline testing."""
    collection = VectorCollection(name="test_pipeline_col")
    records = [
        {
            "id": "chunk_01",
            "vector": [1.0, 0.0, 0.0],
            "text": "All international shipments must include accurate shipping documentation. Required documents may include a commercial invoice and packing list.",
            "metadata": {"source": "shipping_rules.txt", "chunk_index": 1, "section": "Documentation"}
        },
        {
            "id": "chunk_02",
            "vector": [0.0, 1.0, 0.0],
            "text": "Certain regulated product categories including information technology hardware require BIS registration certificate.",
            "metadata": {"source": "customs_requirements.txt", "chunk_index": 3, "section": "Statutory Registrations"}
        },
        {
            "id": "chunk_03",
            "vector": [0.0, 0.0, 1.0],
            "text": "Incoterms 2020 define FOB and CIF shipping responsibilities between seller and buyer.",
            "metadata": {"source": "international_shipping_guide.pdf", "chunk_index": 1, "section": "Incoterms"}
        }
    ]
    collection.upsert(records)
    return collection


class TestEndToEndRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.collection = get_mock_vector_collection()

    # --------------------------------------------------------------------------
    # TEST 4: EMPTY QUERY VALIDATION ERROR
    # --------------------------------------------------------------------------
    def test_empty_query_raises_validation_error(self):
        """Verifies empty or None queries raise ValueError across all pipeline functions."""
        with self.assertRaises(ValueError):
            embed_query("")

        with self.assertRaises(ValueError):
            retrieve_context("", collection=self.collection)

        with self.assertRaises(ValueError):
            generate_answer("", context="some context")

        with self.assertRaises(ValueError):
            answer_query("", collection=self.collection)

    # --------------------------------------------------------------------------
    # TEST 1: VALID QUERY -> RELEVANT CONTEXT -> ANSWER
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.generate_query_embedding")
    def test_valid_query_end_to_end_flow(self, mock_embed):
        """Test 1: Valid query retrieves context, assembles citations, and returns grounded answer."""
        mock_embed.return_value = [1.0, 0.0, 0.0]

        res = answer_query(
            query="What shipping documentation is required?",
            candidate_k=3,
            final_k=2,
            collection=self.collection,
            debug=True
        )

        self.assertIn("answer", res)
        self.assertIn("sources", res)
        self.assertTrue(len(res["retrieved_chunks"]) > 0)
        self.assertIn("shipping_rules.txt", res["sources"])
        self.assertIn("timing", res)
        self.assertIn("debug_info", res)

    # --------------------------------------------------------------------------
    # TEST 2: QUERY WITH METADATA FILTER -> FILTERED CONTEXT
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.generate_query_embedding")
    def test_query_with_metadata_filter(self, mock_embed):
        """Test 2: Query with metadata filter restricts context strictly to matching source."""
        mock_embed.return_value = [0.0, 1.0, 0.0]

        res = answer_query(
            query="What BIS registration is required?",
            metadata_filter={"source": "customs_requirements.txt"},
            collection=self.collection
        )

        for chunk in res["retrieved_chunks"]:
            self.assertEqual(chunk["source"], "customs_requirements.txt")
        self.assertEqual(res["sources"], ["customs_requirements.txt"])

    # --------------------------------------------------------------------------
    # TEST 3: QUERY WITH NO RESULTS -> SAFE FALLBACK
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.generate_query_embedding")
    def test_empty_retrieval_safe_fallback(self, mock_embed):
        """Test 3: Query with no matching results returns safe response without calling LLM."""
        mock_embed.return_value = [1.0, 0.0, 0.0]

        # Overly strict filter returns zero chunks
        res = answer_query(
            query="What documents are needed?",
            metadata_filter={"source": "non_existent_doc.txt"},
            collection=self.collection
        )

        self.assertIn("could not find enough relevant information", res["answer"])
        self.assertEqual(res["sources"], [])
        self.assertEqual(res["retrieved_chunks"], [])

    # --------------------------------------------------------------------------
    # TEST 5: LLM RECEIVES ONLY RETRIEVED CONTEXT
    # --------------------------------------------------------------------------
    def test_assemble_context_preserves_only_retrieved_chunks(self):
        """Test 5: Context assembly converts retrieved chunks into deterministic cited text."""
        chunks = [
            {
                "rank": 1,
                "source": "shipping_rules.txt",
                "chunk_index": 1,
                "chunk_text": "Commercial invoice and packing list required."
            }
        ]

        text, sources = assemble_context(chunks)

        self.assertIn("[1] Source: shipping_rules.txt | Chunk Index: 1", text)
        self.assertIn("Commercial invoice and packing list required.", text)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["citation_id"], "[1]")

    # --------------------------------------------------------------------------
    # TEST 6: SOURCES IN FINAL RESULT MATCH ACTUAL RETRIEVED CHUNKS
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.generate_query_embedding")
    def test_sources_match_retrieved_chunks(self, mock_embed):
        """Test 6: Returned sources match sources of retrieved chunks."""
        mock_embed.return_value = [0.0, 0.0, 1.0]

        res = answer_query(
            query="Explain Incoterms FOB and CIF rules",
            candidate_k=2,
            final_k=1,
            collection=self.collection
        )

        retrieved_sources = [c["source"] for c in res["retrieved_chunks"]]
        for src in res["sources"]:
            self.assertIn(src, retrieved_sources)

    # --------------------------------------------------------------------------
    # TEST 7: EXISTING RE-RANKING WORKS INSIDE PIPELINE
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.generate_query_embedding")
    def test_reranking_integration_in_pipeline(self, mock_embed):
        """Test 7: Re-ranking scores and updates chunk positions inside pipeline."""
        mock_embed.return_value = [1.0, 0.0, 0.0]

        chunks, timing = retrieve_context(
            query="BIS registration IT hardware",
            candidate_k=3,
            final_k=2,
            use_reranking=True,
            collection=self.collection
        )

        self.assertIn("reranking_ms", timing)
        self.assertIn("rerank_score", chunks[0])

    # --------------------------------------------------------------------------
    # TEST 8: EXISTING HYBRID SEARCH WORKS INSIDE PIPELINE
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.generate_query_embedding")
    def test_hybrid_search_integration_in_pipeline(self, mock_embed):
        """Test 8: Hybrid search mode executes inside retrieve_context."""
        mock_embed.return_value = [1.0, 0.0, 0.0]

        chunks, timing = retrieve_context(
            query="commercial invoice packing list",
            candidate_k=3,
            final_k=2,
            use_hybrid=True,
            collection=self.collection
        )

        self.assertTrue(len(chunks) > 0)
        self.assertIn("retrieval_ms", timing)

    # --------------------------------------------------------------------------
    # TEST 9: GROUNDING / HALLUCINATION TEST FOR UNANSWERABLE QUESTION
    # --------------------------------------------------------------------------
    @patch("src.rag_pipeline.generate_query_embedding")
    def test_unanswerable_query_grounding_safeguard(self, mock_embed):
        """Test 9: Unanswerable or empty context question returns safe fallback without hallucination."""
        mock_embed.return_value = [0.0, 0.0, 0.0]

        # Call generate_answer directly with empty context
        gen_res = generate_answer(
            query="What is the speed of light in vacuum?",
            context=""
        )

        self.assertIn("could not find enough relevant information", gen_res["answer"])
        self.assertEqual(gen_res["sources"], [])


if __name__ == "__main__":
    unittest.main()
