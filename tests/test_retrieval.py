"""
Unit Tests for Task 3.33: Metadata Filtering & Hybrid Search Module
=====================================================================
Tests query validation, k validation, metadata filtering precision,
keyword_score() match calculations, hybrid_rank() weighted score combinations,
hybrid_retrieve() pipeline, exact-match terminology scenarios, comparative outputs,
and comprehensive error handling.
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
    keyword_score,
    hybrid_rank,
    hybrid_retrieve,
    format_retrieval_output,
    format_filtered_vs_unfiltered_output,
    format_hybrid_output,
    load_indexed_vector_collection,
    run_retrieval_demonstration,
)


def get_sample_vector_collection() -> VectorCollection:
    """Helper to construct a pre-populated mock VectorCollection with known metadata and text."""
    collection = VectorCollection(name="test_hybrid_col")
    records = [
        {
            "id": "CDLP-ACC-01",
            "vector": [1.0, 0.0, 0.0, 0.0],
            "text": "To reset your learner account password, navigate to Account Settings and click 'Forgot Password'.",
            "metadata": {
                "source": "account-guide.md",
                "chunk_index": 1,
                "section": "Account access",
                "document_type": "md",
                "country": "Global"
            }
        },
        {
            "id": "CDLP-IN-8471-01",
            "vector": [0.0, 1.0, 0.0, 0.0],
            "text": "Customs rules for shipping laptops under HS Code 8471.30 to India require BIS Registration Certificate and DGFT Import License.",
            "metadata": {
                "source": "customs_reg_india.json",
                "chunk_index": 1,
                "section": "Electronics",
                "document_type": "json",
                "country": "India",
                "hs_code": "8471.30"
            }
        },
        {
            "id": "CDLP-DOC-01",
            "vector": [0.0, 0.7071, 0.7071, 0.0],
            "text": "Standard Incoterms 2020 define FOB and CIF shipping responsibilities. Commercial invoice and packing list are mandatory.",
            "metadata": {
                "source": "shipping_rules.txt",
                "chunk_index": 2,
                "section": "Documentation",
                "document_type": "txt",
                "country": "Global",
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
                "section": "Astronomy",
                "document_type": "txt",
                "country": "Global"
            }
        }
    ]
    collection.upsert(records)
    return collection


class TestMetadataFilteringAndHybridSearch(unittest.TestCase):

    def setUp(self):
        self.collection = get_sample_vector_collection()

    # --------------------------------------------------------------------------
    # 1. METADATA FILTERING TESTS
    # --------------------------------------------------------------------------

    @patch("src.retrieval.generate_query_embedding")
    def test_retrieve_with_metadata_filter_section(self, mock_embed):
        """Verifies filtering by section returns only matching section chunks."""
        mock_embed.return_value = [0.5, 0.5, 0.0, 0.0]

        filter_dict = {"section": "Account access"}
        results = retrieve("How to reset password?", k=3, metadata_filter=filter_dict, collection=self.collection)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "account-guide.md")
        self.assertEqual(results[0]["section"], "Account access")

    @patch("src.retrieval.generate_query_embedding")
    def test_retrieve_with_metadata_filter_country(self, mock_embed):
        """Verifies filtering by country returns only matching country chunks."""
        mock_embed.return_value = [0.0, 1.0, 0.0, 0.0]

        filter_dict = {"country": "India"}
        results = retrieve("What are the shipping rules?", k=3, metadata_filter=filter_dict, collection=self.collection)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "customs_reg_india.json")
        self.assertEqual(results[0]["metadata"]["country"], "India")

    def test_retrieve_invalid_metadata_filter_type_raises_value_error(self):
        """Verifies non-dictionary metadata_filter raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            retrieve("What are the rules?", k=3, metadata_filter="invalid_string_filter", collection=self.collection)
        self.assertIn("metadata_filter must be a dictionary", str(ctx.exception))

    @patch("src.retrieval.generate_query_embedding")
    def test_unfiltered_vs_filtered_comparison(self, mock_embed):
        """Verifies comparison between unfiltered (entire corpus) and filtered search."""
        mock_embed.return_value = [0.0, 1.0, 0.0, 0.0]
        query = "What are the requirements?"

        unfiltered = retrieve(query, k=3, metadata_filter=None, collection=self.collection)
        filtered = retrieve(query, k=3, metadata_filter={"country": "India"}, collection=self.collection)

        self.assertEqual(len(unfiltered), 3)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["metadata"]["country"], "India")

    # --------------------------------------------------------------------------
    # 2. KEYWORD SCORING TESTS
    # --------------------------------------------------------------------------

    def test_keyword_score_exact_matches(self):
        """Verifies keyword_score converts to lowercase and calculates match ratio."""
        text = "To reset your password, open Account Settings."
        keywords = ["PASSWORD", "reset"]

        score = keyword_score(text, keywords)
        self.assertEqual(score, 1.0)

    def test_keyword_score_partial_matches(self):
        """Verifies partial keyword matches calculate fractional match ratio."""
        text = "Customs rules for laptops require a commercial invoice."
        keywords = ["laptops", "password", "invoice"]

        # 2 out of 3 match ("laptops", "invoice")
        score = keyword_score(text, keywords)
        self.assertAlmostEqual(score, 0.6667, places=3)

    def test_keyword_score_empty_or_invalid_inputs(self):
        """Verifies empty keyword list or text safely returns 0.0."""
        self.assertEqual(keyword_score("", ["password"]), 0.0)
        self.assertEqual(keyword_score("Valid text", []), 0.0)
        self.assertEqual(keyword_score(None, ["password"]), 0.0)
        self.assertEqual(keyword_score("Valid text", None), 0.0)

    # --------------------------------------------------------------------------
    # 3. HYBRID RANKING TESTS
    # --------------------------------------------------------------------------

    def test_hybrid_rank_combination_and_sorting(self):
        """Verifies hybrid score = (0.8 * vector) + (0.2 * keyword) and results are sorted descending."""
        vector_results = [
            {
                "rank": 1,
                "similarity_score": 0.5000,
                "chunk_text": "Random text without target terms.",
                "source": "doc1.txt",
                "chunk_index": 1,
                "metadata": {"section": "General"}
            },
            {
                "rank": 2,
                "similarity_score": 0.4500,
                "chunk_text": "To reset your password follow instructions.",
                "source": "doc2.txt",
                "chunk_index": 1,
                "metadata": {"section": "Account access"}
            }
        ]

        keywords = ["password", "reset"]
        # doc1: vec=0.5, kw=0.0 -> hybrid = 0.8*0.5 + 0.2*0.0 = 0.4000
        # doc2: vec=0.45, kw=1.0 -> hybrid = 0.8*0.45 + 0.2*1.0 = 0.36 + 0.2 = 0.5600

        ranked = hybrid_rank(vector_results, keywords, vector_weight=0.8, keyword_weight=0.2)

        self.assertEqual(len(ranked), 2)
        # doc2 should boost to rank 1 due to keyword match
        self.assertEqual(ranked[0]["source"], "doc2.txt")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["hybrid_score"], 0.5600)
        self.assertEqual(ranked[0]["vector_score"], 0.4500)  # Original vector score preserved
        self.assertEqual(ranked[0]["keyword_score"], 1.0)

    # --------------------------------------------------------------------------
    # 4. EXACT-MATCH SCENARIO TESTS
    # --------------------------------------------------------------------------

    @patch("src.retrieval.generate_query_embedding")
    def test_exact_match_term_boosting(self, mock_embed):
        """Verifies hybrid search boosts exact terms like 'BIS Registration Certificate' and '8471.30'."""
        # Query vector gives doc2 slightly lower vector score than doc3, but exact keyword match boosts doc2 to #1
        mock_embed.return_value = [0.0, 0.8, 0.5, 0.0]
        query = "Customs rules for shipping laptops in India"
        keywords = ["BIS Registration Certificate", "8471.30"]

        v_only = retrieve(query, k=2, collection=self.collection)
        results = hybrid_retrieve(query, keywords=keywords, k=2, collection=self.collection)

        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertEqual(top_match["source"], "customs_reg_india.json")
        self.assertEqual(top_match["rank"], 1)
        self.assertGreater(top_match["keyword_score"], 0.0)
        self.assertGreater(top_match["hybrid_score"], top_match["vector_score"])


    # --------------------------------------------------------------------------
    # 5. INPUT VALIDATION & ERROR HANDLING TESTS (Test 4 from prompt)
    # --------------------------------------------------------------------------

    def test_retrieve_empty_and_invalid_inputs(self):
        """Verifies handling of invalid/empty queries, invalid k, and empty collection."""
        with self.assertRaises(ValueError):
            retrieve("", k=3, collection=self.collection)

        with self.assertRaises(ValueError):
            retrieve("Valid query", k=0, collection=self.collection)

        with self.assertRaises(ValueError):
            retrieve("Valid query", k=-1, collection=self.collection)

        empty_col = VectorCollection(name="empty")
        with self.assertRaises(ValueError):
            retrieve("Valid query", k=3, collection=empty_col)

    # --------------------------------------------------------------------------
    # 6. OUTPUT FORMATTER TESTS
    # --------------------------------------------------------------------------

    def test_format_filtered_vs_unfiltered_output(self):
        """Verifies formatting of unfiltered vs filtered side-by-side comparative report."""
        unfiltered = [
            {"rank": 1, "similarity_score": 0.8, "source": "doc1.txt", "section": "SecA", "chunk_index": 1, "chunk_text": "Text 1"}
        ]
        filtered = [
            {"rank": 1, "similarity_score": 0.8, "source": "doc1.txt", "section": "SecA", "chunk_index": 1, "chunk_text": "Text 1"}
        ]

        fmt = format_filtered_vs_unfiltered_output("Query", unfiltered, filtered, {"section": "SecA"})
        self.assertIn("UNFILTERED RESULTS", fmt)
        self.assertIn("FILTERED RESULTS", fmt)
        self.assertIn("Rank: 1", fmt)
        self.assertIn("Score: 0.8000", fmt)

    def test_format_hybrid_output(self):
        """Verifies formatting of hybrid search report."""
        hybrid_res = [
            {
                "rank": 1,
                "vector_score": 0.7500,
                "keyword_score": 1.0000,
                "hybrid_score": 0.8000,
                "source": "doc1.txt",
                "section": "SecA",
                "chunk_index": 1,
                "chunk_text": "Sample text"
            }
        ]

        fmt = format_hybrid_output("Test query", ["sample"], hybrid_res)
        self.assertIn("HYBRID SEARCH RESULTS", fmt)
        self.assertIn("Vector Score: 0.7500", fmt)
        self.assertIn("Keyword Score: 1.0000", fmt)
        self.assertIn("Hybrid Score: 0.8000", fmt)


if __name__ == "__main__":
    unittest.main()
