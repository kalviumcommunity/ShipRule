"""
Unit Tests for src/citation_manager.py (Citation Mapping & Source Verifiability)
================================================================================
Tests:
TEST 1: Citation markers [1], [2] are generated for retrieved chunks.
TEST 2: Citation [1] maps to the first retrieved chunk.
TEST 3: Citation [2] maps to the second retrieved chunk.
TEST 4: Citation metadata contains source document.
TEST 5: Citation metadata contains chunk ID and index.
TEST 6: Citation metadata contains section/page when available.
TEST 7: Original retrieved text is preserved completely without truncation.
TEST 8: Valid citation passes verification.
TEST 9: Unknown citation [99] fails verification.
TEST 10: No-source answer contains no fabricated citation.
TEST 11: Unsupported question triggers existing fallback.
TEST 12: Generated answer containing [1] succeeds in verification.
TEST 13: Detailed developer view formats cleanly and displays all source metadata.
"""

import unittest
import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.citation_manager import (
    extract_citations,
    build_citation_registry,
    verify_citation,
    verify_answer_citations,
    format_citation_registry,
    format_citation_details,
    format_sample_cited_answer,
    export_citation_mapping_json
)
from src.rag_pipeline import answer_query, answer_query_without_retrieval


class TestCitationManager(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            {
                "id": "chunk_india_001",
                "source": "customs_reg_india.json",
                "section": "Customs Record - Laptop Computers (India)",
                "chunk_index": 0,
                "page": 1,
                "country": "India",
                "hs_code": "8471.30",
                "doc_type": "RegulationData",
                "chunk_text": "Customs Record for Laptop Computers in India: Basic Customs Duty is 0%, IGST is 18%. Mandatory BIS CRS registration required."
            },
            {
                "id": "chunk_prd_002",
                "source": "cdlp_prd_v1.0.md",
                "section": "6. Dataset & Data Source Documentation",
                "chunk_index": 1,
                "page": 4,
                "country": "India",
                "doc_type": "PRD",
                "chunk_text": "CDLP Dataset Documentation: Schema includes country, HS code, duty rates, and mandatory import documentation requirements."
            },
            {
                "id": "chunk_guide_003",
                "source": "international_shipping_guide.pdf",
                "section": "Incoterms 2020",
                "chunk_index": 2,
                "page": 2,
                "chunk_text": "Under CIF terms, the seller delivers goods on board and pays freight and minimum insurance coverage to the destination port."
            }
        ]

    # --------------------------------------------------------------------------
    # TEST 1: CITATION MARKERS ARE GENERATED FOR RETRIEVED CHUNKS
    # --------------------------------------------------------------------------
    def test_01_citation_markers_generated_for_chunks(self):
        """TEST 1: Citation markers [1], [2], [3] are generated sequentially for chunks."""
        registry = build_citation_registry(self.sample_chunks)
        self.assertIn("[1]", registry)
        self.assertIn("[2]", registry)
        self.assertIn("[3]", registry)
        self.assertEqual(len(registry), 3)

    # --------------------------------------------------------------------------
    # TEST 2: CITATION [1] MAPS TO FIRST RETRIEVED CHUNK
    # --------------------------------------------------------------------------
    def test_02_citation_1_maps_to_first_chunk(self):
        """TEST 2: Citation [1] maps strictly to the first retrieved chunk."""
        registry = build_citation_registry(self.sample_chunks)
        first_entry = registry["[1]"]
        self.assertEqual(first_entry["source"], "customs_reg_india.json")
        self.assertEqual(first_entry["chunk_id"], "chunk_india_001")
        self.assertEqual(first_entry["chunk_index"], 0)
        self.assertEqual(first_entry["section"], "Customs Record - Laptop Computers (India)")
        self.assertEqual(first_entry["page"], 1)

    # --------------------------------------------------------------------------
    # TEST 3: CITATION [2] MAPS TO SECOND RETRIEVED CHUNK
    # --------------------------------------------------------------------------
    def test_03_citation_2_maps_to_second_chunk(self):
        """TEST 3: Citation [2] maps strictly to the second retrieved chunk."""
        registry = build_citation_registry(self.sample_chunks)
        second_entry = registry["[2]"]
        self.assertEqual(second_entry["source"], "cdlp_prd_v1.0.md")
        self.assertEqual(second_entry["chunk_id"], "chunk_prd_002")
        self.assertEqual(second_entry["chunk_index"], 1)
        self.assertEqual(second_entry["section"], "6. Dataset & Data Source Documentation")
        self.assertEqual(second_entry["page"], 4)

    # --------------------------------------------------------------------------
    # TEST 4: CITATION METADATA CONTAINS SOURCE DOCUMENT
    # --------------------------------------------------------------------------
    def test_04_citation_metadata_contains_source_document(self):
        """TEST 4: Citation metadata correctly records source document filename."""
        registry = build_citation_registry(self.sample_chunks)
        for marker, entry in registry.items():
            self.assertIn("source", entry)
            self.assertTrue(entry["source"].endswith((".json", ".md", ".pdf", ".txt")))

    # --------------------------------------------------------------------------
    # TEST 5: CITATION METADATA CONTAINS CHUNK ID AND INDEX
    # --------------------------------------------------------------------------
    def test_05_citation_metadata_contains_chunk_id_and_index(self):
        """TEST 5: Citation metadata records identifiable chunk_id and integer chunk_index."""
        registry = build_citation_registry(self.sample_chunks)
        for marker, entry in registry.items():
            self.assertIn("chunk_id", entry)
            self.assertIn("chunk_index", entry)
            self.assertIsInstance(entry["chunk_index"], int)
            self.assertTrue(bool(entry["chunk_id"]))

    # --------------------------------------------------------------------------
    # TEST 6: CITATION METADATA CONTAINS SECTION/PAGE WHEN AVAILABLE
    # --------------------------------------------------------------------------
    def test_06_citation_metadata_contains_section_and_page(self):
        """TEST 6: Section and page are captured when present, or set to None without inventing values."""
        registry = build_citation_registry(self.sample_chunks)
        self.assertEqual(registry["[1]"]["section"], "Customs Record - Laptop Computers (India)")
        self.assertEqual(registry["[1]"]["page"], 1)

        # Chunk with no page/section
        bare_chunk = [{"source": "unknown.txt", "text": "Bare text."}]
        bare_reg = build_citation_registry(bare_chunk)
        self.assertIsNone(bare_reg["[1]"]["section"])
        self.assertIsNone(bare_reg["[1]"]["page"])

    # --------------------------------------------------------------------------
    # TEST 7: ORIGINAL RETRIEVED TEXT IS PRESERVED
    # --------------------------------------------------------------------------
    def test_07_original_retrieved_text_preserved(self):
        """TEST 7: Full original retrieved chunk text is preserved in registry."""
        registry = build_citation_registry(self.sample_chunks)
        for idx, chunk in enumerate(self.sample_chunks, start=1):
            marker = f"[{idx}]"
            self.assertEqual(registry[marker]["original_text"], chunk["chunk_text"])

    # --------------------------------------------------------------------------
    # TEST 8: VALID CITATION PASSES VERIFICATION
    # --------------------------------------------------------------------------
    def test_08_valid_citation_passes_verification(self):
        """TEST 8: Valid citations [1] and [2] pass verification against the registry."""
        registry = build_citation_registry(self.sample_chunks)
        answer = "Laptop imports into India require BIS CRS registration [1] and customs documentation [2]."
        ver = verify_answer_citations(answer, registry, is_retrieval_mode=True)

        self.assertTrue(ver["is_valid"])
        self.assertEqual(ver["verification_status"], "PASS")
        self.assertEqual(ver["grounding_status"], "GROUNDED")
        self.assertEqual(ver["used_citations"], ["[1]", "[2]"])
        self.assertEqual(len(ver["unsupported_citations"]), 0)

    # --------------------------------------------------------------------------
    # TEST 9: UNKNOWN CITATION [99] FAILS VERIFICATION
    # --------------------------------------------------------------------------
    def test_09_unknown_citation_fails_verification(self):
        """TEST 9: Fabricated citation [99] is detected and rejected with status FAIL."""
        registry = build_citation_registry(self.sample_chunks)
        answer = "India requires special customs certificate [99]."
        ver = verify_answer_citations(answer, registry, is_retrieval_mode=True)

        self.assertFalse(ver["is_valid"])
        self.assertEqual(ver["verification_status"], "FAIL")
        self.assertEqual(ver["grounding_status"], "FAILED")
        self.assertIn("[99]", ver["unsupported_citations"])
        self.assertIn("INVALID", ver["report"])

    # --------------------------------------------------------------------------
    # TEST 10: NO-SOURCE ANSWER CONTAINS NO FABRICATED CITATION
    # --------------------------------------------------------------------------
    def test_10_no_source_answer_contains_no_fabricated_citation(self):
        """TEST 10: Fallback response contains zero fabricated citations."""
        fallback_answer = "The provided context is insufficient to answer this question."
        extracted = extract_citations(fallback_answer)
        self.assertEqual(len(extracted), 0)

        ver = verify_answer_citations(fallback_answer, registry={}, is_retrieval_mode=False)
        self.assertTrue(ver["is_valid"])
        self.assertEqual(ver["verification_status"], "PASS")
        self.assertEqual(ver["grounding_status"], "NO RETRIEVAL CONTEXT")

    # --------------------------------------------------------------------------
    # TEST 11: UNSUPPORTED QUESTION TRIGGERS EXISTING FALLBACK
    # --------------------------------------------------------------------------
    def test_11_unsupported_question_triggers_fallback(self):
        """TEST 11: Unsupported query with zero retrieval returns deterministic fallback."""
        result = answer_query(
            query="What is the duty on item XYZ-9000?",
            metadata_filter={"source": "non_existent_file_9999.pdf"}
        )
        self.assertIn("insufficient to answer this question", result["answer"])
        self.assertEqual(len(result["citation_registry"]), 0)
        self.assertEqual(result["citation_verification"]["verification_status"], "PASS")

    # --------------------------------------------------------------------------
    # TEST 12: GENERATED ANSWER CAN CONTAIN [1] AND VALIDATION SUCCEEDS
    # --------------------------------------------------------------------------
    def test_12_generated_answer_contains_citation_and_validates(self):
        """TEST 12: Answer containing citation [1] is parsed and successfully verified."""
        registry = build_citation_registry(self.sample_chunks)
        answer = "Under CIF terms, the seller pays freight and insurance to the destination port [3]."
        ver = verify_answer_citations(answer, registry, is_retrieval_mode=True)

        self.assertTrue(ver["is_valid"])
        self.assertEqual(ver["verification_status"], "PASS")
        self.assertIn("[3]", ver["used_citations"])

    # --------------------------------------------------------------------------
    # TEST 13: DETAILED DEVELOPER VIEW FORMATS CLEANLY
    # --------------------------------------------------------------------------
    def test_13_detailed_citation_view(self):
        """TEST 13: format_citation_details produces a clean, verifiable inspection block."""
        registry = build_citation_registry(self.sample_chunks)
        details = format_citation_details("[1]", registry)

        self.assertIn("CITATION DETAILS", details)
        self.assertIn("Source Document        : customs_reg_india.json", details)
        self.assertIn("Chunk ID               : chunk_india_001", details)
        self.assertIn("Section                : Customs Record - Laptop Computers (India)", details)
        self.assertIn("Page                   : 1", details)
        self.assertIn("Original Retrieved Text:", details)
        self.assertIn("Verification:", details)
        self.assertIn("PASS", details)


if __name__ == "__main__":
    unittest.main()
