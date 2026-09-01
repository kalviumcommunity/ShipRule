"""
Unit Tests for Chunk Metadata & Source Tracking Module (KDU 3.22)
==================================================================
Tests chunk metadata tagging, consistent schema validation, additional metadata
attachment (section, position, page), and source traceability.
"""

import unittest
import sys
import os

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chunk_metadata import (
    create_metadata_dict,
    validate_chunk_metadata_schema,
    tag_chunks,
    chunk_document_with_metadata,
    trace_chunk_source,
    REQUIRED_METADATA_KEYS,
)


class TestChunkMetadata(unittest.TestCase):

    def setUp(self):
        self.sample_prd_text = (
            "8.1 Data Integration Module: The system SHALL store customs regulation data "
            "by country and HS code including duty rates, required import documents, "
            "restricted-item status, source agency, and source URL."
        )
        self.sample_regulation_text = (
            "Customs Record for Laptop Computers in India: Destination Country: India. "
            "HS Code: 8471.30. Duty Rate: 7.5% BCD + 10% SWS. Required Documents: Commercial Invoice, "
            "Bill of Lading, BIS Registration Certificate. Source Agency: DGFT & CBIC, India. "
            "Source URL: https://www.cbic.gov.in. Last Confirmed Date: 2026-02-10."
        )

    def test_store_source_identifier(self):
        """Task 1: Ensure each chunk stores its source document identifier."""
        chunks = chunk_document_with_metadata(
            text=self.sample_prd_text,
            source="cdlp_prd_v1.0.md",
            section="8.1 Data Integration Module"
        )
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("source", chunk["metadata"])
            self.assertEqual(chunk["metadata"]["source"], "cdlp_prd_v1.0.md")

    def test_attach_additional_metadata(self):
        """Task 2: Attach section, position (char_start/end), chunk_index, and page metadata."""
        chunks = chunk_document_with_metadata(
            text=self.sample_prd_text,
            source="cdlp_prd_v1.0.md",
            section="8.1 Data Integration Module",
            doc_type="PRD",
            chunk_size=100,
            chunk_overlap=20,
            page=4
        )
        self.assertGreaterEqual(len(chunks), 2)
        
        # Check first chunk metadata
        meta0 = chunks[0]["metadata"]
        self.assertEqual(meta0["section"], "8.1 Data Integration Module")
        self.assertEqual(meta0["chunk_index"], 0)
        self.assertEqual(meta0["char_start"], 0)
        self.assertEqual(meta0["char_end"], 100)
        self.assertEqual(meta0["page"], 4)
        self.assertEqual(meta0["doc_type"], "PRD")

        # Check second chunk position overlap
        meta1 = chunks[1]["metadata"]
        self.assertEqual(meta1["chunk_index"], 1)
        self.assertEqual(meta1["char_start"], 80)
        self.assertEqual(meta1["char_end"], 180)

    def test_consistent_structure_across_corpus(self):
        """Task 3: Keep metadata alongside text in a consistent structure across corpus."""
        prd_chunks = chunk_document_with_metadata(
            text=self.sample_prd_text,
            source="cdlp_prd_v1.0.md",
            section="8.1 Data Integration Module",
            doc_type="PRD"
        )
        reg_chunks = chunk_document_with_metadata(
            text=self.sample_regulation_text,
            source="customs_reg_india.json",
            section="Laptop Import Requirements",
            doc_type="RegulationData",
            extra_metadata={
                "country": "India",
                "hs_code": "8471.30",
                "source_agency": "DGFT & CBIC, India",
                "source_url": "https://www.cbic.gov.in",
                "last_confirmed_date": "2026-02-10"
            }
        )

        all_chunks = prd_chunks + reg_chunks
        self.assertGreater(len(all_chunks), 1)

        # Validate that every single chunk in the corpus has identical required schema keys
        first_keys = set(all_chunks[0]["metadata"].keys())
        for idx, chunk in enumerate(all_chunks):
            self.assertTrue(validate_chunk_metadata_schema(chunk))
            self.assertEqual(set(chunk["metadata"].keys()), first_keys)

    def test_trace_chunk_to_source(self):
        """Task 4: Demonstrate that a retrieved chunk can be traced back to its exact source."""
        reg_chunks = chunk_document_with_metadata(
            text=self.sample_regulation_text,
            source="customs_reg_india.json",
            section="Laptop Import Requirements",
            doc_type="RegulationData",
            extra_metadata={
                "country": "India",
                "hs_code": "8471.30",
                "source_agency": "DGFT & CBIC, India",
                "source_url": "https://www.cbic.gov.in",
                "last_confirmed_date": "2026-02-10"
            }
        )

        retrieved_chunk = reg_chunks[0]
        traceback = trace_chunk_source(retrieved_chunk)

        self.assertEqual(traceback["source"], "customs_reg_india.json")
        self.assertEqual(traceback["section"], "Laptop Import Requirements")
        self.assertEqual(traceback["country"], "India")
        self.assertEqual(traceback["hs_code"], "8471.30")
        self.assertEqual(traceback["source_agency"], "DGFT & CBIC, India")
        self.assertEqual(traceback["source_url"], "https://www.cbic.gov.in")
        self.assertIn("customs_reg_india.json", traceback["formatted_citation"])
        self.assertIn("https://www.cbic.gov.in", traceback["formatted_citation"])
        self.assertIn("DGFT & CBIC, India", traceback["formatted_citation"])

    def test_tag_chunks_helper(self):
        """Test tag_chunks function directly with raw tuples."""
        raw_tuples = [
            ("First snippet of document", 0, 25),
            ("Second snippet of document", 20, 46)
        ]
        tagged = tag_chunks(
            source="test_doc.txt",
            raw_chunks=raw_tuples,
            section="Intro",
            doc_type="Test"
        )
        self.assertEqual(len(tagged), 2)
        self.assertEqual(tagged[0]["text"], "First snippet of document")
        self.assertEqual(tagged[0]["metadata"]["char_start"], 0)
        self.assertEqual(tagged[0]["metadata"]["char_end"], 25)
        self.assertEqual(tagged[1]["metadata"]["chunk_index"], 1)


if __name__ == "__main__":
    unittest.main()
