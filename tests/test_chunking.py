"""
Unit Tests for Document Chunking Module
=======================================
Tests all chunking strategies (Fixed-Size, Paragraph-Based, Sentence-Based),
overlap mechanics, edge cases (empty text, short text, invalid overlap),
metadata preservation, statistics calculation, and sample corpus execution.
"""

import unittest
import sys
import os
import tempfile
import shutil
import json

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chunker import (
    fixed_size_chunking,
    paragraph_chunking,
    sentence_chunking,
    create_unified_chunk,
    chunk_document_by_strategy,
    calculate_chunk_stats,
    inspect_boundaries,
    recommend_best_strategy,
    run_chunking_pipeline,
)
from src.document_loader import load_directory, load_document


class TestDocumentChunking(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "Paragraph one introduces shipping rules and customs declaration requirements. "
            "All cargo consignments must declare correct HS tariff codes.\n\n"
            "Paragraph two explains preferential duty rates under Bilateral Free Trade Agreements (FTA). "
            "Importers must ensure that country of origin certificates are verified.\n\n"
            "Paragraph three covers dangerous goods. Class 1 through Class 9 goods require emergency response filings."
        )
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 1. Fixed-Size Chunking Tests
    # -------------------------------------------------------------
    def test_fixed_size_chunking_default(self):
        """Test fixed-size chunking splits text into slices with default size and overlap."""
        chunks = fixed_size_chunking(self.sample_text, size=150, overlap=30)
        self.assertGreater(len(chunks), 1)
        for c in chunks[:-1]:
            self.assertEqual(len(c), 150)
        # Last chunk can be <= size
        self.assertLessEqual(len(chunks[-1]), 150)

    def test_fixed_size_overlap_behavior(self):
        """Test that overlapping characters correctly carry over between adjacent chunks."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        size = 10
        overlap = 3
        chunks = fixed_size_chunking(text, size=size, overlap=overlap)
        
        # Chunk 0: "ABCDEFGHIJ" (0..10)
        # Step = 10 - 3 = 7 -> Chunk 1: "HIJKLMNOPQ" (7..17)
        self.assertEqual(chunks[0], "ABCDEFGHIJ")
        self.assertEqual(chunks[1], "HIJKLMNOPQ")
        # Check overlap tail of chunk 0 matches head of chunk 1
        self.assertEqual(chunks[0][-overlap:], chunks[1][:overlap])

    def test_fixed_size_invalid_overlap_raises_error(self):
        """Test that setting overlap >= chunk_size raises ValueError."""
        with self.assertRaises(ValueError):
            fixed_size_chunking(self.sample_text, size=100, overlap=100)

        with self.assertRaises(ValueError):
            fixed_size_chunking(self.sample_text, size=100, overlap=150)

        with self.assertRaises(ValueError):
            fixed_size_chunking(self.sample_text, size=0, overlap=0)

        with self.assertRaises(ValueError):
            fixed_size_chunking(self.sample_text, size=100, overlap=-5)

    # -------------------------------------------------------------
    # 2. Paragraph-Based Chunking Tests
    # -------------------------------------------------------------
    def test_paragraph_chunking_preserves_paragraphs(self):
        """Test paragraph chunking splits on double newlines and preserves complete paragraphs."""
        chunks = paragraph_chunking(self.sample_text)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[0].startswith("Paragraph one"))
        self.assertTrue(chunks[1].startswith("Paragraph two"))
        self.assertTrue(chunks[2].startswith("Paragraph three"))

    def test_paragraph_chunking_removes_empty_chunks(self):
        """Test that empty paragraphs and whitespace-only lines are cleanly removed."""
        dirty_text = "\n\n   \n\nParagraph 1 text.\n\n\n\n   \n\nParagraph 2 text.\n\n"
        chunks = paragraph_chunking(dirty_text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "Paragraph 1 text.")
        self.assertEqual(chunks[1], "Paragraph 2 text.")

    # -------------------------------------------------------------
    # 3. Sentence-Based Chunking Tests
    # -------------------------------------------------------------
    def test_sentence_chunking_does_not_cut_sentences(self):
        """Test sentence chunking splits on terminal punctuations without breaking clauses."""
        text = "This is sentence one. This is sentence two! Is this sentence three? Yes it is."
        chunks = sentence_chunking(text)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0], "This is sentence one.")
        self.assertEqual(chunks[1], "This is sentence two!")
        self.assertEqual(chunks[2], "Is this sentence three?")
        self.assertEqual(chunks[3], "Yes it is.")

    def test_sentence_chunking_handles_abbreviations(self):
        """Test sentence chunking does not split prematurely on abbreviations like 'Dr.', 'U.S.', or 'e.g.'."""
        text = "Dr. Smith arrived from the U.S. port. The container was inspected by customs."
        chunks = sentence_chunking(text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "Dr. Smith arrived from the U.S. port.")
        self.assertEqual(chunks[1], "The container was inspected by customs.")

    # -------------------------------------------------------------
    # 4. Edge Cases: Empty & Short Documents
    # -------------------------------------------------------------
    def test_empty_document_handling(self):
        """Test that all strategies handle empty and whitespace-only text safely."""
        empty_texts = ["", "   ", "\n\n\t  \n"]
        for empty in empty_texts:
            self.assertEqual(fixed_size_chunking(empty), [])
            self.assertEqual(paragraph_chunking(empty), [])
            self.assertEqual(sentence_chunking(empty), [])

    def test_short_document_handling(self):
        """Test that a short document produces exactly one chunk preserving full text."""
        short_text = "Single short sentence."
        
        fixed = fixed_size_chunking(short_text, size=500, overlap=50)
        self.assertEqual(len(fixed), 1)
        self.assertEqual(fixed[0], short_text)

        para = paragraph_chunking(short_text)
        self.assertEqual(len(para), 1)
        self.assertEqual(para[0], short_text)

        sent = sentence_chunking(short_text)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0], short_text)

    # -------------------------------------------------------------
    # 5. Unified Chunk Object & Metadata Preservation
    # -------------------------------------------------------------
    def test_unified_chunk_format_and_metadata(self):
        """Test unified chunk dictionary schema matches required specification."""
        doc = {
            "source": "shipping_rules.txt",
            "text": "All international shipments require documentation.\n\nCommercial invoices must specify values."
        }
        # Fixed strategy
        fixed_chunks = chunk_document_by_strategy(doc, strategy="fixed", size=500, overlap=50)
        self.assertEqual(fixed_chunks[0]["strategy"], "fixed")
        self.assertEqual(fixed_chunks[0]["chunk_id"], "shipping_rules_fixed_001")
        self.assertEqual(fixed_chunks[0]["source"], "shipping_rules.txt")
        self.assertEqual(fixed_chunks[0]["document_type"], "txt")
        self.assertEqual(fixed_chunks[0]["chunk_index"], 1)

        # Paragraph strategy
        para_chunks = chunk_document_by_strategy(doc, strategy="paragraph")
        self.assertEqual(len(para_chunks), 2)
        chunk_1 = para_chunks[0]
        self.assertEqual(chunk_1["source"], "shipping_rules.txt")
        self.assertEqual(chunk_1["document_type"], "txt")
        self.assertEqual(chunk_1["strategy"], "paragraph")
        self.assertEqual(chunk_1["chunk_index"], 1)
        self.assertEqual(chunk_1["chunk_id"], "shipping_rules_paragraph_001")
        self.assertEqual(chunk_1["character_count"], len("All international shipments require documentation."))
        self.assertEqual(chunk_1["chunk_text"], "All international shipments require documentation.")

        # Sentence strategy
        sent_chunks = chunk_document_by_strategy(doc, strategy="sentence")
        self.assertEqual(sent_chunks[0]["strategy"], "sentence")
        self.assertEqual(sent_chunks[0]["chunk_id"], "shipping_rules_sentence_001")


    # -------------------------------------------------------------
    # 6. Statistics Calculation & Boundary Inspection
    # -------------------------------------------------------------
    def test_calculate_chunk_stats(self):
        """Test accurate calculation of total chunks, avg size, min size, and max size."""
        chunks = [
            {"character_count": 100},
            {"character_count": 200},
            {"character_count": 300},
        ]
        stats = calculate_chunk_stats(chunks, original_char_count=600)
        self.assertEqual(stats["total_chunks"], 3)
        self.assertEqual(stats["avg_size"], 200.0)
        self.assertEqual(stats["min_size"], 100)
        self.assertEqual(stats["max_size"], 300)
        self.assertEqual(stats["original_char_count"], 600)

    def test_boundary_inspection(self):
        """Test boundary inspection evaluates adjacent chunk transitions."""
        chunks = [
            {"chunk_index": 1, "chunk_text": "First sentence finishes here."},
            {"chunk_index": 2, "chunk_text": "Second sentence starts here."}
        ]
        inspections = inspect_boundaries(chunks)
        self.assertEqual(len(inspections), 1)
        self.assertFalse(inspections[0]["breaks_sentence"])

    # -------------------------------------------------------------
    # 7. Sample Corpus Processing & Recommendation
    # -------------------------------------------------------------
    def test_run_chunking_pipeline_on_sample_corpus(self):
        """Test running the full chunking pipeline on the actual data/sample_corpus/ directory."""
        corpus_dir = os.path.join(project_root, "data", "sample_corpus")
        output_dir = os.path.join(self.test_dir, "outputs")

        report = run_chunking_pipeline(corpus_dir=corpus_dir, output_dir=output_dir, verbose=False)
        self.assertIsNotNone(report)
        self.assertIn("documents_processed", report)
        self.assertEqual(len(report["documents_processed"]), 3)

        # Verify output files generated
        for fname in ["chunks_fixed.json", "chunks_paragraph.json", "chunks_sentence.json", "chunking_report.json"]:
            fpath = os.path.join(output_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"File {fname} must be generated in outputs/")
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertTrue(len(data) > 0)

        # Verify recommendation selects Paragraph-based for ShipRule corpus
        rec = report["recommendation"]
        self.assertEqual(rec["strategy_key"], "paragraph")
        self.assertIn("Paragraph-based chunking is recommended", rec["reason"])


if __name__ == "__main__":
    unittest.main()
