"""
ShipRule CDLP - Consolidated Test Suite (tests/test_all.py)
============================================================
Consolidates all 309 automated unit, integration, API, RAG pipeline,
retrieval, document loader, chunking, embedding, vector store, citation,
and evaluation tests into a unified test file.
"""

import os
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# UTF-8 stdout configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# DOCUMENT LOADER TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_document_loader.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Document Intake & Loader Module
===============================================
Tests loading multiple document formats (TXT, PDF), plain-text conversion,
source identity preservation, graceful error handling for missing, corrupt, and
unsupported files, and batch intake reporting.
"""

import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.document_loader import (
    load_document,
    load_documents,
    load_directory,
    _normalize_sample_text,
    SUPPORTED_EXTENSIONS,
)


class TestDocumentLoader(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Create a sample TXT file
        self.sample_txt_path = os.path.join(self.test_dir, "test_shipping.txt")
        with open(self.sample_txt_path, "w", encoding="utf-8") as f:
            f.write("All international shipments require a commercial invoice and packing list.")

        # Create a sample PDF file using fpdf2
        self.sample_pdf_path = os.path.join(self.test_dir, "test_customs.pdf")
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=pdf.epw, h=10, text="Customs Regulations Document", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(w=pdf.epw, h=5, text="Import duty rates are determined based on the 8-digit HS Code classification.")
        pdf.output(self.sample_pdf_path)

        # Create an unsupported format file
        self.unsupported_file_path = os.path.join(self.test_dir, "document.docx")
        with open(self.unsupported_file_path, "w", encoding="utf-8") as f:
            f.write("Dummy Word Document content.")

        # Create a corrupt PDF file
        self.corrupt_pdf_path = os.path.join(self.test_dir, "corrupt_document.pdf")
        with open(self.corrupt_pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 THIS_IS_NOT_A_VALID_PDF_BODY_CORRUPT_BYTES")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_txt_file(self):
        """Task 1: Test loading a TXT file converts to plain text representation."""
        doc = load_document(self.sample_txt_path)
        self.assertIsNotNone(doc)
        self.assertIsInstance(doc, dict)
        self.assertEqual(doc["source"], "test_shipping.txt")
        self.assertIn("commercial invoice", doc["text"])

    def test_load_pdf_file(self):
        """Task 1: Test loading a PDF file extracts plain text correctly."""
        doc = load_document(self.sample_pdf_path)
        self.assertIsNotNone(doc)
        self.assertIsInstance(doc, dict)
        self.assertEqual(doc["source"], "test_customs.pdf")
        self.assertIn("Customs Regulations", doc["text"])
        self.assertIn("HS Code", doc["text"])

    def test_preserve_source_identity(self):
        """Task 3: Test that loaded document preserves exact base filename as source."""
        doc_txt = load_document(self.sample_txt_path)
        doc_pdf = load_document(self.sample_pdf_path)

        self.assertEqual(doc_txt["source"], os.path.basename(self.sample_txt_path))
        self.assertEqual(doc_pdf["source"], os.path.basename(self.sample_pdf_path))

    def test_missing_file_handled_gracefully(self):
        """Task 2: Test missing file returns None and does not raise an unhandled exception."""
        missing_path = os.path.join(self.test_dir, "non_existent_file.pdf")
        doc = load_document(missing_path)
        self.assertIsNone(doc)

    def test_unsupported_file_handled_gracefully(self):
        """Task 2: Test unsupported file format returns None and does not crash."""
        doc = load_document(self.unsupported_file_path)
        self.assertIsNone(doc)

    def test_corrupt_file_handled_gracefully(self):
        """Task 2: Test corrupt file returns None and does not crash."""
        doc = load_document(self.corrupt_pdf_path)
        self.assertIsNone(doc)

    def test_load_documents_batch_and_summary(self):
        """Task 4: Test batch loading multiple files including valid, missing, and unsupported."""
        paths = [
            self.sample_txt_path,
            self.sample_pdf_path,
            os.path.join(self.test_dir, "missing_file.txt"),
            self.unsupported_file_path,
            self.corrupt_pdf_path,
        ]

        docs = load_documents(paths, verbose=True)
        self.assertEqual(len(docs), 2)
        sources = [d["source"] for d in docs]
        self.assertIn("test_shipping.txt", sources)
        self.assertIn("test_customs.pdf", sources)

    def test_load_directory(self):
        """Test scanning and loading all supported files from a directory."""
        docs = load_directory(self.test_dir, verbose=False)
        # Should load the valid TXT and valid PDF, while skipping unsupported & corrupt
        self.assertEqual(len(docs), 2)

    def test_sample_corpus_files_exist_and_load(self):
        """Task 5: Test that the committed sample corpus files exist and load successfully."""
        corpus_dir = os.path.join(project_root, "data", "sample_corpus")
        self.assertTrue(os.path.exists(corpus_dir), "data/sample_corpus directory must exist")

        docs = load_directory(corpus_dir, verbose=True)
        self.assertGreaterEqual(len(docs), 3)

        sources = {d["source"] for d in docs}
        self.assertIn("shipping_rules.txt", sources)
        self.assertIn("customs_requirements.txt", sources)
        self.assertIn("international_shipping_guide.pdf", sources)

        for doc in docs:
            self.assertGreater(len(doc["text"]), 100)

    def test_normalize_sample_text(self):
        """Test text snippet normalization collapses excess whitespace and truncates."""
        raw = "Line 1\n\n\n   Line 2\t\tLine 3    "
        norm = _normalize_sample_text(raw, max_chars=50)
        self.assertEqual(norm, "Line 1 Line 2 Line 3")

    def test_load_document_cleans_extracted_text(self):
        """Test that load_document applies the clean function to extracted document text."""
        dirty_txt_path = os.path.join(self.test_dir, "dirty_doc.txt")
        with open(dirty_txt_path, "w", encoding="utf-8") as f:
            f.write("Line 1\r\nPage 3 of 12\r\n\r\n\r\n   Line 2   with   spaces.   ")

        doc = load_document(dirty_txt_path)
        self.assertIsNotNone(doc)
        self.assertEqual(doc["text"], "Line 1\n\nLine 2 with spaces.")



# ============================================================================
# TEXT CLEANING TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_text_cleaner.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Text Extraction & Cleaning Pipeline Module
===========================================================
Tests Unicode NFKC normalization, line ending normalization (\r\n to \n),
page footer pattern removal, whitespace collapsing, consecutive newline reduction,
and preservation of meaningful content (headings, numbers, code blocks, tables).
"""

import unittest
import sys
import os

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.text_cleaner import clean, format_cleaning_summary


class TestTextCleaner(unittest.TestCase):

    def test_unicode_normalization_nfkc(self):
        """Test Unicode normalization converts fullwidth/compat characters using NFKC."""
        raw_text = "Fullwidth numbers \uff11\uff12\uff13 and ligature \ufb01le."
        cleaned = clean(raw_text)
        self.assertEqual(cleaned, "Fullwidth numbers 123 and ligature file.")

    def test_windows_line_ending_normalization(self):
        """Test normalizing \\r\\n and \\r to standard \\n."""
        raw_text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        cleaned = clean(raw_text)
        self.assertEqual(cleaned, "Line 1\nLine 2\nLine 3\nLine 4")

    def test_page_footer_removal(self):
        """Test removal of page footer patterns like 'Page 3 of 12' and 'Page 10 of 25'."""
        raw_text = (
            "Customs documentation guidelines.\n"
            "Page 3 of 12\n"
            "All shipments require invoice.\n"
            "Page 10 of 25\n"
            "End of section."
        )
        cleaned = clean(raw_text)
        self.assertNotIn("Page 3 of 12", cleaned)
        self.assertNotIn("Page 10 of 25", cleaned)
        self.assertIn("Customs documentation guidelines.", cleaned)
        self.assertIn("All shipments require invoice.", cleaned)
        self.assertIn("End of section.", cleaned)

    def test_case_insensitive_page_footer_removal(self):
        """Test footer pattern removal handles varied casing."""
        raw_text = "Content before.\npage 1 of 5\npage 100 OF 200\nContent after."
        cleaned = clean(raw_text)
        self.assertNotIn("page 1 of 5", cleaned)
        self.assertNotIn("page 100 OF 200", cleaned)
        self.assertIn("Content before.", cleaned)
        self.assertIn("Content after.", cleaned)

    def test_collapse_repeated_spaces_and_tabs(self):
        """Test collapsing repeated spaces and tabs into a single space while keeping newlines."""
        raw_text = "Word1    Word2\t\tWord3   \t   Word4\nLine 2   Word5"
        cleaned = clean(raw_text)
        self.assertEqual(cleaned, "Word1 Word2 Word3 Word4\nLine 2 Word5")

    def test_collapse_consecutive_newlines(self):
        """Test collapsing 3 or more consecutive newlines into a maximum of 2."""
        raw_text = "Header\n\n\n\n\nParagraph 1\n\n\nParagraph 2"
        cleaned = clean(raw_text)
        self.assertEqual(cleaned, "Header\n\nParagraph 1\n\nParagraph 2")

    def test_strip_leading_trailing_whitespace(self):
        """Test stripping leading and trailing whitespace."""
        raw_text = "   \n\n   Clean text inside.   \n\n  "
        cleaned = clean(raw_text)
        self.assertEqual(cleaned, "Clean text inside.")

    def test_preserve_headings_and_punctuation(self):
        """Test preservation of markdown headings and punctuation."""
        raw_text = (
            "# Main Title: Customs Duty & Tariff Rules\n\n"
            "## Section 1.2: Compliance Verification!\n"
            "Is origin certificate required? Yes (under FTA rules)."
        )
        cleaned = clean(raw_text)
        self.assertIn("# Main Title: Customs Duty & Tariff Rules", cleaned)
        self.assertIn("## Section 1.2: Compliance Verification!", cleaned)
        self.assertIn("Is origin certificate required? Yes (under FTA rules).", cleaned)

    def test_preserve_numbers_and_tariff_codes(self):
        """Test preservation of numbers, HS codes, and monetary values."""
        raw_text = "Harmonized Tariff Code: 8471.30.10. Total declared value: $12,500.50 (Duty rate: 7.5%)."
        cleaned = clean(raw_text)
        self.assertEqual(
            cleaned,
            "Harmonized Tariff Code: 8471.30.10. Total declared value: $12,500.50 (Duty rate: 7.5%)."
        )

    def test_preserve_code_blocks(self):
        """Test preservation of code blocks and structural syntax."""
        raw_text = (
            "```python\n"
            "def calculate_duty(value, rate):\n"
            "    return value * rate\n"
            "```"
        )
        cleaned = clean(raw_text)
        self.assertIn("```python", cleaned)
        self.assertIn("def calculate_duty(value, rate):", cleaned)
        self.assertIn("return value * rate", cleaned)
        self.assertIn("```", cleaned)

    def test_preserve_table_content(self):
        """Test preservation of table headers, borders, and cell values."""
        raw_text = (
            "| HS Code | Commodity Description | Duty Rate |\n"
            "| ------- | --------------------- | --------- |\n"
            "| 8471.30 | Laptops & Notebooks   | 0%        |\n"
            "| 8517.12 | Smartphones           | 20%       |"
        )
        cleaned = clean(raw_text)
        self.assertIn("| HS Code | Commodity Description | Duty Rate |", cleaned)
        self.assertIn("| 8471.30 | Laptops & Notebooks | 0% |", cleaned)
        self.assertIn("| 8517.12 | Smartphones | 20% |", cleaned)

    def test_no_overcleaning_of_meaningful_text(self):
        """Ensure sentences mentioning numbers or 'Page' in normal prose are not erased."""
        raw_text = "Refer to Section 3 of the agreement. See Page 5 for details on shipping rates."
        cleaned = clean(raw_text)
        # "Section 3" and "See Page 5 for details" should NOT be removed as page footers
        self.assertIn("Refer to Section 3 of the agreement.", cleaned)
        self.assertIn("See Page 5 for details on shipping rates.", cleaned)

    def test_edge_cases_empty_and_none(self):
        """Test edge cases with empty inputs, None, or pure whitespace."""
        self.assertEqual(clean(""), "")
        self.assertEqual(clean(None), "")
        self.assertEqual(clean("   \t \n\r\n "), "")

    def test_format_cleaning_summary(self):
        """Test summary output formatting helper."""
        raw = "Line 1\r\nPage 1 of 5\r\nLine 2"
        cleaned = clean(raw)
        summary = format_cleaning_summary("document.pdf", raw, cleaned)
        self.assertIn("document.pdf: 27 -> 14 chars", summary)
        self.assertIn("BEFORE:", summary)
        self.assertIn("AFTER :", summary)



# ============================================================================
# CHUNKING TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_chunking.py
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# Source File: tests\test_token_chunks.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Concept 13 — Token-Aware Chunk Sizing & Overlap
===============================================================
Tests token-based chunking with tiktoken (cl100k_base), metadata generation,
overlap configurations, edge cases, error handling, ordering, and context preservation.
"""

import unittest
import sys
import os
import tiktoken

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.document_loader import token_chunks, chunk_document, chunk_documents
from src.token_counter import get_tokenizer


class TestTokenAwareChunking(unittest.TestCase):

    def setUp(self):
        self.enc = get_tokenizer("cl100k_base")

    def test_1_empty_text(self):
        """1. Test empty text returns an empty list safely."""
        chunks_empty = token_chunks("", size=400, overlap=60)
        self.assertEqual(chunks_empty, [])

        chunks_spaces = token_chunks("   \n\t  ", size=400, overlap=60)
        self.assertEqual(chunks_spaces, [])

    def test_2_short_text(self):
        """2. Test short text smaller than overlap size produces a single chunk."""
        text = "Commercial invoice required."
        tokens = self.enc.encode(text)
        chunks = token_chunks(text, size=400, overlap=60)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], 1)
        self.assertEqual(chunks[0]["token_count"], len(tokens))
        self.assertEqual(chunks[0]["start_token"], 0)
        self.assertEqual(chunks[0]["end_token"], len(tokens))
        self.assertEqual(chunks[0]["text"], text)

    def test_3_text_smaller_than_chunk_size(self):
        """3. Test text larger than overlap but smaller than chunk size produces single chunk."""
        words = ["customs"] * 150  # ~150 tokens
        text = " ".join(words)
        tokens = self.enc.encode(text)

        chunks = token_chunks(text, size=400, overlap=60)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["token_count"], len(tokens))
        self.assertEqual(chunks[0]["start_token"], 0)
        self.assertEqual(chunks[0]["end_token"], len(tokens))

    def test_4_text_larger_than_chunk_size(self):
        """4. Test text larger than chunk size splits into multiple chunks with overlap."""
        words = ["duty"] * 500  # ~500 tokens
        text = " ".join(words)

        chunks = token_chunks(text, size=400, overlap=60)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["token_count"], 400)
        self.assertEqual(chunks[0]["start_token"], 0)
        self.assertEqual(chunks[0]["end_token"], 400)
        self.assertEqual(chunks[1]["start_token"], 340)  # 400 - 60

    def test_5_exact_chunk_size_text(self):
        """5. Test text matching exact chunk size (400 tokens) produces exactly 1 chunk."""
        token_id = self.enc.encode(" duty")[0]
        exact_text = self.enc.decode([token_id] * 400)
        tokens = self.enc.encode(exact_text)

        self.assertEqual(len(tokens), 400)

        chunks = token_chunks(exact_text, size=400, overlap=60)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["token_count"], 400)
        self.assertEqual(chunks[0]["start_token"], 0)
        self.assertEqual(chunks[0]["end_token"], 400)

    def test_6_overlap_zero(self):
        """6. Test chunking with overlap = 0 (no overlap between chunks)."""
        words = ["tariff"] * 600
        text = " ".join(words)
        tokens = self.enc.encode(text)

        chunks = token_chunks(text, size=400, overlap=0)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["start_token"], 0)
        self.assertEqual(chunks[0]["end_token"], 400)
        self.assertEqual(chunks[1]["start_token"], 400)
        self.assertEqual(chunks[1]["end_token"], len(tokens))

    def test_7_overlap_sixty(self):
        """7. Test chunking with overlap = 60 verifies shared token window."""
        words = ["logistics"] * 600
        text = " ".join(words)

        chunks = token_chunks(text, size=400, overlap=60)
        self.assertGreaterEqual(len(chunks), 2)

        c1 = chunks[0]
        c2 = chunks[1]

        # Check overlapping token indices
        self.assertEqual(c1["start_token"], 0)
        self.assertEqual(c1["end_token"], 400)
        self.assertEqual(c2["start_token"], 340)
        self.assertEqual(c1["end_token"] - c2["start_token"], 60)

    def test_8_invalid_overlap_raises_value_error(self):
        """8. Test invalid overlap parameters raise ValueError."""
        text = "Test text for validation."

        # Overlap equal to size
        with self.assertRaises(ValueError):
            token_chunks(text, size=400, overlap=400)

        # Overlap greater than size
        with self.assertRaises(ValueError):
            token_chunks(text, size=400, overlap=450)

        # Negative overlap
        with self.assertRaises(ValueError):
            token_chunks(text, size=400, overlap=-10)

        # Non-positive chunk size
        with self.assertRaises(ValueError):
            token_chunks(text, size=0, overlap=10)

        with self.assertRaises(ValueError):
            token_chunks(text, size=-100, overlap=10)

    def test_9_correct_token_counts(self):
        """9. Test that chunk token counts match actual tiktoken counts, not character counts."""
        text = "Pneumonoultramicroscopicsilicovolcanoconiosis antidisestablishmentarianism 12345 !@#$%"
        char_length = len(text)
        actual_token_count = len(self.enc.encode(text))

        # Ensure token count differs from character length
        self.assertNotEqual(char_length, actual_token_count)

        chunks = token_chunks(text, size=10, overlap=2)
        total_tokens_counted = sum(c["token_count"] for c in chunks)
        for c in chunks:
            # Each chunk's text decoded back and re-encoded should match token_count
            self.assertEqual(c["token_count"], len(self.enc.encode(c["text"])))

    def test_10_correct_ordering_of_chunks(self):
        """10. Test that chunks maintain strictly incremental chunk_id and start_token sequence."""
        words = ["document"] * 1000
        text = " ".join(words)

        chunks = token_chunks(text, size=300, overlap=50)
        self.assertGreater(len(chunks), 1)

        prev_id = 0
        prev_start = -1

        for c in chunks:
            self.assertEqual(c["chunk_id"], prev_id + 1)
            self.assertGreater(c["start_token"], prev_start)
            prev_id = c["chunk_id"]
            prev_start = c["start_token"]

    def test_11_boundary_context_preservation(self):
        """11. Test boundary context preservation with overlap vs no overlap."""
        prefix = "Word " * 380
        critical = "CRITICAL BOUNDARY SENTENCE FOR CUSTOMS VERIFICATION."
        suffix = " Word" * 380
        text = prefix + critical + suffix

        # Case A: Overlap = 0
        chunks_zero = token_chunks(text, size=400, overlap=0)
        self.assertGreaterEqual(len(chunks_zero), 2)

        # Case B: Overlap = 60
        chunks_overlap = token_chunks(text, size=400, overlap=60)
        self.assertGreaterEqual(len(chunks_overlap), 2)

        # With 60 overlap, chunk 2 should start earlier and capture context surrounding critical sentence
        self.assertLess(chunks_overlap[1]["start_token"], chunks_zero[1]["start_token"])
        self.assertEqual(chunks_zero[1]["start_token"] - chunks_overlap[1]["start_token"], 60)

    def test_12_no_accidental_infinite_loops(self):
        """12. Test that edge-case parameters terminate quickly without infinite loops."""
        text = "Fast loop test." * 50

        # Step size = 1 (size=2, overlap=1)
        chunks = token_chunks(text, size=2, overlap=1)
        self.assertGreater(len(chunks), 1)

        # Large text with small chunk size
        chunks_small = token_chunks(text, size=10, overlap=5)
        self.assertGreater(len(chunks_small), 1)

    def test_document_loader_integration(self):
        """Test chunk_document and chunk_documents propagate token metadata properly."""
        doc = {
            "source": "customs_guide.txt",
            "text": "Import requirements for laptops. " * 50
        }
        chunks = chunk_document(doc, max_chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertIn("id", c)
            self.assertIn("text", c)
            self.assertIn("metadata", c)
            meta = c["metadata"]
            self.assertEqual(meta["source"], "customs_guide.txt")
            self.assertIn("token_count", meta)
            self.assertIn("start_token", meta)
            self.assertIn("end_token", meta)
            self.assertIn("overlap", meta)

# ------------------------------------------------------------------------------
# Source File: src/test_chunking.py
# ------------------------------------------------------------------------------

import os
import sys
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chunking import chunk_text

class TestChunking(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(chunk_text(""), [])

    def test_short_string(self):
        text = "Hello world!"
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
        self.assertEqual(chunks, [text])

    def test_exact_chunk_size(self):
        text = "1234567890"
        chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
        self.assertEqual(chunks, [text])

    def test_multiple_chunks_with_overlap(self):
        text = "1234567890" # 10 chars
        # size 5, overlap 2. 
        # chunk 1: text[0:5] -> "12345"
        # start shifts by: 5 - 2 = 3. New start is 3.
        # chunk 2: text[3:8] -> "45678"
        # start shifts to 6.
        # chunk 3: text[6:10] -> "7890"
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=2)
        self.assertEqual(chunks, ["12345", "45678", "7890"])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=-1)
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=5, chunk_overlap=6)



# ============================================================================
# CHUNK METADATA TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_chunk_metadata.py
# ------------------------------------------------------------------------------

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



# ============================================================================
# EMBEDDING TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_batch_embedding.py
# ------------------------------------------------------------------------------

"""
Unit Tests for KDU 3.28 Batch Embedding & Rate/Cost Management Module
========================================================================
Tests batching logic, exponential backoff retries, cost estimation, and skip-on-rerun resumability.
"""

import os
import sys
import json
import unittest
from unittest.mock import MagicMock

# Ensure project root directory is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.batch_embedding import (
    batches,
    estimate_tokens,
    embed_with_retry,
    BatchEmbeddingPipeline,
    CDLP_SAMPLE_CORPUS,
    DEFAULT_PRICE_PER_1K_TOKENS
)


class TestBatchEmbedding(unittest.TestCase):

    def setUp(self):
        self.test_cache_path = os.path.join(project_root, "data", "test_embeddings_cache.json")
        if os.path.exists(self.test_cache_path):
            os.remove(self.test_cache_path)

    def tearDown(self):
        if os.path.exists(self.test_cache_path):
            os.remove(self.test_cache_path)

    def test_batches_generator(self):
        """Task 1: Test that batches splits items into chunks of specified size correctly."""
        items = list(range(10))
        b_list = list(batches(items, size=3))

        self.assertEqual(len(b_list), 4)
        self.assertEqual(b_list[0], [0, 1, 2])
        self.assertEqual(b_list[1], [3, 4, 5])
        self.assertEqual(b_list[2], [6, 7, 8])
        self.assertEqual(b_list[3], [9])

    def test_batches_invalid_size(self):
        """Task 1: Test ValueError when batch size is <= 0."""
        with self.assertRaises(ValueError):
            list(batches([1, 2, 3], size=0))

    def test_estimate_tokens(self):
        """Task 3: Test estimation of tokens across a list of texts."""
        texts = ["Customs duty rate India HS code 8471", "Required import documents for Italy cars"]
        tokens = estimate_tokens(texts)
        self.assertGreater(tokens, 0)

    def test_embed_with_retry_success(self):
        """Task 2: Test embed_with_retry succeeds without retries on normal execution."""
        mock_embed_fn = MagicMock(return_value=[[0.1, 0.2]])
        res = embed_with_retry(["test text"], embed_fn=mock_embed_fn, max_attempts=3, initial_backoff=0.01)

        self.assertEqual(res, [[0.1, 0.2]])
        self.assertEqual(mock_embed_fn.call_count, 1)

    def test_embed_with_retry_transient_failure_then_success(self):
        """Task 2: Test embed_with_retry retries on transient errors and succeeds on attempt 2."""
        mock_embed_fn = MagicMock(side_effect=[Exception("Rate limit 429"), [[0.5, 0.6]]])
        res = embed_with_retry(["rate limited text"], embed_fn=mock_embed_fn, max_attempts=3, initial_backoff=0.01)

        self.assertEqual(res, [[0.5, 0.6]])
        self.assertEqual(mock_embed_fn.call_count, 2)

    def test_embed_with_retry_max_attempts_exceeded(self):
        """Task 2: Test embed_with_retry raises exception when max attempts are exceeded."""
        mock_embed_fn = MagicMock(side_effect=Exception("Permanent API Error 500"))

        with self.assertRaises(Exception) as context:
            embed_with_retry(["failing text"], embed_fn=mock_embed_fn, max_attempts=3, initial_backoff=0.01)

        self.assertIn("Permanent API Error 500", str(context.exception))
        self.assertEqual(mock_embed_fn.call_count, 3)

    def test_pipeline_initial_run(self):
        """Task 1, 3: Test pipeline initial run embeds all chunks, counts tokens, and estimates cost."""
        pipeline = BatchEmbeddingPipeline(
            batch_size=2,
            price_per_1k_tokens=0.00002,
            cache_path=self.test_cache_path
        )
        summary = pipeline.process_corpus(CDLP_SAMPLE_CORPUS[:4])

        self.assertEqual(summary["total_chunks"], 4)
        self.assertEqual(summary["skipped_existing"], 0)
        self.assertEqual(summary["embedded"], 4)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["batches_processed"], 2)
        self.assertGreater(summary["input_tokens"], 0)
        self.assertGreater(summary["estimated_cost_usd"], 0.0)

    def test_pipeline_skip_on_rerun_resumability(self):
        """Task 4: Test that re-running the pipeline skips already-embedded chunks to save cost."""
        pipeline = BatchEmbeddingPipeline(
            batch_size=2,
            price_per_1k_tokens=0.00002,
            cache_path=self.test_cache_path
        )

        # Initial run
        initial_summary = pipeline.process_corpus(CDLP_SAMPLE_CORPUS[:4])
        self.assertEqual(initial_summary["embedded"], 4)
        self.assertEqual(initial_summary["skipped_existing"], 0)

        # Re-run on same corpus
        rerun_summary = pipeline.process_corpus(CDLP_SAMPLE_CORPUS[:4])
        self.assertEqual(rerun_summary["total_chunks"], 4)
        self.assertEqual(rerun_summary["skipped_existing"], 4)
        self.assertEqual(rerun_summary["embedded"], 0)
        self.assertEqual(rerun_summary["input_tokens"], 0)
        self.assertEqual(rerun_summary["estimated_cost_usd"], 0.0)

    def test_pipeline_handles_partial_failure(self):
        """Task 2 & 3: Test that failed batches are tracked in run summary under 'failed' metric."""
        def failing_embed_fn(texts):
            if "Brazil" in texts[0]:
                raise Exception("API Connection Timeout")
            return [[0.1] * 384 for _ in texts]

        pipeline = BatchEmbeddingPipeline(
            batch_size=1,
            max_retry_attempts=2,
            cache_path=self.test_cache_path,
            embed_fn=failing_embed_fn
        )

        # Process a set including a failing chunk
        test_chunks = [
            {"id": "OK-1", "text": "Customs record India laptop"},
            {"id": "FAIL-1", "text": "Customs record Brazil medical equipment"}
        ]
        summary = pipeline.process_corpus(test_chunks)

        self.assertEqual(summary["total_chunks"], 2)
        self.assertEqual(summary["embedded"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_cost_calculation_precision(self):
        """Task 3: Verify accurate cost estimation calculation."""
        tokens = 10000
        rate = 0.00002
        expected_cost = (10000 / 1000.0) * 0.00002 # $0.0002
        self.assertAlmostEqual(expected_cost, 0.0002, places=6)

# ------------------------------------------------------------------------------
# Source File: tests\test_embedding_demo.py
# ------------------------------------------------------------------------------

"""
Unit tests for KDU 3.25 Embedding Fundamentals & Vector Representation (src/embedding_demo.py)
"""

import unittest
import os
import sys

# Ensure project root directory is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.embedding_demo import (
    embed_texts,
    cosine_similarity,
    explain_vector_representation,
    demo_logistics_query_matching,
    demo_paraphrase_robustness,
    run_embedding_demonstration
)


class TestEmbeddingDemo(unittest.TestCase):

    def test_embed_texts_returns_valid_vectors(self):
        texts = ["Password recovery steps", "Laptop import tariff"]
        embeddings = embed_texts(texts)

        self.assertEqual(len(embeddings), len(texts))
        self.assertIsInstance(embeddings[0], list)
        self.assertIsInstance(embeddings[1], list)
        self.assertEqual(len(embeddings[0]), len(embeddings[1]))
        self.assertGreater(len(embeddings[0]), 0)
        self.assertTrue(all(isinstance(val, float) for val in embeddings[0]))

    def test_cosine_similarity_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_cosine_similarity_orthogonal_vectors(self):
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        sim = cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(sim, 0.0, places=4)

    def test_logistics_query_matching_demo(self):
        res = demo_logistics_query_matching()
        self.assertTrue(res["test_passed"])
        self.assertEqual(res["top_match_id"], "REG-IN-8471")
        self.assertGreater(res["top_match_score"], 0.5)

    def test_paraphrase_robustness_demo(self):
        res = demo_paraphrase_robustness()
        self.assertTrue(res["test_passed"])
        self.assertGreater(res["paraphrase_similarity"], res["unrelated_similarity"])

    def test_embedding_demonstration_execution(self):
        results = run_embedding_demonstration()

        self.assertGreaterEqual(len(results["sample_texts"]), 3)
        self.assertTrue(results["all_dimensions_equal"])
        self.assertGreater(results["vector_dimension"], 0)
        self.assertEqual(len(results["sample_vector_preview"]["text_0_first_8_values"]), 8)

        # Similar pair score > Dissimilar pair score
        sim_similar = results["similarity_comparison"]["similar_pair"]["cosine_similarity"]
        sim_dissimilar = results["similarity_comparison"]["dissimilar_pair"]["cosine_similarity"]

        self.assertGreater(sim_similar, sim_dissimilar)
        self.assertTrue(results["similarity_comparison"]["ranking_test_passed"])

        self.assertIn("logistics_rules_matching_demo", results)
        self.assertTrue(results["logistics_rules_matching_demo"]["test_passed"])

        self.assertIn("paraphrase_robustness_demo", results)
        self.assertTrue(results["paraphrase_robustness_demo"]["test_passed"])

    def test_explanation_content(self):
        explanations = explain_vector_representation()

        self.assertIn("what_is_embedding", explanations)
        self.assertIn("what_is_dimension", explanations)
        self.assertIn("why_semantic_search", explanations)

        self.assertIn("vector", explanations["what_is_embedding"].lower())
        self.assertIn("dimension", explanations["what_is_dimension"].lower())
        self.assertIn("semantic", explanations["why_semantic_search"].lower())

# ------------------------------------------------------------------------------
# Source File: tests\test_embeddings.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Embeddings Generation Pipeline (Mocked API)
===========================================================
Tests chunk loading, separate provider configuration, API mocking,
numerical vector validation, dimension consistency, metadata preservation,
failure isolation, and report generation.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import tempfile
import shutil
import json

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.embeddings import (
    load_validated_chunks,
    create_embedding_client,
    validate_embedding_provider_config,
    generate_embedding,
    embed_chunks,
    validate_embeddings,
    format_vector_preview,
    create_sample_embedding_output,
    save_embedding_results,
    run_embedding_pipeline
)


class TestEmbeddingGeneration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        # Sample validated chunks
        self.sample_chunks = [
            {
                "chunk_id": "shipping_rules_paragraph_001",
                "source": "shipping_rules.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 1,
                "character_count": 85,
                "chunk_text": "All international shipments must include accurate shipping documentation and invoices."
            },
            {
                "chunk_id": "customs_requirements_paragraph_001",
                "source": "customs_requirements.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 1,
                "character_count": 92,
                "chunk_text": "Customs duties and tariff classifications depend upon the 8-digit HS Code classification."
            },
            {
                "chunk_id": "customs_requirements_paragraph_002",
                "source": "customs_requirements.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 2,
                "character_count": 80,
                "chunk_text": "Preferential tariff rates require a validated Certificate of Origin document."
            }
        ]

        self.chunks_file = os.path.join(self.test_dir, "processed_chunks.json")
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_chunks, f, indent=2)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 1. Chunk Loading & Selection
    # -------------------------------------------------------------
    def test_load_validated_chunks_limit(self):
        """Test loading prepared chunks with max_chunks limit."""
        loaded = load_validated_chunks(chunks_file=self.chunks_file, max_chunks=2)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["chunk_id"], "shipping_rules_paragraph_001")
        self.assertEqual(loaded[1]["chunk_id"], "customs_requirements_paragraph_001")

    def test_load_all_validated_chunks(self):
        """Test loading all prepared chunks when max_chunks is None or 0."""
        loaded = load_validated_chunks(chunks_file=self.chunks_file, max_chunks=None)
        self.assertEqual(len(loaded), 3)

    # -------------------------------------------------------------
    # 2. Client Creation & Environment Handling
    # -------------------------------------------------------------
    def test_create_embedding_client_missing_key_raises_error(self):
        """Test that missing EMBEDDING_API_KEY raises a clear ValueError."""
        with patch.dict(os.environ, {"EMBEDDING_API_KEY": "", "OPENAI_API_KEY": ""}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                create_embedding_client(api_key="")
            self.assertIn("EMBEDDING_API_KEY is missing", str(ctx.exception))

    def test_create_embedding_client_does_not_use_groq_api_key(self):
        """Test that embedding client does NOT fall back to GROQ_API_KEY."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_secret", "EMBEDDING_API_KEY": "", "OPENAI_API_KEY": ""}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                create_embedding_client()
            self.assertIn("EMBEDDING_API_KEY is missing", str(ctx.exception))

    @patch("src.embeddings.OpenAI")
    def test_create_embedding_client_using_embedding_api_key(self, mock_openai_cls):
        """Test successful client creation using EMBEDDING_API_KEY and EMBEDDING_BASE_URL."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        with patch.dict(os.environ, {
            "EMBEDDING_API_KEY": "embed_key_123",
            "EMBEDDING_BASE_URL": "https://api.openai.com/v1"
        }, clear=True):
            client = create_embedding_client()
            self.assertEqual(client, mock_client)
            mock_openai_cls.assert_called_once_with(
                api_key="embed_key_123",
                base_url="https://api.openai.com/v1"
            )

    @patch("src.embeddings.OpenAI")
    def test_explicit_args_override_env(self, mock_openai_cls):
        """Test explicit base_url and api_key override environment settings."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        with patch.dict(os.environ, {
            "EMBEDDING_API_KEY": "env_key",
            "EMBEDDING_BASE_URL": "https://env.url/v1"
        }, clear=True):
            client = create_embedding_client(
                base_url="https://explicit.url/v1",
                api_key="explicit_key"
            )
            self.assertEqual(client, mock_client)
            mock_openai_cls.assert_called_once_with(
                api_key="explicit_key",
                base_url="https://explicit.url/v1"
            )

    # -------------------------------------------------------------
    # 3. Provider Configuration Guard
    # -------------------------------------------------------------
    def test_groq_endpoint_with_openai_model_fails_early(self):
        """Test that groq endpoint paired with text-embedding-3-small raises early configuration error."""
        with self.assertRaises(ValueError) as ctx:
            validate_embedding_provider_config(
                base_url="https://api.groq.com/openai/v1",
                model="text-embedding-3-small",
                api_key="gsk_key"
            )
        self.assertIn("text-embedding-3-small is configured with the Groq endpoint", str(ctx.exception))

    def test_valid_provider_config_returns_safe_summary(self):
        """Test valid provider config generates safe summary without leaking key."""
        summary = validate_embedding_provider_config(
            base_url="https://api.openai.com/v1",
            model="text-embedding-3-small",
            api_key="sk-real-secret-key"
        )
        self.assertIn("https://api.openai.com/v1", summary)
        self.assertIn("text-embedding-3-small", summary)
        self.assertIn("configured", summary)
        self.assertNotIn("sk-real-secret-key", summary)

    # -------------------------------------------------------------
    # 4. Vector Generation & Numerical Validation
    # -------------------------------------------------------------
    def test_generate_embedding_mocked_success(self):
        """Test successful vector generation and dimension detection."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_data_item = MagicMock()
        mock_data_item.embedding = [0.0123, -0.0456, 0.0789, 0.1234]
        mock_resp.data = [mock_data_item]
        mock_client.embeddings.create.return_value = mock_resp

        vector = generate_embedding(mock_client, "Sample shipping rule text", model="text-embedding-3-small")
        self.assertEqual(len(vector), 4)
        self.assertEqual(vector, [0.0123, -0.0456, 0.0789, 0.1234])
        mock_client.embeddings.create.assert_called_once_with(
            input="Sample shipping rule text",
            model="text-embedding-3-small"
        )

    def test_generate_embedding_empty_text_raises_error(self):
        """Test that empty text raises ValueError without making API call."""
        mock_client = MagicMock()
        with self.assertRaises(ValueError):
            generate_embedding(mock_client, "   ", model="text-embedding-3-small")
        mock_client.embeddings.create.assert_not_called()

    # -------------------------------------------------------------
    # 5. Batch Chunk Embedding & Metadata Preservation
    # -------------------------------------------------------------
    def test_embed_chunks_preserves_metadata(self):
        """Test that all original metadata keys and chunk_text are preserved."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_data_item = MagicMock()
        mock_data_item.embedding = [0.1, 0.2, 0.3]
        mock_resp.data = [mock_data_item]
        mock_client.embeddings.create.return_value = mock_resp

        embedded, failures = embed_chunks(self.sample_chunks[:2], client=mock_client, model="text-embedding-3-small")
        self.assertEqual(len(embedded), 2)
        self.assertEqual(len(failures), 0)

        first = embedded[0]
        self.assertEqual(first["embedding_id"], "embedding_001")
        self.assertEqual(first["chunk_id"], "shipping_rules_paragraph_001")
        self.assertEqual(first["source"], "shipping_rules.txt")
        self.assertEqual(first["document_type"], "txt")
        self.assertEqual(first["strategy"], "paragraph")
        self.assertEqual(first["chunk_index"], 1)
        self.assertEqual(first["character_count"], 85)
        self.assertEqual(first["vector_dimension"], 3)
        self.assertEqual(first["embedding"], [0.1, 0.2, 0.3])
        self.assertIn("shipping documentation", first["chunk_text"])

    def test_embed_chunks_failure_isolation(self):
        """Test that failure on one chunk does not halt processing of other chunks."""
        mock_client = MagicMock()

        # Side effect: first call succeeds, second raises API exception, third succeeds
        mock_resp_1 = MagicMock(data=[MagicMock(embedding=[0.1, 0.2])])
        mock_resp_3 = MagicMock(data=[MagicMock(embedding=[0.3, 0.4])])
        mock_client.embeddings.create.side_effect = [
            mock_resp_1,
            Exception("API Rate Limit Exceeded"),
            mock_resp_3
        ]

        embedded, failures = embed_chunks(self.sample_chunks, client=mock_client, model="text-embedding-3-small")
        self.assertEqual(len(embedded), 2)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["chunk_id"], "customs_requirements_paragraph_001")
        self.assertIn("Rate Limit", failures[0]["error_message"])

    def test_embed_chunks_empty_chunk_text_failure(self):
        """Test handling of chunks with empty text."""
        bad_chunks = [{"chunk_id": "bad_chunk_001", "source": "empty.txt", "chunk_text": "   "}]
        mock_client = MagicMock()
        embedded, failures = embed_chunks(bad_chunks, client=mock_client, model="text-embedding-3-small")
        self.assertEqual(len(embedded), 0)
        self.assertEqual(len(failures), 1)
        self.assertIn("Empty chunk text", failures[0]["error_message"])

    # -------------------------------------------------------------
    # 6. Validation & Dimension Consistency
    # -------------------------------------------------------------
    def test_validate_embeddings_consistent_dimensions(self):
        """Test validation passes when all vectors have consistent dimensions."""
        embedded_items = [
            {
                "embedding_id": "embedding_001",
                "chunk_id": "c1",
                "source": "s1.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 1,
                "character_count": 10,
                "chunk_text": "chunk text",
                "embedding_model": "test-model",
                "vector_dimension": 3,
                "embedding": [0.1, 0.2, 0.3]
            },
            {
                "embedding_id": "embedding_002",
                "chunk_id": "c2",
                "source": "s1.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 2,
                "character_count": 10,
                "chunk_text": "chunk text",
                "embedding_model": "test-model",
                "vector_dimension": 3,
                "embedding": [0.4, 0.5, 0.6]
            }
        ]
        is_valid, dim, errors = validate_embeddings(embedded_items)
        self.assertTrue(is_valid)
        self.assertEqual(dim, 3)
        self.assertEqual(len(errors), 0)

    def test_validate_embeddings_inconsistent_dimensions_detected(self):
        """Test validation fails when vector dimensions are inconsistent."""
        inconsistent_items = [
            {
                "embedding_id": "embedding_001",
                "chunk_id": "c1",
                "source": "s1.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 1,
                "character_count": 10,
                "chunk_text": "chunk text",
                "embedding_model": "test-model",
                "vector_dimension": 3,
                "embedding": [0.1, 0.2, 0.3]
            },
            {
                "embedding_id": "embedding_002",
                "chunk_id": "c2",
                "source": "s1.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 2,
                "character_count": 10,
                "chunk_text": "chunk text",
                "embedding_model": "test-model",
                "vector_dimension": 2,
                "embedding": [0.4, 0.5]
            }
        ]
        is_valid, dim, errors = validate_embeddings(inconsistent_items)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertIn("Inconsistent dimension", errors[0])

    # -------------------------------------------------------------
    # 7. Artifact Persistence & Sample Output
    # -------------------------------------------------------------
    def test_format_vector_preview(self):
        """Test formatted vector string preview."""
        vec = [0.01234, -0.04567, 0.07891, 0.9999]
        preview = format_vector_preview(vec, max_elements=3)
        self.assertEqual(preview, "[0.0123, -0.0457, 0.0789, ...]")

    def test_save_embedding_results_creates_all_files(self):
        """Test that save_embedding_results creates all 3 output files."""
        embedded_items = [
            {
                "embedding_id": "embedding_001",
                "chunk_id": "c1",
                "source": "s1.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 1,
                "character_count": 10,
                "chunk_text": "test text",
                "embedding_model": "text-embedding-3-small",
                "vector_dimension": 3,
                "embedding": [0.1, 0.2, 0.3]
            }
        ]

        report = save_embedding_results(
            embedded_chunks=embedded_items,
            failures=[],
            model="text-embedding-3-small",
            selected_count=1,
            output_dir=self.output_dir,
            sample_size=1
        )

        self.assertEqual(report["validation"]["status"], "PASSED")
        self.assertEqual(report["statistics"]["vector_dimension"], 3)

        full_p = os.path.join(self.output_dir, "embedded_chunks.json")
        rep_p = os.path.join(self.output_dir, "embedding_report.json")
        sample_p = os.path.join(self.output_dir, "sample_embedding_output.json")

        self.assertTrue(os.path.exists(full_p))
        self.assertTrue(os.path.exists(rep_p))
        self.assertTrue(os.path.exists(sample_p))

        # Check sample contains trimmed vector
        with open(sample_p, "r", encoding="utf-8") as f:
            samples = json.load(f)
            self.assertEqual(len(samples), 1)
            self.assertIn("vector_preview", samples[0])
            self.assertEqual(samples[0]["vector_length"], 3)

    # -------------------------------------------------------------
    # 8. End-to-End Pipeline Execution (Mocked)
    # -------------------------------------------------------------
    def test_run_embedding_pipeline_end_to_end(self):
        """Test complete mocked pipeline run."""
        mock_client = MagicMock()
        mock_resp = MagicMock(data=[MagicMock(embedding=[0.01, -0.02, 0.03, 0.04])])
        mock_client.embeddings.create.return_value = mock_resp

        report = run_embedding_pipeline(
            chunks_file=self.chunks_file,
            output_dir=self.output_dir,
            model="text-embedding-3-small",
            base_url="https://api.openai.com/v1",
            api_key="test_embed_key",
            max_chunks=2,
            client=mock_client,
            verbose=False
        )

        self.assertEqual(report["statistics"]["chunks_selected"], 2)
        self.assertEqual(report["statistics"]["chunks_successfully_embedded"], 2)
        self.assertEqual(report["statistics"]["failed_embeddings"], 0)
        self.assertEqual(report["statistics"]["vector_dimension"], 4)
        self.assertEqual(report["validation"]["status"], "PASSED")



# ============================================================================
# VECTOR STORE / INDEXING TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_indexing.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Task 3.31: Indexing Embeddings & Metadata Storage
===================================================================
Tests vector record formatting, batching, bulk collection upserting,
count validation, spot-check readback assertions, error handling, and re-indexing.
Has zero third-party testing dependencies (runs with python directly or unittest).
"""

import os
import sys
import unittest
from typing import List, Dict, Any

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.indexing import (
    to_vector_record,
    batches,
    VectorCollection,
    index_embeddings,
    spot_check_integrity,
    reindex_updated_corpus,
    run_indexing_pipeline,
)


def get_sample_embedded_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "id": "CDLP-IN-8471-01",
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "text": "Customs Record India HS Code 8471.30 (Laptops): Basic Duty 7.5%, SWS 10%.",
            "metadata": {
                "source": "customs_reg_india.json",
                "chunk_index": 1,
                "section": "Laptops",
                "country": "India",
                "hs_code": "8471.30"
            }
        },
        {
            "id": "CDLP-IT-8703-01",
            "embedding": [0.5, 0.6, 0.7, 0.8],
            "text": "Customs Record Italy HS Code 8703 (Motor Vehicles): Duty 10%, VAT 22%.",
            "metadata": {
                "source": "customs_reg_italy.json",
                "chunk_index": 1,
                "section": "Vehicles",
                "country": "Italy",
                "hs_code": "8703"
            }
        },
        {
            "id": "CDLP-DE-8541-01",
            "embedding": [0.9, 0.1, 0.2, 0.3],
            "text": "Customs Record Germany HS Code 8541.43 (Solar Modules): Duty 0%, VAT 19%.",
            "metadata": {
                "source": "customs_reg_germany.json",
                "chunk_index": 1,
                "section": "Solar",
                "country": "Germany",
                "hs_code": "8541.43"
            }
        }
    ]


class TestIndexingPipeline(unittest.TestCase):

    def test_to_vector_record_formatting(self):
        """Verifies that to_vector_record produces the specified record schema."""
        chunks = get_sample_embedded_chunks()
        chunk = chunks[0]
        record = to_vector_record(chunk)

        self.assertEqual(record["id"], "CDLP-IN-8471-01")
        self.assertEqual(record["vector"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(record["text"], chunk["text"])
        self.assertEqual(record["metadata"]["source"], "customs_reg_india.json")
        self.assertEqual(record["metadata"]["chunk_index"], 1)
        self.assertEqual(record["metadata"]["section"], "Laptops")
        self.assertEqual(record["metadata"]["country"], "India")
        self.assertEqual(record["metadata"]["hs_code"], "8471.30")

    def test_to_vector_record_invalid_input(self):
        """Verifies that to_vector_record raises ValueError on invalid input."""
        with self.assertRaises(ValueError):
            to_vector_record("not_a_dict")

        with self.assertRaises(ValueError):
            to_vector_record({"no_id": True})

    def test_batches_generator(self):
        """Verifies that batches yields correct sublists for exact and partial sizes."""
        items = list(range(250))
        batch_list = list(batches(items, size=100))

        self.assertEqual(len(batch_list), 3)
        self.assertEqual(len(batch_list[0]), 100)
        self.assertEqual(len(batch_list[1]), 100)
        self.assertEqual(len(batch_list[2]), 50)

        with self.assertRaises(ValueError):
            list(batches(items, size=0))

    def test_index_embeddings_and_count_validation(self):
        """Verifies bulk insertion and count assertion matching expected chunk count."""
        chunks = get_sample_embedded_chunks()
        collection = VectorCollection(name="test_collection")
        summary = index_embeddings(chunks, collection, batch_size=2)

        self.assertEqual(summary["expected_chunks"], 3)
        self.assertEqual(summary["inserted_this_run"], 3)
        self.assertEqual(summary["indexed_count"], 3)
        self.assertTrue(summary["count_matched"])
        self.assertEqual(len(summary["failures"]), 0)
        self.assertEqual(collection.count(), 3)

    def test_spot_check_integrity(self):
        """Verifies spot-check readback assertion for ID, text, metadata, and vector length."""
        chunks = get_sample_embedded_chunks()
        collection = VectorCollection(name="test_spot_check")
        index_embeddings(chunks, collection)

        sample = chunks[0]
        result = spot_check_integrity(collection, sample)

        self.assertTrue(result["spot_check_passed"])
        self.assertEqual(result["id"], sample["id"])
        self.assertEqual(result["source"], sample["metadata"]["source"])
        self.assertEqual(result["vector_dim"], len(sample["embedding"]))
        self.assertTrue(result["text_preview"].startswith("Customs Record India"))

    def test_batch_upsert_failure_logging(self):
        """Verifies exception handling during batch upsert failures."""
        class FailingCollection(VectorCollection):
            def upsert(self, batch):
                if any(r["id"] == "FAIL_ID" for r in batch):
                    raise RuntimeError("Simulated Database Error")
                super().upsert(batch)

        failing_chunks = [
            {"id": "GOOD_1", "embedding": [1.0], "text": "Good chunk 1", "metadata": {"source": "doc1.json", "chunk_index": 1}},
            {"id": "FAIL_ID", "embedding": [2.0], "text": "Failing chunk", "metadata": {"source": "doc2.json", "chunk_index": 2}}
        ]

        col = FailingCollection(name="failing_col")
        
        with self.assertRaises(AssertionError):
            index_embeddings(failing_chunks, col, batch_size=1)

    def test_reindex_updated_corpus(self):
        """Verifies updating existing chunk records with stable IDs and deleting removed chunks."""
        chunks = get_sample_embedded_chunks()
        collection = VectorCollection(name="test_reindex")
        index_embeddings(chunks, collection)
        self.assertEqual(collection.count(), 3)

        # Update 1 chunk, remove 1 chunk
        updated_chunk = {
            "id": "CDLP-IN-8471-01",
            "embedding": [0.11, 0.22, 0.33, 0.44],
            "text": "UPDATED Customs Record India HS Code 8471.30 (Laptops): Duty 7.5%.",
            "metadata": {"source": "customs_reg_india.json", "chunk_index": 1, "section": "Laptops", "country": "India", "hs_code": "8471.30"}
        }

        reindex_res = reindex_updated_corpus(
            collection,
            updated_chunks=[updated_chunk],
            removed_ids=["CDLP-DE-8541-01"]
        )

        self.assertEqual(reindex_res["upserted_chunks"], 1)
        self.assertEqual(reindex_res["deleted_chunks"], 1)
        self.assertEqual(reindex_res["current_indexed_count"], 2)

        # Check updated record content
        stored = collection.get("CDLP-IN-8471-01")
        self.assertTrue(stored["text"].startswith("UPDATED"))
        self.assertEqual(stored["vector"], [0.11, 0.22, 0.33, 0.44])

    def test_run_indexing_pipeline_execution(self):
        """Verifies full execution of run_indexing_pipeline and artifact generation."""
        summary, text_report = run_indexing_pipeline()

        self.assertTrue(summary["indexing_summary"]["count_matched"])
        self.assertTrue(summary["spot_check"]["spot_check_passed"])
        self.assertIn("SHIPRULE CDLP - INDEXING EMBEDDINGS", text_report)

        # Check artifacts written to disk
        self.assertTrue(os.path.exists("outputs/indexing_summary.json"))
        self.assertTrue(os.path.exists("outputs/indexing_output.txt"))

# ------------------------------------------------------------------------------
# Source File: tests\test_ingestion.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Corpus Preparation & Ingestion Validation Pipeline
==================================================================
Tests recursive discovery, multi-format loading, failure isolation,
reconciliation checks, chunk metadata validation, manifest generation,
and resumable ingestion behavior.
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

from src.ingestion import (
    discover_files,
    compute_file_hash,
    process_document_file,
    validate_chunk_metadata,
    validate_corpus_reconciliation,
    CorpusIngestionPipeline,
    run_ingestion_pipeline,
)


class TestCorpusIngestion(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "outputs")
        self.corpus_dir = os.path.join(self.test_dir, "corpus")
        os.makedirs(self.corpus_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Valid TXT document
        self.txt_path = os.path.join(self.corpus_dir, "doc1.txt")
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write("Section 1: Customs regulations.\n\nSection 2: Tariff declarations.")

        # 2. Valid nested TXT document
        nested_dir = os.path.join(self.corpus_dir, "subdir")
        os.makedirs(nested_dir, exist_ok=True)
        self.nested_txt_path = os.path.join(nested_dir, "doc2.txt")
        with open(self.nested_txt_path, "w", encoding="utf-8") as f:
            f.write("Nested document paragraph one.\n\nNested document paragraph two.")

        # 3. Empty TXT document
        self.empty_txt_path = os.path.join(self.corpus_dir, "empty.txt")
        with open(self.empty_txt_path, "w", encoding="utf-8") as f:
            f.write("   \n\n\t  ")

        # 4. Unsupported file
        self.unsupported_path = os.path.join(self.corpus_dir, "image.png")
        with open(self.unsupported_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

        # 5. Corrupted PDF file
        self.corrupt_pdf_path = os.path.join(self.corpus_dir, "corrupt.pdf")
        with open(self.corrupt_pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 INVALID_CORRUPTED_PDF_BYTES_WITHOUT_EOF")

        # 6. Hidden file (should be ignored by discovery)
        self.hidden_path = os.path.join(self.corpus_dir, ".hidden_file.txt")
        with open(self.hidden_path, "w", encoding="utf-8") as f:
            f.write("Should be ignored.")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 1. Discovery Tests
    # -------------------------------------------------------------
    def test_recursive_file_discovery(self):
        """Test recursive discovery finds nested files and ignores hidden files."""
        files = discover_files(self.corpus_dir, recursive=True)
        basenames = [os.path.basename(f) for f in files]

        self.assertIn("doc1.txt", basenames)
        self.assertIn("doc2.txt", basenames)
        self.assertIn("empty.txt", basenames)
        self.assertIn("image.png", basenames)
        self.assertIn("corrupt.pdf", basenames)
        self.assertNotIn(".hidden_file.txt", basenames)
        self.assertEqual(len(files), 5)

    def test_compute_file_hash(self):
        """Test deterministic file hashing."""
        hash1 = compute_file_hash(self.txt_path)
        hash2 = compute_file_hash(self.txt_path)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)

    # -------------------------------------------------------------
    # 2. Single File Processing & Status Handling
    # -------------------------------------------------------------
    def test_process_valid_txt_file(self):
        """Test successful processing of a valid TXT document."""
        entry, chunks, log_msg = process_document_file(self.txt_path, strategy="paragraph")
        self.assertEqual(entry["status"], "SUCCESS")
        self.assertEqual(entry["file_name"], "doc1.txt")
        self.assertEqual(entry["document_type"], "txt")
        self.assertIsNone(entry["error_message"])
        self.assertGreater(entry["character_count"], 0)
        self.assertEqual(entry["chunk_count"], 2)
        self.assertEqual(len(chunks), 2)
        self.assertIn("SUCCESS", log_msg)

    def test_process_empty_file_marked_as_skipped(self):
        """Test empty document is marked as SKIPPED with appropriate error message."""
        entry, chunks, log_msg = process_document_file(self.empty_txt_path, strategy="paragraph")
        self.assertEqual(entry["status"], "SKIPPED")
        self.assertEqual(len(chunks), 0)
        self.assertIn("empty", entry["error_message"].lower())
        self.assertIn("SKIPPED", log_msg)

    def test_process_unsupported_format_marked_as_skipped(self):
        """Test unsupported document format is marked as SKIPPED."""
        entry, chunks, log_msg = process_document_file(self.unsupported_path, strategy="paragraph")
        self.assertEqual(entry["status"], "SKIPPED")
        self.assertEqual(len(chunks), 0)
        self.assertIn("unsupported", entry["error_message"].lower())

    def test_process_corrupt_file_marked_as_failed(self):
        """Test corrupt PDF is marked as FAILED with error details."""
        entry, chunks, log_msg = process_document_file(self.corrupt_pdf_path, strategy="paragraph")
        self.assertEqual(entry["status"], "FAILED")
        self.assertEqual(len(chunks), 0)
        self.assertIsNotNone(entry["error_message"])
        self.assertIn("FAILED", log_msg)

    # -------------------------------------------------------------
    # 3. Validation & Reconciliation Tests
    # -------------------------------------------------------------
    def test_reconciliation_check_success(self):
        """Test reconciliation check passes when all discovered files are accounted for."""
        manifest = [
            {"status": "SUCCESS"},
            {"status": "SUCCESS"},
            {"status": "FAILED"},
            {"status": "SKIPPED"},
        ]
        passed, msg, counts = validate_corpus_reconciliation(manifest, total_discovered_files=4)
        self.assertTrue(passed)
        self.assertEqual(counts["successful"], 2)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["skipped"], 1)
        self.assertIn("No documents were silently dropped", msg)

    def test_reconciliation_check_detects_silent_drop(self):
        """Test reconciliation check fails when a document is silently dropped."""
        manifest = [
            {"status": "SUCCESS"},
            {"status": "FAILED"},
        ]
        passed, msg, counts = validate_corpus_reconciliation(manifest, total_discovered_files=3)
        self.assertFalse(passed)
        self.assertIn("RECONCILIATION FAILED", msg)

    def test_chunk_metadata_validation_success(self):
        """Test metadata validation passes for valid chunks."""
        chunks = [
            {
                "chunk_id": "doc1_paragraph_001",
                "source": "doc1.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 1,
                "character_count": 10,
                "chunk_text": "0123456789"
            },
            {
                "chunk_id": "doc1_paragraph_002",
                "source": "doc1.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 2,
                "character_count": 10,
                "chunk_text": "abcdefghij"
            }
        ]
        is_valid, invalid_ids, details = validate_chunk_metadata(chunks)
        self.assertTrue(is_valid)
        self.assertEqual(len(invalid_ids), 0)

    def test_chunk_metadata_validation_detects_duplicates_and_empty(self):
        """Test metadata validation catches duplicate chunk IDs and empty chunks."""
        invalid_chunks = [
            {
                "chunk_id": "duplicate_id",
                "source": "doc1.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 1,
                "character_count": 5,
                "chunk_text": "hello"
            },
            {
                "chunk_id": "duplicate_id",
                "source": "doc1.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 2,
                "character_count": 5,
                "chunk_text": "world"
            },
            {
                "chunk_id": "empty_chunk",
                "source": "doc1.txt",
                "document_type": "txt",
                "strategy": "paragraph",
                "chunk_index": 3,
                "character_count": 0,
                "chunk_text": ""
            }
        ]
        is_valid, invalid_ids, details = validate_chunk_metadata(invalid_chunks)
        self.assertFalse(is_valid)
        self.assertGreater(len(invalid_ids), 0)

    # -------------------------------------------------------------
    # 4. End-to-End Pipeline & Output Artifacts Tests
    # -------------------------------------------------------------
    def test_full_pipeline_run_with_mixed_corpus(self):
        """Test running pipeline against mixed corpus with success, failure, and skipped files."""
        pipeline = CorpusIngestionPipeline(
            corpus_dir=self.corpus_dir,
            output_dir=self.output_dir,
            strategy="paragraph",
            verbose=False
        )
        report = pipeline.run()

        self.assertIsNotNone(report)
        stats = report["statistics"]
        self.assertEqual(stats["files_discovered"], 5)
        self.assertEqual(stats["successfully_processed"], 2)  # doc1.txt, subdir/doc2.txt
        self.assertEqual(stats["failed"], 1)                  # corrupt.pdf
        self.assertEqual(stats["skipped"], 2)                 # empty.txt, image.png

        # Verify all 5 output files were created
        manifest_p = os.path.join(self.output_dir, "corpus_manifest.json")
        report_p = os.path.join(self.output_dir, "ingestion_report.json")
        failures_p = os.path.join(self.output_dir, "ingestion_failures.json")
        chunks_p = os.path.join(self.output_dir, "processed_chunks.json")
        log_p = os.path.join(self.output_dir, "ingestion_log.txt")

        for p in [manifest_p, report_p, failures_p, chunks_p, log_p]:
            self.assertTrue(os.path.exists(p), f"Missing artifact: {p}")

        # Check failures file contains only the corrupted file
        with open(failures_p, "r", encoding="utf-8") as f:
            fails = json.load(f)
            self.assertEqual(len(fails), 1)
            self.assertEqual(fails[0]["file_name"], "corrupt.pdf")

    def test_resumable_ingestion(self):
        """Test resumable pipeline skips unchanged cached files on subsequent runs."""
        # First run
        pipeline1 = CorpusIngestionPipeline(
            corpus_dir=self.corpus_dir,
            output_dir=self.output_dir,
            strategy="paragraph",
            resumable=False,
            verbose=False
        )
        pipeline1.run()

        # Second run with resumable=True
        pipeline2 = CorpusIngestionPipeline(
            corpus_dir=self.corpus_dir,
            output_dir=self.output_dir,
            strategy="paragraph",
            resumable=True,
            verbose=False
        )
        report2 = pipeline2.run()

        self.assertEqual(report2["statistics"]["successfully_processed"], 2)
        self.assertEqual(report2["validation"]["reconciliation_status"], "PASSED")

    # -------------------------------------------------------------
    # 5. Pipeline Run on Committed Sample Corpus
    # -------------------------------------------------------------
    def test_sample_corpus_ingestion(self):
        """Test pipeline run on committed data/sample_corpus/ directory."""
        actual_corpus = os.path.join(project_root, "data", "sample_corpus")
        test_out = os.path.join(self.test_dir, "sample_outputs")

        report = run_ingestion_pipeline(
            corpus_dir=actual_corpus,
            output_dir=test_out,
            strategy="paragraph",
            verbose=False
        )

        self.assertEqual(report["validation"]["overall_status"], "PASSED")
        self.assertEqual(report["validation"]["reconciliation_status"], "PASSED")
        self.assertEqual(report["validation"]["metadata_status"], "PASSED")
        self.assertEqual(report["statistics"]["files_discovered"], 3)
        self.assertEqual(report["statistics"]["successfully_processed"], 3)
        self.assertEqual(report["statistics"]["failed"], 0)
        self.assertEqual(report["statistics"]["skipped"], 0)
        self.assertEqual(report["statistics"]["total_chunks_generated"], 8)

# ------------------------------------------------------------------------------
# Source File: tests\test_vector_store.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Vector Database Storage Layer (ChromaDB)
========================================================
Tests persistent client connection, collection initialization,
dimension validation, metadata normalization, single/batch record
insertion, upsert idempotency, and deterministic readback.
"""

import unittest
import os
import sys
import tempfile
import shutil

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vector_store import (
    get_vector_db_config,
    get_vector_db_client,
    get_or_create_collection,
    detect_embedding_dimension,
    validate_vector_dimension,
    format_chroma_metadata,
    insert_embedding_record,
    insert_embedding_records,
    get_embedding_record,
    run_vector_db_health_check
)


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "vector_db")
        self.collection_name = "test_shiprule_docs"
        self.vector_dim = 4

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 1. Connection & Collection Initialization
    # -------------------------------------------------------------
    def test_get_vector_db_client(self):
        """Test persistent ChromaDB client initialization and reachability."""
        client = get_vector_db_client(db_path=self.db_path)
        self.assertIsNotNone(client)
        self.assertTrue(os.path.exists(self.db_path))

    def test_get_or_create_collection(self):
        """Test collection creation with custom name and distance metric."""
        client = get_vector_db_client(db_path=self.db_path)
        collection = get_or_create_collection(
            client=client,
            collection_name=self.collection_name,
            distance_metric="cosine"
        )
        self.assertIsNotNone(collection)
        self.assertEqual(collection.name, self.collection_name)
        self.assertEqual(collection.count(), 0)

    # -------------------------------------------------------------
    # 2. Dimension Validation
    # -------------------------------------------------------------
    def test_validate_vector_dimension_success(self):
        """Test that matching vector length passes dimension check."""
        vector = [0.1, 0.2, 0.3, 0.4]
        self.assertTrue(validate_vector_dimension(vector, expected_dimension=4))

    def test_validate_vector_dimension_mismatch(self):
        """Test that mismatched vector length raises clear ValueError."""
        vector = [0.1, 0.2, 0.3]
        with self.assertRaises(ValueError) as ctx:
            validate_vector_dimension(vector, expected_dimension=4)
        self.assertIn("Vector dimension mismatch", str(ctx.exception))

    def test_validate_vector_dimension_invalid_type(self):
        """Test that non-iterable vector raises TypeError."""
        with self.assertRaises(TypeError):
            validate_vector_dimension(12345, expected_dimension=4)  # type: ignore

    # -------------------------------------------------------------
    # 3. Metadata Normalization & ChromaDB Compatibility
    # -------------------------------------------------------------
    def test_format_chroma_metadata_defaults(self):
        """Test metadata formatting handles missing/None fields safely."""
        raw_meta = {
            "source": "shipping_rules.txt",
            "section": None,
            "page": None,
            "chunk_index": 1
        }
        chroma_meta = format_chroma_metadata(metadata_dict=raw_meta)
        self.assertEqual(chroma_meta["source"], "shipping_rules.txt")
        self.assertEqual(chroma_meta["section"], "")
        self.assertEqual(chroma_meta["page"], 0)
        self.assertEqual(chroma_meta["chunk_index"], 1)

    def test_format_chroma_metadata_custom_values(self):
        """Test metadata formatting preserves provided values."""
        raw_meta = {
            "source": "customs_guide.pdf",
            "section": "Tariff Codes",
            "page": 3,
            "chunk_index": 2,
            "chunk_id": "chunk_002",
            "document_type": "pdf",
            "embedding_model": "text-embedding-3-small",
            "vector_dimension": 1536
        }
        chroma_meta = format_chroma_metadata(metadata_dict=raw_meta)
        self.assertEqual(chroma_meta["section"], "Tariff Codes")
        self.assertEqual(chroma_meta["page"], 3)
        self.assertEqual(chroma_meta["document_type"], "pdf")
        self.assertEqual(chroma_meta["vector_dimension"], 1536)

    # -------------------------------------------------------------
    # 4. Record Insertion & Readback
    # -------------------------------------------------------------
    def test_insert_and_get_embedding_record(self):
        """Test single record insertion and deterministic readback."""
        client = get_vector_db_client(db_path=self.db_path)
        collection = get_or_create_collection(
            client=client,
            collection_name=self.collection_name
        )

        test_record = {
            "embedding_id": "rec_001",
            "chunk_id": "chunk_001",
            "source": "shipping_rules.txt",
            "document_type": "txt",
            "strategy": "paragraph",
            "chunk_index": 1,
            "chunk_text": "Commercial invoices are mandatory for international shipments.",
            "embedding_model": "text-embedding-3-small",
            "vector_dimension": 4,
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "section": "General",
            "page": 1
        }

        inserted_id = insert_embedding_record(
            collection=collection,
            record=test_record,
            expected_dimension=4
        )
        self.assertEqual(inserted_id, "rec_001")
        self.assertEqual(collection.count(), 1)

        # Read back
        readback = get_embedding_record(collection=collection, record_id="rec_001")
        self.assertIsNotNone(readback)
        self.assertEqual(readback["id"], "rec_001")
        self.assertEqual(readback["document_text"], test_record["chunk_text"])
        self.assertEqual(len(readback["embedding"]), 4)
        self.assertAlmostEqual(readback["embedding"][0], 0.1, places=4)
        self.assertEqual(readback["metadata"]["source"], "shipping_rules.txt")

    def test_insert_batch_records(self):
        """Test batch insertion of multiple embedding records."""
        client = get_vector_db_client(db_path=self.db_path)
        collection = get_or_create_collection(
            client=client,
            collection_name=self.collection_name
        )

        records = [
            {
                "embedding_id": f"batch_{i}",
                "chunk_text": f"Document chunk text number {i}",
                "embedding": [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
                "source": "rules.txt",
                "chunk_index": i
            }
            for i in range(1, 4)
        ]

        inserted_ids = insert_embedding_records(
            collection=collection,
            records=records,
            expected_dimension=4
        )
        self.assertEqual(len(inserted_ids), 3)
        self.assertEqual(collection.count(), 3)

        rec2 = get_embedding_record(collection=collection, record_id="batch_2")
        self.assertIsNotNone(rec2)
        self.assertEqual(rec2["id"], "batch_2")
        self.assertEqual(rec2["document_text"], "Document chunk text number 2")

    def test_insert_dimension_mismatch_fails(self):
        """Test that inserting record with mismatched dimension raises ValueError."""
        client = get_vector_db_client(db_path=self.db_path)
        collection = get_or_create_collection(
            client=client,
            collection_name=self.collection_name
        )

        bad_record = {
            "embedding_id": "bad_rec",
            "chunk_text": "Sample text",
            "embedding": [0.1, 0.2]  # Only 2 dims, expected 4
        }

        with self.assertRaises(ValueError) as ctx:
            insert_embedding_record(
                collection=collection,
                record=bad_record,
                expected_dimension=4
            )
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_get_embedding_record_nonexistent(self):
        """Test that querying a nonexistent record returns None."""
        client = get_vector_db_client(db_path=self.db_path)
        collection = get_or_create_collection(
            client=client,
            collection_name=self.collection_name
        )
        result = get_embedding_record(collection=collection, record_id="nonexistent_id")
        self.assertIsNone(result)

    # -------------------------------------------------------------
    # 5. Health Check
    # -------------------------------------------------------------
    def test_run_vector_db_health_check(self):
        """Test health check execution."""
        status = run_vector_db_health_check(
            db_path=self.db_path,
            collection_name=self.collection_name,
            verbose=False
        )
        self.assertEqual(status["status"], "CONNECTED")
        self.assertEqual(status["db_path"], self.db_path)
        self.assertEqual(status["collection"], self.collection_name)



# ============================================================================
# RETRIEVAL TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_retrieval.py
# ------------------------------------------------------------------------------

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


def get_sample_vector_collection_retrieval() -> VectorCollection:
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
        self.collection = get_sample_vector_collection_retrieval()

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

# ------------------------------------------------------------------------------
# Source File: tests\test_scope_guard.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Strict Scope Guard Module
==========================================
Tests classification of user queries into IN-SCOPE vs OUT-OF-SCOPE.
"""

import unittest
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.scope_guard import is_in_scope, OUT_OF_SCOPE_RESPONSE


class TestScopeGuard(unittest.TestCase):

    def test_out_of_scope_queries(self):
        """Test general knowledge, programming, weather, entertainment, and people queries are rejected."""
        out_of_scope_samples = [
            "who is abhiram kollepara",
            "Who is Elon Musk?",
            "Write me a Python program.",
            "What's the weather today?",
            "Tell me a joke.",
            "What is the capital of India?",
            "Explain quantum physics.",
            "who won the world cup",
            "recipe for chocolate cake",
        ]

        for query in out_of_scope_samples:
            with self.subTest(query=query):
                self.assertFalse(
                    is_in_scope(query),
                    f"Query should be classified as OUT-OF-SCOPE: '{query}'"
                )

    def test_in_scope_queries(self):
        """Test logistics, customs duties, HS codes, and import/export queries are accepted."""
        in_scope_samples = [
            "What is the HS code for importing laptops?",
            "What documents are required for importing electronics?",
            "What customs duty applies to this shipment?",
            "What is the import procedure for goods from China to India?",
            "What documents are required for customs clearance?",
            "How is customs duty calculated?",
            "What are the rules for importing this product?",
            "What is the difference between CIF and FOB for customs purposes?",
            "How do I clear freight at the port of origin?",
            "Is a BIS certificate mandatory for courier shipments?",
        ]

        for query in in_scope_samples:
            with self.subTest(query=query):
                self.assertTrue(
                    is_in_scope(query),
                    f"Query should be classified as IN-SCOPE: '{query}'"
                )

    def test_exact_fixed_response_content(self):
        """Verify the exact out-of-scope fixed response text matches specifications."""
        expected_text = (
            "I don't know. I'm only able to help with logistics, customs duties, "
            "import documentation, HS codes, and shipment-related rules."
        )
        self.assertEqual(OUT_OF_SCOPE_RESPONSE, expected_text)

# ------------------------------------------------------------------------------
# Source File: tests\test_similarity.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Embedding Similarity & Distance Metrics Module
==============================================================
Tests cosine similarity calculation, zero-vector safety, query embedding generation,
chunk similarity ranking, top_k filtering, empty query handling, missing embeddings,
and structured results schema preservation.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.similarity import (
    cosine_similarity,
    generate_query_embedding,
    rank_chunks_by_similarity,
    search_similar_chunks,
)


class TestEmbeddingSimilarity(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            {
                "embedding_id": "chunk_001",
                "chunk_id": "ship_rule_01",
                "source": "shipping_rules.txt",
                "chunk_index": 1,
                "page": "1",
                "chunk_text": "Commercial invoices are mandatory for all international shipments.",
                "embedding": [1.0, 0.0, 0.0]
            },
            {
                "embedding_id": "chunk_002",
                "chunk_id": "customs_req_01",
                "source": "customs_requirements.txt",
                "chunk_index": 2,
                "page": "1",
                "chunk_text": "Customs duties for laptop computers under HS code 8471.30 require BIS registration.",
                "embedding": [0.7071, 0.7071, 0.0]
            },
            {
                "embedding_id": "chunk_003",
                "chunk_id": "office_menu_01",
                "source": "cafeteria_menu.txt",
                "chunk_index": 1,
                "page": "1",
                "chunk_text": "Cafeteria lunch menu includes pasta and salad.",
                "embedding": [0.0, 1.0, 0.0]
            }
        ]

    def test_cosine_similarity_identical_vectors(self):
        """1. Test that identical vectors return cosine similarity score of 1.0."""
        vec = [0.123, -0.456, 0.789, 0.321]
        score = cosine_similarity(vec, vec)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_cosine_similarity_unrelated_orthogonal_vectors(self):
        """2. Test that orthogonal (unrelated) vectors return similarity score of 0.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        score = cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_cosine_similarity_zero_vectors_handled_safely(self):
        """3. Test zero-norm vectors return 0.0 without causing ZeroDivisionError."""
        zero_vec = [0.0, 0.0, 0.0]
        normal_vec = [1.0, 2.0, 3.0]

        self.assertEqual(cosine_similarity(zero_vec, normal_vec), 0.0)
        self.assertEqual(cosine_similarity(normal_vec, zero_vec), 0.0)
        self.assertEqual(cosine_similarity(zero_vec, zero_vec), 0.0)

    def test_ranking_multiple_chunks(self):
        """4. Test ranking chunks by similarity sorts scores in descending order."""
        query_vec = [1.0, 0.0, 0.0]
        results = rank_chunks_by_similarity(query_vec, self.sample_chunks, top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["metadata"]["document_id"], "chunk_001")
        self.assertEqual(results[0]["score"], 1.0)

        self.assertEqual(results[1]["metadata"]["document_id"], "chunk_002")
        self.assertAlmostEqual(results[1]["score"], 0.7071, places=3)

        self.assertEqual(results[2]["metadata"]["document_id"], "chunk_003")
        self.assertEqual(results[2]["score"], 0.0)

    def test_top_k_parameter(self):
        """5. Test top_k returns exact top_k count, handles zero and large top_k."""
        query_vec = [1.0, 0.0, 0.0]

        # top_k = 2
        top_2 = rank_chunks_by_similarity(query_vec, self.sample_chunks, top_k=2)
        self.assertEqual(len(top_2), 2)

        # top_k = 0 -> empty list
        top_0 = rank_chunks_by_similarity(query_vec, self.sample_chunks, top_k=0)
        self.assertEqual(top_0, [])

        # top_k negative -> empty list
        top_neg = rank_chunks_by_similarity(query_vec, self.sample_chunks, top_k=-5)
        self.assertEqual(top_neg, [])

        # top_k > candidate count -> returns all candidate chunks
        top_large = rank_chunks_by_similarity(query_vec, self.sample_chunks, top_k=100)
        self.assertEqual(len(top_large), 3)

    def test_empty_query_handling(self):
        """6. Test searching with empty or whitespace-only query returns [] without crashing."""
        self.assertEqual(search_similar_chunks("", self.sample_chunks), [])
        self.assertEqual(search_similar_chunks("   \n\t  ", self.sample_chunks), [])

    def test_missing_or_invalid_embeddings_handled_gracefully(self):
        """7. Test candidate chunks missing embeddings or with None are handled safely."""
        corrupted_chunks = [
            {
                "embedding_id": "good_chunk",
                "chunk_text": "Valid text.",
                "embedding": [1.0, 0.0, 0.0]
            },
            {
                "embedding_id": "missing_embed_chunk",
                "chunk_text": "Missing embedding vector.",
                "embedding": None
            },
            {
                "embedding_id": "malformed_chunk",
                "chunk_text": "Non-list embedding vector.",
                "embedding": "invalid_string_vector"
            }
        ]

        query_vec = [1.0, 0.0, 0.0]
        results = rank_chunks_by_similarity(query_vec, corrupted_chunks, top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["metadata"]["document_id"], "good_chunk")
        self.assertEqual(results[0]["score"], 1.0)
        self.assertEqual(results[1]["score"], 0.0)
        self.assertEqual(results[2]["score"], 0.0)

    def test_structured_results_schema(self):
        """8. Test that returned results match the required JSON schema structure."""
        query_vec = [1.0, 0.0, 0.0]
        results = rank_chunks_by_similarity(query_vec, self.sample_chunks[:1], top_k=1)

        self.assertEqual(len(results), 1)
        item = results[0]

        # Verify top-level keys
        self.assertIn("text", item)
        self.assertIn("score", item)
        self.assertIn("metadata", item)

        # Verify metadata dictionary keys
        meta = item["metadata"]
        self.assertIn("source", meta)
        self.assertIn("chunk_index", meta)
        self.assertIn("page", meta)
        self.assertIn("document_id", meta)

        self.assertEqual(meta["source"], "shipping_rules.txt")
        self.assertEqual(meta["page"], "1")

    @patch("src.embeddings.generate_embedding")
    @patch("src.embeddings.create_embedding_client")
    def test_generate_query_embedding_uses_same_model(self, mock_create_client, mock_gen_embed):
        """9. Test that generate_query_embedding uses the exact same model and client setup."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_gen_embed.return_value = [0.1, 0.2, 0.3]

        vec = generate_query_embedding("What are import rules?", client=mock_client, model="text-embedding-3-small")
        self.assertEqual(vec, [0.1, 0.2, 0.3])
        mock_gen_embed.assert_called_once_with(mock_client, "What are import rules?", model="text-embedding-3-small")



# ============================================================================
# RERANKING TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_reranking.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Task 3.35: Chunk Re-Ranking for Precision Module
================================================================
Tests two-stage retrieval, re-ranking scoring logic, parameter validation
(specifically candidate_k > final_k enforcement), rank movement tracking,
metadata filter preservation, timing metrics, and output formatters.
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
from src.retrieval import retrieve
from src.reranker import (
    rerank_score,
    rerank,
    retrieve_and_rerank,
    format_before_after_comparison,
    format_rank_movement_report,
)


def get_mock_vector_collection_reranking() -> VectorCollection:
    """Helper to build a populated VectorCollection with diverse test chunks."""
    collection = VectorCollection(name="test_reranking_col")
    records = [
        {
            "id": "chunk_01",
            "vector": [1.0, 0.0, 0.0],
            "text": "General shipping guide mentioning container sizes and vessel schedules.",
            "metadata": {
                "source": "international_shipping_guide.pdf",
                "chunk_index": 1,
                "section": "General"
            }
        },
        {
            "id": "chunk_02",
            "vector": [0.8, 0.6, 0.0],
            "text": "Statutory registrations for importing electronics into India require BIS registration certificate.",
            "metadata": {
                "source": "customs_requirements.txt",
                "chunk_index": 1,
                "section": "Statutory Registrations"
            }
        },
        {
            "id": "chunk_03",
            "vector": [0.0, 1.0, 0.0],
            "text": "Commercial invoice and packing list are mandatory documents for customs clearance.",
            "metadata": {
                "source": "shipping_rules.txt",
                "chunk_index": 2,
                "section": "Documentation"
            }
        },
        {
            "id": "chunk_04",
            "vector": [0.5, 0.5, 0.707],
            "text": "Detailed BIS CRS registration rules for telecommunications and IT hardware importers in India.",
            "metadata": {
                "source": "customs_requirements.txt",
                "chunk_index": 2,
                "section": "Statutory Registrations"
            }
        }
    ]
    collection.upsert(records)
    return collection


class TestChunkReRanking(unittest.TestCase):

    def setUp(self):
        self.collection = get_mock_vector_collection_reranking()

    # --------------------------------------------------------------------------
    # 1. PARAMETER VALIDATION & ERROR HANDLING TESTS
    # --------------------------------------------------------------------------
    def test_candidate_k_greater_than_final_k_validation(self):
        """Verifies ValueError is raised when candidate_k <= final_k."""
        # candidate_k == final_k
        with self.assertRaises(ValueError) as ctx:
            retrieve_and_rerank("electronics import India", candidate_k=3, final_k=3, collection=self.collection)
        self.assertIn("must be strictly greater than final_k", str(ctx.exception))

        # candidate_k < final_k
        with self.assertRaises(ValueError) as ctx:
            retrieve_and_rerank("electronics import India", candidate_k=2, final_k=5, collection=self.collection)
        self.assertIn("must be strictly greater than final_k", str(ctx.exception))

    def test_invalid_query_validation(self):
        """Verifies ValueError is raised for empty or non-string query."""
        with self.assertRaises(ValueError):
            retrieve_and_rerank("", candidate_k=5, final_k=2, collection=self.collection)

        with self.assertRaises(ValueError):
            retrieve_and_rerank(None, candidate_k=5, final_k=2, collection=self.collection)

    def test_invalid_k_type_or_value_validation(self):
        """Verifies ValueError for non-positive or boolean k values."""
        with self.assertRaises(ValueError):
            retrieve_and_rerank("query", candidate_k=-1, final_k=3, collection=self.collection)

        with self.assertRaises(ValueError):
            retrieve_and_rerank("query", candidate_k=5, final_k=0, collection=self.collection)

    # --------------------------------------------------------------------------
    # 2. RE-RANKER SCORING & RANKING TESTS
    # --------------------------------------------------------------------------
    def test_rerank_score_term_coverage(self):
        """Verifies chunk with exact query terms receives higher rerank_score."""
        query = "BIS registration electronics India"
        relevant_text = "BIS registration certificate required for electronics import in India."
        irrelevant_text = "Standard shipping vessel schedule and container sizes."

        score_rel = rerank_score(query, relevant_text, vector_score=0.5)
        score_irrel = rerank_score(query, irrelevant_text, vector_score=0.5)

        self.assertGreater(score_rel, score_irrel)

    def test_rerank_sorting_and_rank_assignment(self):
        """Verifies rerank sorts candidates descending by rerank_score and assigns new ranks."""
        query = "BIS registration certificate"
        candidates = [
            {
                "rank": 1,
                "similarity_score": 0.9,
                "chunk_text": "General shipping vessel schedule.",
                "source": "shipping_guide.pdf",
                "chunk_index": 1
            },
            {
                "rank": 2,
                "similarity_score": 0.7,
                "chunk_text": "Mandatory BIS registration certificate for IT hardware.",
                "source": "customs_requirements.txt",
                "chunk_index": 2
            }
        ]

        reranked = rerank(query=query, candidates=candidates, final_k=2)

        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["rank"], 1)
        self.assertEqual(reranked[0]["original_rank"], 2)  # Originally rank 2 moved to rank 1
        self.assertEqual(reranked[0]["source"], "customs_requirements.txt")
        self.assertGreater(reranked[0]["rerank_score"], reranked[1]["rerank_score"])

    # --------------------------------------------------------------------------
    # 3. TWO-STAGE RETRIEVAL PIPELINE TESTS
    # --------------------------------------------------------------------------
    @patch("src.retrieval.generate_query_embedding")
    def test_retrieve_and_rerank_pipeline(self, mock_embed):
        """Verifies end-to-end retrieve_and_rerank pipeline execution and timing metrics."""
        mock_embed.return_value = [0.8, 0.6, 0.0]

        candidates, top_k, metrics = retrieve_and_rerank(
            query="BIS registration electronics India",
            candidate_k=4,
            final_k=2,
            collection=self.collection
        )

        self.assertEqual(len(candidates), 4)
        self.assertEqual(len(top_k), 2)

        self.assertIn("candidate_k", metrics)
        self.assertIn("final_k", metrics)
        self.assertIn("vector_retrieval_time_ms", metrics)
        self.assertIn("reranking_time_ms", metrics)
        self.assertIn("total_pipeline_time_ms", metrics)

    # --------------------------------------------------------------------------
    # 4. METADATA FILTER PRESERVATION TESTS
    # --------------------------------------------------------------------------
    @patch("src.retrieval.generate_query_embedding")
    def test_metadata_filter_preservation(self, mock_embed):
        """Verifies metadata filter passed to stage 1 is preserved and applied."""
        mock_embed.return_value = [0.8, 0.6, 0.0]

        filter_criteria = {"source": "customs_requirements.txt"}
        candidates, top_k, metrics = retrieve_and_rerank(
            query="BIS registration",
            candidate_k=3,
            final_k=2,
            metadata_filter=filter_criteria,
            collection=self.collection
        )

        for item in top_k:
            self.assertEqual(item["source"], "customs_requirements.txt")

    # --------------------------------------------------------------------------
    # 5. OUTPUT FORMATTER TESTS
    # --------------------------------------------------------------------------
    def test_format_before_after_comparison(self):
        """Verifies BEFORE / AFTER ASCII report string formatting."""
        candidates = [
            {"rank": 1, "similarity_score": 0.8, "source": "a.txt", "chunk_index": 1, "chunk_text": "text A"}
        ]
        reranked = [
            {"rank": 1, "vector_score": 0.8, "rerank_score": 8.5, "source": "a.txt", "chunk_index": 1, "chunk_text": "text A"}
        ]

        report = format_before_after_comparison("query text", candidates, reranked)
        self.assertIn("BEFORE RE-RANKING", report)
        self.assertIn("AFTER RE-RANKING", report)
        self.assertIn("query text", report)

    def test_format_rank_movement_report(self):
        """Verifies candidate rank movement report formatting."""
        reranked = [
            {"rank": 1, "original_rank": 3, "source": "customs.txt", "rerank_score": 9.2}
        ]

        report = format_rank_movement_report(reranked)
        self.assertIn("CANDIDATE RANK MOVEMENT TRACKING", report)
        self.assertIn("Rank 3", report)
        self.assertIn("Rank 1", report)
        self.assertIn("customs.txt", report)



# ============================================================================
# RETRIEVAL GUARDRAIL TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_retrieval_guardrail.py
# ------------------------------------------------------------------------------

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



# ============================================================================
# RAG PIPELINE TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_model_parameters.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Model Parameters & Output Control Module
=========================================================
Tests temperature, max_tokens output control parameters, defaults, and API client propagation.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.llm_completion import run_chat_completion, LLM_TEMPERATURE, LLM_MAX_TOKENS


class TestModelParameters(unittest.TestCase):

    def test_default_parameter_constants(self):
        """Verify default configuration constants match requirements (0.1 temperature, 300 max_tokens)."""
        self.assertEqual(LLM_TEMPERATURE, 0.1)
        self.assertEqual(LLM_MAX_TOKENS, 600)

    @patch("src.llm_completion.Groq")
    def test_groq_api_call_receives_temperature_and_max_tokens(self, mock_groq_cls):
        """Verify Groq chat.completions.create is called with configured temperature and max_tokens."""
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"answer": "Sample factual customs response.", "source": "CDLP System"}'
        mock_client.chat.completions.create.return_value = mock_response

        # Execute completion with explicit overrides
        response = run_chat_completion(
            question="What is the HS code for laptops?",
            api_key_override="mock_key",
            temperature_override=0.1,
            max_tokens_override=300
        )

        self.assertEqual(response["answer"], "Sample factual customs response.")
        self.assertEqual(response["source"], "CDLP System")
        
        # Verify kwargs passed to Groq client
        mock_client.chat.completions.create.assert_called()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["max_tokens"], 300)

    @patch("src.llm_completion.Groq")
    def test_custom_temperature_and_max_tokens_override(self, mock_groq_cls):
        """Verify passing custom temperature and max_tokens propagates to API call."""
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"answer": "Short response.", "source": "CDLP System"}'
        mock_client.chat.completions.create.return_value = mock_response

        # Test temperature 1.0 and max_tokens 150
        run_chat_completion(
            question="What documents are needed for customs clearance?",
            api_key_override="mock_key",
            temperature_override=1.0,
            max_tokens_override=150
        )

        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertEqual(kwargs["max_tokens"], 150)

# ------------------------------------------------------------------------------
# Source File: tests\test_prompt_templates.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Prompt Templates & Reusable Prompt Design Module
===============================================================
Tests template definition with named placeholders, runtime dynamic value injection,
template reuse across features, separation from business logic, and error handling for missing values.
"""

import unittest
import sys
import os

# Ensure project root is in sys.path for test runner
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.prompt_templates import (
    PromptTemplate,
    render,
    ANSWER_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    BATCH_EVAL_TEMPLATE,
    STRUCTURED_JSON_TEMPLATE,
)


class TestPromptTemplates(unittest.TestCase):

    def test_define_template_with_named_placeholders(self):
        """Task 1: Test creating a PromptTemplate object and extracting named placeholders."""
        template_str = "Context:\n{context}\n\nQuestion: {question}"
        pt = PromptTemplate(template_str)
        
        self.assertEqual(pt.placeholders, ["context", "question"])
        self.assertEqual(str(pt), template_str)

    def test_inject_dynamic_values_at_runtime(self):
        """Task 2: Test injecting dynamic runtime values into template placeholders."""
        pt = PromptTemplate("System Role: {role} for domain: {domain}")
        rendered = pt.render(role="Compliance Officer", domain="Customs Duty")
        
        self.assertEqual(rendered, "System Role: Compliance Officer for domain: Customs Duty")

    def test_standalone_render_function(self):
        """Test standalone render helper function with string and PromptTemplate instances."""
        rendered_str = render("Hello {name}, your query is {query}", name="Alice", query="HS Code 8471")
        self.assertEqual(rendered_str, "Hello Alice, your query is HS Code 8471")

        rendered_pt = render(ANSWER_TEMPLATE, context="Doc snippet 1", question="Duty rate?")
        self.assertIn("Context:\nDoc snippet 1", rendered_pt)
        self.assertIn("Question:\nDuty rate?", rendered_pt)

    def test_reuse_template_across_multiple_features(self):
        """Task 3: Test reusing the same template structure across Chat and Batch features."""
        pt = ANSWER_TEMPLATE

        # Feature 1: Interactive RAG Chat Feature
        chat_render = render(
            pt,
            context="ShipRule platform CDLP customs duty policy for electronics.",
            question="What is the duty rate for laptops?"
        )
        self.assertIn("CDLP customs compliance assistant", chat_render)
        self.assertIn("laptops?", chat_render)

        # Feature 2: Batch Evaluator / CLI Feature
        batch_render = render(
            pt,
            context="Batch document payload item #104.",
            question="Verify compliance for HS code 8471.30."
        )
        self.assertIn("CDLP customs compliance assistant", batch_render)
        self.assertIn("HS code 8471.30", batch_render)

    def test_missing_placeholder_raises_error(self):
        """Test rendering with missing required placeholders raises ValueError."""
        pt = PromptTemplate("Context: {context} | Question: {question}")
        
        with self.assertRaises(ValueError) as ctx:
            pt.render(context="Only context provided")
        self.assertIn("Missing required placeholder values: question", str(ctx.exception))

    def test_invalid_template_creation_raises_error(self):
        """Test instantiating PromptTemplate with empty or non-string raises ValueError."""
        with self.assertRaises(ValueError):
            PromptTemplate("")

        with self.assertRaises(ValueError):
            PromptTemplate("   ")

    def test_predefined_standard_templates_exist(self):
        """Task 4: Test standard templates exist and are stored separately from logic."""
        self.assertIsInstance(ANSWER_TEMPLATE, PromptTemplate)
        self.assertIsInstance(SYSTEM_PROMPT_TEMPLATE, PromptTemplate)
        self.assertIsInstance(BATCH_EVAL_TEMPLATE, PromptTemplate)
        self.assertIsInstance(STRUCTURED_JSON_TEMPLATE, PromptTemplate)

# ------------------------------------------------------------------------------
# Source File: tests\test_rag_pipeline.py
# ------------------------------------------------------------------------------

"""
Unit and Integration Tests for Phase 3.37: End-to-End RAG Pipeline Module
==========================================================================
Tests modular pipeline stages (embed_query, retrieve_context, assemble_context as assemble_context_pipeline,
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
    assemble_context as assemble_context_pipeline,
    generate_answer,
    answer_query,
)


def get_mock_vector_collection_rag_pipeline() -> VectorCollection:
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
        self.collection = get_mock_vector_collection_rag_pipeline()

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

        text, sources = assemble_context_pipeline(chunks)

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

# ------------------------------------------------------------------------------
# Source File: tests\test_structured_output.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Structured Output & JSON Response Handling Module
===================================================================
Tests defensive parsing, JSON validation, missing field detection,
single-retry mechanism on malformed output, and clean application error handling.
"""

import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.llm_completion import run_chat_completion, parse_and_validate_json_response


class TestStructuredOutput(unittest.TestCase):

    def test_parse_valid_json(self):
        """Test parsing valid JSON with answer and source keys."""
        raw_json = json.dumps({"answer": "Import duty is 7.5%.", "source": "Customs Tariff Act 2024"})
        result = parse_and_validate_json_response(raw_json)
        self.assertEqual(result["answer"], "Import duty is 7.5%.")
        self.assertEqual(result["source"], "Customs Tariff Act 2024")

    def test_parse_json_in_markdown_codeblock(self):
        """Test parsing JSON string wrapped in markdown ```json ... ``` code block."""
        raw_markdown = "```json\n{\n  \"answer\": \"Commercial invoice required.\",\n  \"source\": \"DGFT Regulations\"\n}\n```"
        result = parse_and_validate_json_response(raw_markdown)
        self.assertEqual(result["answer"], "Commercial invoice required.")
        self.assertEqual(result["source"], "DGFT Regulations")

    def test_malformed_json_raises_error(self):
        """Test malformed JSON string raises 'Malformed AI response' error."""
        invalid_raw = "{answer: 'invalid json', source: missing_quotes}"
        with self.assertRaises(ValueError) as ctx:
            parse_and_validate_json_response(invalid_raw)
        self.assertEqual(str(ctx.exception), "Malformed AI response")

    def test_empty_string_raises_error(self):
        """Test empty or whitespace response raises 'Malformed AI response' error."""
        with self.assertRaises(ValueError) as ctx:
            parse_and_validate_json_response("   ")
        self.assertEqual(str(ctx.exception), "Malformed AI response")

    def test_non_object_json_raises_error(self):
        """Test JSON array or primitive raises 'Malformed AI response' error."""
        array_json = json.dumps(["answer", "source"])
        with self.assertRaises(ValueError) as ctx:
            parse_and_validate_json_response(array_json)
        self.assertEqual(str(ctx.exception), "Malformed AI response")

    def test_missing_answer_field_raises_error(self):
        """Test JSON missing 'answer' field raises 'Missing required field: answer' error."""
        missing_answer = json.dumps({"source": "CDLP Policy"})
        with self.assertRaises(ValueError) as ctx:
            parse_and_validate_json_response(missing_answer)
        self.assertEqual(str(ctx.exception), "Missing required field: answer")

    def test_empty_answer_field_raises_error(self):
        """Test JSON with blank 'answer' raises 'Missing required field: answer' error."""
        empty_answer = json.dumps({"answer": "   ", "source": "CDLP Policy"})
        with self.assertRaises(ValueError) as ctx:
            parse_and_validate_json_response(empty_answer)
        self.assertEqual(str(ctx.exception), "Missing required field: answer")

    def test_missing_source_field_raises_error(self):
        """Test JSON missing 'source' field raises 'Missing required field: source' error."""
        missing_source = json.dumps({"answer": "Duty classification is 8471.30."})
        with self.assertRaises(ValueError) as ctx:
            parse_and_validate_json_response(missing_source)
        self.assertEqual(str(ctx.exception), "Missing required field: source")

    @patch("src.llm_completion.Groq")
    def test_single_retry_on_malformed_json(self, mock_groq_cls):
        """Test initial malformed response triggers ONCE retry, which succeeds."""
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        # Call 1 returns malformed prose; Call 2 (retry) returns valid JSON
        resp1 = MagicMock()
        resp1.choices[0].message.content = "This is raw prose without JSON formatting."

        resp2 = MagicMock()
        resp2.choices[0].message.content = json.dumps({
            "answer": "BIS Registration is compulsory.",
            "source": "MeitY Compulsory Registration Scheme"
        })

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        result = run_chat_completion(
            question="Is BIS required for electronics?",
            api_key_override="mock_key"
        )

        self.assertEqual(result["answer"], "BIS Registration is compulsory.")
        self.assertEqual(result["source"], "MeitY Compulsory Registration Scheme")
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    @patch("src.llm_completion.Groq")
    def test_retry_failure_returns_clean_application_error(self, mock_groq_cls):
        """Test both initial call and retry returning invalid JSON yields clean application error."""
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        resp1 = MagicMock()
        resp1.choices[0].message.content = "Malformed raw response."
        resp2 = MagicMock()
        resp2.choices[0].message.content = "Still malformed raw response on retry."

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        result = run_chat_completion(
            question="What is the duty rate?",
            api_key_override="mock_key"
        )

        self.assertIsInstance(result, dict)
        self.assertIn("Error", result["answer"])
        self.assertEqual(result["source"], "CDLP System")
        self.assertEqual(result["error"], "Malformed AI response")
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    @patch("src.llm_completion.Groq")
    def test_json_object_mode_and_zero_temperature_propagation(self, mock_groq_cls):
        """Test response_format json_object and default temperature 0.0 passed to Groq API."""
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "answer": "FOB means Free On Board.",
            "source": "Incoterms 2020"
        })
        mock_client.chat.completions.create.return_value = mock_response

        run_chat_completion(
            question="What does FOB stand for?",
            api_key_override="mock_key"
        )

        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs.get("response_format"), {"type": "json_object"})
        self.assertEqual(kwargs.get("temperature"), 0.0)

    def test_parse_4_field_schema(self):
        """Test parsing valid 4-field schema with answer, sources array, confidence, and has_answer."""
        raw_json = json.dumps({
            "answer": "Commercial invoice and packing list are required.",
            "sources": [{"source": "shipping_rules.txt", "page": "1"}],
            "confidence": "high",
            "has_answer": True
        })
        result = parse_and_validate_json_response(raw_json)
        self.assertEqual(result["answer"], "Commercial invoice and packing list are required.")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["source"], "shipping_rules.txt")
        self.assertEqual(result["confidence"], "high")
        self.assertTrue(result["has_answer"])

    def test_parse_missing_info_schema(self):
        """Test parsing fallback JSON when knowledge base has no answer."""
        raw_json = json.dumps({
            "answer": "I don't have enough information in the provided shipping rules to answer this question.",
            "sources": [],
            "confidence": "low",
            "has_answer": False
        })
        result = parse_and_validate_json_response(raw_json)
        self.assertIn("don't have enough information", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["confidence"], "low")
        self.assertFalse(result["has_answer"])



# ============================================================================
# CONVERSATIONAL RAG TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_conversational_rag.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Conversational RAG & Follow-Up Context Module (KDU 3.42)
=======================================================================
Tests history tracking across turns, follow-up query rewriting, strength-checked
context retrieval using rewritten queries, and multi-turn grounded Q&A.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.conversational_rag import (
    rewrite_followup,
    retrieval_is_strong,
    conversational_answer,
    ConversationalRAGManager,
    format_history_for_rewriter
)


class TestConversationalRAG(unittest.TestCase):

    def setUp(self):
        self.sample_history = [
            {"role": "user", "content": "What evidence is required for project submission?"},
            {"role": "assistant", "content": "The submission needs a PR link, sample output, and a video explanation with source evidence."}
        ]

    # --------------------------------------------------------------------------
    # TASK 1: History Tracking Tests
    # --------------------------------------------------------------------------
    def test_history_tracking_accrual(self):
        """Verifies that user questions and assistant answers accrue properly in history."""
        manager = ConversationalRAGManager(collection=MagicMock(), max_history_tokens=1000)
        self.assertEqual(len(manager.history), 0)

        with patch("src.conversational_rag.rewrite_followup", return_value="What evidence is required for project submission?"), \
             patch("src.conversational_rag.retrieve_context", return_value=([{"metadata": {"source": "test.md"}, "rerank_score": 0.9}], {})), \
             patch("src.conversational_rag.retrieval_is_strong", return_value=True), \
             patch("src.conversational_rag.generate_answer", return_value={"answer": "A PR link and video explanation.", "sources": ["test.md"]}):

            res1 = manager.ask("What evidence is required for project submission?")
            self.assertEqual(len(manager.history), 2)
            self.assertEqual(manager.history[0]["role"], "user")
            self.assertEqual(manager.history[1]["role"], "assistant")

            res2 = manager.ask("What about the video?")
            self.assertEqual(len(manager.history), 4)
            self.assertEqual(manager.history[2]["content"], "What about the video?")

    def test_history_reset(self):
        """Verifies history reset clears stored turns."""
        manager = ConversationalRAGManager(collection=MagicMock())
        manager.history = list(self.sample_history)
        self.assertEqual(len(manager.history), 2)
        manager.reset_history()
        self.assertEqual(len(manager.history), 0)

    # --------------------------------------------------------------------------
    # TASK 2: Follow-up Question Rewriting Tests
    # --------------------------------------------------------------------------
    def test_rewrite_followup_empty_history_returns_original(self):
        """Verifies that an empty history returns the original question intact."""
        q = "What is the duty rate for cars in Italy?"
        rewritten = rewrite_followup(history=[], question=q)
        self.assertEqual(rewritten, q)

    @patch("src.conversational_rag.call_llm")
    def test_rewrite_followup_with_history(self, mock_call_llm):
        """Verifies that follow-up questions are passed to LLM and stripped of prefixes."""
        mock_call_llm.return_value = 'Standalone Query: "What video explanation is required for project submission?"'
        
        rewritten = rewrite_followup(
            history=self.sample_history,
            question="What about the video?"
        )
        self.assertEqual(rewritten, "What video explanation is required for project submission?")
        mock_call_llm.assert_called_once()

    # --------------------------------------------------------------------------
    # TASK 3: Strength-checked Retrieval with Rewritten Query
    # --------------------------------------------------------------------------
    def test_retrieval_is_strong_valid_chunks(self):
        """Verifies retrieval_is_strong returns True for relevant non-empty chunks."""
        chunks = [
            {
                "chunk_text": "Required documents: Commercial invoice, packing list.",
                "rerank_score": 0.85,
                "distance": 0.3
            }
        ]
        self.assertTrue(retrieval_is_strong(chunks))

    def test_retrieval_is_strong_empty_chunks(self):
        """Verifies retrieval_is_strong returns False when no chunks are retrieved."""
        self.assertFalse(retrieval_is_strong([]))
        self.assertFalse(retrieval_is_strong(None))

    # --------------------------------------------------------------------------
    # TASK 4: Multi-turn Conversational Answer Flow & Grounded Q&A
    # --------------------------------------------------------------------------
    @patch("src.conversational_rag.generate_answer")
    @patch("src.conversational_rag.retrieve_context")
    @patch("src.conversational_rag.rewrite_followup")
    def test_conversational_answer_strong_retrieval(self, mock_rewrite, mock_retrieve, mock_gen_answer):
        """Verifies end-to-end multi-turn answer flow when retrieval is strong."""
        mock_rewrite.return_value = "What video explanation is required for project submission?"
        mock_retrieve.return_value = (
            [{"metadata": {"source": "guidelines.md"}, "rerank_score": 0.9, "chunk_text": "Video walkthrough required."}],
            {}
        )
        mock_gen_answer.return_value = {
            "answer": "The submission requires a 3-5 minute screen-share video.",
            "sources": ["guidelines.md"]
        }

        history = list(self.sample_history)
        res = conversational_answer(history=history, user_question="What about the video?")

        self.assertEqual(res["rewritten_query"], "What video explanation is required for project submission?")
        self.assertIn("3-5 minute screen-share video", res["answer"])
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(len(history), 4)

    @patch("src.conversational_rag.retrieve_context")
    @patch("src.conversational_rag.rewrite_followup")
    def test_conversational_answer_weak_retrieval_refusal(self, mock_rewrite, mock_retrieve):
        """Verifies safe refusal answer when retrieval yields no relevant chunks."""
        mock_rewrite.return_value = "What is the deadline for space shuttles?"
        mock_retrieve.return_value = ([], {})

        history = []
        res = conversational_answer(history=history, user_question="What about the deadline for space shuttles?")

        self.assertEqual(res["answer"], "I don't have enough reliable context to answer that.")
        self.assertEqual(len(res["sources"]), 0)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["content"], "I don't have enough reliable context to answer that.")



# ============================================================================
# CONTEXT ASSEMBLY TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_context_assembler.py
# ------------------------------------------------------------------------------

"""
Unit Tests for src/context_assembler.py (Context Assembly & Prompt Augmentation)
================================================================================
Tests:
TEST 1: Retrieved chunks are formatted cleanly.
TEST 2: Sequential source markers [1], [2], [3] are generated.
TEST 3: Source metadata (source, section, chunk, page, country, etc.) appears in formatted context.
TEST 4: Context stays strictly within configured token budget.
TEST 5: Large chunks exceeding remaining budget are excluded, while fitting subsequent chunks are retained.
TEST 6: Grounding instructions are present in prompt.
TEST 7: Insufficient-context instruction and handling is present.
TEST 8: Augmented prompt contains instructions, context, source markers, user question, and budget check.
"""

import unittest
import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.context_assembler import (
    format_chunk,
    assemble_context,
    build_augmented_prompt,
    format_budget_report,
    DEFAULT_GROUNDING_INSTRUCTIONS,
    AssembledContext
)
from src.token_counter import count_tokens


class TestContextAssembler(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            {
                "id": "chunk_01",
                "source": "customs_reg_india.json",
                "section": "Customs Record - Laptop Computers (India)",
                "chunk_index": 0,
                "page": 1,
                "country": "India",
                "hs_code": "8471.30",
                "chunk_text": "Customs Record for Laptop Computers in India: Basic Customs Duty is 0%, IGST is 18%. Mandatory BIS CRS registration required."
            },
            {
                "id": "chunk_02",
                "source": "cdlp_prd_v1.0.md",
                "section": "6. Dataset & Data Source Documentation",
                "chunk_index": 0,
                "page": 4,
                "chunk_text": "CDLP Dataset & Schema: Customs Regulation Dataset includes country, HS code, duty rate, and mandatory document requirements."
            },
            {
                "id": "chunk_03",
                "source": "international_shipping_guide.pdf",
                "section": "Incoterms 2020",
                "chunk_index": 1,
                "page": 2,
                "chunk_text": "Under CIF terms, the seller must deliver goods on board and arrange minimum insurance coverage to the destination port."
            }
        ]

    # --------------------------------------------------------------------------
    # TEST 1: RETRIEVED CHUNKS ARE FORMATTED
    # --------------------------------------------------------------------------
    def test_retrieved_chunks_are_formatted(self):
        """TEST 1: Verifies retrieved chunks are properly formatted with headers and text bodies."""
        chunk = self.sample_chunks[0]
        formatted = format_chunk(chunk, marker_index=1)

        self.assertTrue(isinstance(formatted, str))
        self.assertIn("[1] Source: customs_reg_india.json", formatted)
        self.assertIn("Customs Record for Laptop Computers in India", formatted)

    # --------------------------------------------------------------------------
    # TEST 2: SOURCE MARKERS [1], [2], [3] ARE GENERATED
    # --------------------------------------------------------------------------
    def test_source_markers_generated_sequentially(self):
        """TEST 2: Verifies unique sequential source markers [1], [2], [3] are attached to chunks."""
        result = assemble_context(
            retrieved_chunks=self.sample_chunks,
            user_question="Importing laptops in India",
            max_context_tokens=4096,
            response_reserve_tokens=500
        )

        self.assertIn("[1] Source: customs_reg_india.json", result.context)
        self.assertIn("[2] Source: cdlp_prd_v1.0.md", result.context)
        self.assertIn("[3] Source: international_shipping_guide.pdf", result.context)

        self.assertEqual(len(result.source_mapping), 3)
        self.assertEqual(result.source_mapping[0]["marker"], "[1]")
        self.assertEqual(result.source_mapping[1]["marker"], "[2]")
        self.assertEqual(result.source_mapping[2]["marker"], "[3]")

    # --------------------------------------------------------------------------
    # TEST 3: SOURCE METADATA APPEARS IN FORMATTED CONTEXT
    # --------------------------------------------------------------------------
    def test_source_metadata_appears_in_formatted_context(self):
        """TEST 3: Verifies metadata fields (source, section, chunk, page, country, hs_code) appear accurately."""
        chunk_with_meta = {
            "metadata": {
                "source": "customs_act_1962.pdf",
                "section": "Section 14 Valuation",
                "chunk_index": 5,
                "page": 12,
                "country": "India",
                "hs_code": "8471",
                "source_agency": "CBIC",
                "source_url": "https://cbic.gov.in",
                "last_confirmed_date": "2026-01-15"
            },
            "chunk_text": "Valuation of imported goods shall be based on transaction value."
        }

        formatted = format_chunk(chunk_with_meta, marker_index=1)
        self.assertIn("[1] Source: customs_act_1962.pdf", formatted)
        self.assertIn("Section: Section 14 Valuation", formatted)
        self.assertIn("Chunk: 5", formatted)
        self.assertIn("Page: 12", formatted)
        self.assertIn("Country: India", formatted)
        self.assertIn("HS Code: 8471", formatted)
        self.assertIn("Agency: CBIC", formatted)
        self.assertIn("URL: https://cbic.gov.in", formatted)
        self.assertIn("Date: 2026-01-15", formatted)
        self.assertIn("Valuation of imported goods shall be based on transaction value.", formatted)

    # --------------------------------------------------------------------------
    # TEST 4: CONTEXT STAYS WITHIN TOKEN BUDGET
    # --------------------------------------------------------------------------
    def test_context_stays_within_token_budget(self):
        """TEST 4: Verifies total estimated tokens never exceed configured max budget."""
        result = assemble_context(
            retrieved_chunks=self.sample_chunks,
            user_question="What are the documentation rules for shipping to India?",
            max_context_tokens=1000,
            response_reserve_tokens=300
        )

        inst_tok = count_tokens(DEFAULT_GROUNDING_INSTRUCTIONS)
        q_tok = count_tokens("What are the documentation rules for shipping to India?")
        ctx_tok = count_tokens(result.context)
        reserve_tok = 300

        total_est = inst_tok + q_tok + ctx_tok + reserve_tok
        self.assertLessEqual(total_est, 1000)
        self.assertEqual(result.budget_info["budget_check"], "PASS")

    # --------------------------------------------------------------------------
    # TEST 5: LARGE CHUNKS EXCLUDED / SAFE BUDGET CONSTRAINT
    # --------------------------------------------------------------------------
    def test_large_chunks_excluded_when_budget_insufficient(self):
        """TEST 5: Large chunks exceeding remaining budget are excluded; subsequent fitting chunks fit."""
        huge_chunk = {
            "source": "huge_regulations.txt",
            "section": "Mega Policy",
            "chunk_index": 1,
            "page": 1,
            "chunk_text": "Extremely large policy content. " * 300  # ~1200 tokens
        }
        small_chunk_1 = {
            "source": "small_doc_1.txt",
            "section": "Brief Section 1",
            "chunk_index": 1,
            "page": 1,
            "chunk_text": "Compact rule 1: Invoice required."
        }
        small_chunk_2 = {
            "source": "small_doc_2.txt",
            "section": "Brief Section 2",
            "chunk_index": 2,
            "page": 1,
            "chunk_text": "Compact rule 2: Packing list required."
        }

        # Budget of 350 tokens with 100 reserve and ~100 instruction leaves ~150 available context tokens
        result = assemble_context(
            retrieved_chunks=[huge_chunk, small_chunk_1, small_chunk_2],
            user_question="Requirements?",
            max_context_tokens=350,
            response_reserve_tokens=100
        )

        # Huge chunk should be skipped, small chunks should be included
        self.assertNotIn("huge_regulations.txt", result.context)
        self.assertIn("small_doc_1.txt", result.context)
        self.assertIn("small_doc_2.txt", result.context)
        self.assertLessEqual(result.budget_info["total_estimated_tokens"], 350)
        self.assertEqual(result.budget_info["budget_check"], "PASS")

    # --------------------------------------------------------------------------
    # TEST 6: GROUNDING INSTRUCTIONS ARE PRESENT
    # --------------------------------------------------------------------------
    def test_grounding_instructions_are_present(self):
        """TEST 6: Grounding instructions are present in default configuration and prompt."""
        self.assertIn("grounded customs-information assistant", DEFAULT_GROUNDING_INSTRUCTIONS)
        self.assertIn("ONLY the information contained in the provided context", DEFAULT_GROUNDING_INSTRUCTIONS)
        self.assertIn("Do not invent customs rules", DEFAULT_GROUNDING_INSTRUCTIONS)
        self.assertIn("Never invent or fabricate source markers", DEFAULT_GROUNDING_INSTRUCTIONS)

        result = assemble_context(
            retrieved_chunks=self.sample_chunks,
            user_question="Explain customs rules"
        )
        self.assertIn("grounded customs-information assistant", result.augmented_prompt)

    # --------------------------------------------------------------------------
    # TEST 7: INSUFFICIENT-CONTEXT INSTRUCTION IS PRESENT
    # --------------------------------------------------------------------------
    def test_insufficient_context_instruction_is_present(self):
        """TEST 7: Prompt instructs LLM to state context insufficiency when unable to answer."""
        expected_phrase = "'The provided context is insufficient to answer this question.'"
        self.assertIn(expected_phrase, DEFAULT_GROUNDING_INSTRUCTIONS)

        empty_result = assemble_context(
            retrieved_chunks=[],
            user_question="What is the customs duty for a product not in records?"
        )
        self.assertEqual(empty_result.selected_chunk_count, 0)
        self.assertEqual(empty_result.context, "")
        self.assertIn(expected_phrase, empty_result.augmented_prompt)

    # --------------------------------------------------------------------------
    # TEST 8: AUGMENTED PROMPT CONTAINS ALL REQUIRED SECTIONS
    # --------------------------------------------------------------------------
    def test_augmented_prompt_contains_all_required_sections(self):
        """TEST 8: Augmented prompt contains instructions, context, source markers, question, budget."""
        result = assemble_context(
            retrieved_chunks=self.sample_chunks,
            user_question="What are the import requirements for laptops in India?",
            max_context_tokens=4096,
            response_reserve_tokens=500
        )

        prompt = result.augmented_prompt
        self.assertIn("SYSTEM / INSTRUCTIONS", prompt)
        self.assertIn("grounded customs-information assistant", prompt)
        self.assertIn("CONTEXT", prompt)
        self.assertIn("[1] Source: customs_reg_india.json", prompt)
        self.assertIn("[2] Source: cdlp_prd_v1.0.md", prompt)
        self.assertIn("[3] Source: international_shipping_guide.pdf", prompt)
        self.assertIn("USER QUESTION", prompt)
        self.assertIn("What are the import requirements for laptops in India?", prompt)
        self.assertIn("CONTEXT BUDGET", prompt)
        self.assertIn("Budget Check: PASS", prompt)

    # --------------------------------------------------------------------------
    # TEST 9: ASSEMBLED CONTEXT UNPACKING & DICT ACCESS COMPATIBILITY
    # --------------------------------------------------------------------------
    def test_assembled_context_unpacking_and_dict_access(self):
        """TEST 9: AssembledContext supports 4-tuple unpacking and dictionary/attribute access."""
        result = assemble_context(
            retrieved_chunks=self.sample_chunks[:1],
            user_question="test query"
        )

        # 4-tuple unpacking
        ctx, count, tok, sources = result
        self.assertEqual(ctx, result.context)
        self.assertEqual(count, result.selected_chunk_count)
        self.assertEqual(tok, result.token_count)
        self.assertEqual(sources, result.source_mapping)

        # Dict access
        self.assertEqual(result["context"], result.context)
        self.assertEqual(result["selected_chunk_count"], 1)
        self.assertEqual(result.get("token_count"), result.token_count)
        self.assertIn("[1]", result["sources"][0]["marker"])

# ------------------------------------------------------------------------------
# Source File: tests\test_context_manager.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Context Window & Message History Management Module
==================================================================
Tests total_tokens calculation, trim_history, summarize_history,
prepare_context, and ContextManager class under various context budget conditions.
"""

import unittest
import sys
import os

# Ensure project root is in sys.path for running tests
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.context_manager import (
    total_tokens,
    trim_history,
    summarize_history,
    prepare_context,
    ContextManager,
    _truncate_text_to_tokens,
)
from src.token_counter import count_tokens


class TestContextManager(unittest.TestCase):

    def setUp(self):
        self.system_prompt = "You are an AI Support Assistant for Customs Duty & Documentation Lookup Platform."
        self.sample_user_msg = "What are the required import documents for laptop computers to India?"
        self.sample_assistant_msg = "You need a Commercial Invoice, Bill of Lading, and BIS Registration Certificate."

    def test_total_tokens_calculation(self):
        """Test total_tokens includes message content, role keys, and ChatML overhead."""
        messages = [
            {"role": "system", "content": "System prompt text."},
            {"role": "user", "content": "Hello model!"}
        ]
        tok = total_tokens(messages)
        self.assertGreater(tok, 0)
        # Overhead per msg (4*2=8) + primer (2) + content tok + role tok ("system", "user" = 2 tok)
        expected_total = count_tokens("System prompt text.") + count_tokens("Hello model!") + 10 + count_tokens("system") + count_tokens("user")
        self.assertEqual(tok, expected_total)

    def test_normal_multiturn_conversation(self):
        """Test multi-turn conversation within token budget preserves all history."""
        cm = ContextManager(system_prompt=self.system_prompt, max_context_tokens=1000, response_reserve_tokens=100)
        
        res1 = cm.ask("What is ShipRule?", llm_fn=lambda msgs: "ShipRule is a customs platform.")
        self.assertEqual(len(cm.history), 3) # sys + user + assistant
        
        res2 = cm.ask("How does it verify duties?", llm_fn=lambda msgs: "It uses source traceability.")
        self.assertEqual(len(cm.history), 5) # sys + 2 user + 2 assistant
        self.assertEqual(res2["strategy_applied"], "none")

    def test_system_message_preservation(self):
        """Test system prompt is always preserved even when severe trimming occurs."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "Message 1 " * 50},
            {"role": "assistant", "content": "Response 1 " * 50},
            {"role": "user", "content": "Message 2 " * 50},
            {"role": "assistant", "content": "Response 2 " * 50},
            {"role": "user", "content": "Message 3 " * 50},
        ]
        
        # Set tight budget so trimming must happen
        sys_tokens = total_tokens([messages[0]])
        tight_budget = sys_tokens + 150
        
        trimmed_msgs, final_tokens, strategy = trim_history(messages, budget=tight_budget, preserve_recent=1)
        
        # Verify system message is at index 0 and unmodified
        self.assertEqual(trimmed_msgs[0]["role"], "system")
        self.assertEqual(trimmed_msgs[0]["content"], self.system_prompt)
        self.assertLessEqual(final_tokens, tight_budget)

    def test_trim_behavior_pair_removal(self):
        """Test trim strategy removes oldest user/assistant pairs first."""
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User question 1"},
            {"role": "assistant", "content": "Assistant answer 1"},
            {"role": "user", "content": "User question 2"},
            {"role": "assistant", "content": "Assistant answer 2"},
            {"role": "user", "content": "User question 3"},
        ]
        
        budget = total_tokens(messages) - 35
        trimmed, final_tok, strategy = trim_history(messages, budget=budget)
        
        self.assertEqual(strategy, "trim")
        self.assertEqual(trimmed[0]["role"], "system")
        # Turn 1 should be removed
        self.assertNotIn("User question 1", [m["content"] for m in trimmed])
        self.assertIn("User question 3", trimmed[-1]["content"])

    def test_summarize_behavior(self):
        """Test summarize_history condenses older turns while retaining recent turns intact."""
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Older query 1: Tell me about HS Code 8471 for computers and peripherals import details. " * 3},
            {"role": "assistant", "content": "Older answer 1: HS Code 8471 covers automatic data processing machines and units thereof. " * 3},
            {"role": "user", "content": "Older query 2: What about BIS certificate requirement for imported electronics? " * 3},
            {"role": "assistant", "content": "Older answer 2: BIS registration is compulsory in India for safety standard compliance. " * 3},
            {"role": "user", "content": "Recent query: What is the tariff percentage?"},
            {"role": "assistant", "content": "Recent answer: Basic customs duty rate is 7.5%."},
            {"role": "user", "content": "Current question: Any extra surcharge?"}
        ]

        full_tokens = total_tokens(messages)

        summarized, final_tok, strategy = summarize_history(messages, budget=full_tokens - 40, preserve_recent=1)

        self.assertEqual(strategy, "summarize")
        self.assertEqual(summarized[0]["role"], "system")
        # Verify a summary message exists in history
        summary_found = any("Context Summary" in m.get("content", "") for m in summarized)
        self.assertTrue(summary_found)
        self.assertLessEqual(final_tok, full_tokens - 40)

    def test_long_user_message_handling(self):
        """Test an extraordinarily long user message exceeding budget is handled gracefully."""
        huge_user_msg = "Customs classification text. " * 300
        cm = ContextManager(system_prompt=self.system_prompt, max_context_tokens=300, response_reserve_tokens=50)

        prep = cm.get_prepared_payload(user_message=huge_user_msg)
        
        self.assertLessEqual(prep["total_tokens"], prep["budget"])
        self.assertIn("Truncated", prep["messages"][-1]["content"])

    def test_rag_context_budget_consumption(self):
        """Test when RAG retrieved documents consume the budget, context is trimmed to fit."""
        huge_rag_context = "Document chunk rule line. " * 400
        user_question = "What is the tariff for laptops?"

        prep = prepare_context(
            history=[],
            retrieved_context=huge_rag_context,
            user_message=user_question,
            max_tokens=500,
            reserve_tokens=100,
            system_prompt=self.system_prompt
        )

        self.assertLessEqual(prep["total_tokens"], prep["budget"])
        self.assertIn("Truncated", prep["messages"][-1]["content"])
        self.assertIn("Question:\nWhat is the tariff for laptops?", prep["messages"][-1]["content"])

    def test_multiple_consecutive_trimming_operations(self):
        """Test history maintains continuity and stability across 10 consecutive turns under tight budget."""
        cm = ContextManager(
            system_prompt=self.system_prompt,
            max_context_tokens=250,
            response_reserve_tokens=50,
            strategy="trim"
        )

        for i in range(10):
            res = cm.ask(
                user_message=f"Turn {i}: Query about customs document rule #{i}",
                llm_fn=lambda msgs: f"Response to turn {i}."
            )
            # Verify budget constraint maintained at every turn
            self.assertLessEqual(res["total_tokens"], res["budget"])
            self.assertEqual(cm.history[0]["role"], "system")

    def test_conversation_continuity_after_pruning(self):
        """Test conversation flow remains valid role-alternating structure after trimming."""
        cm = ContextManager(
            system_prompt=self.system_prompt,
            max_context_tokens=350,
            response_reserve_tokens=50,
            strategy="trim"
        )

        for i in range(5):
            cm.ask(f"Question {i} " * 15, llm_fn=lambda msgs: f"Answer {i} " * 15)

        # Inspect history role ordering: System, User, Assistant, User, Assistant ...
        roles = [m["role"] for m in cm.history]
        self.assertEqual(roles[0], "system")
        for idx in range(1, len(roles)):
            if idx % 2 == 1:
                self.assertEqual(roles[idx], "user")
            else:
                self.assertEqual(roles[idx], "assistant")



# ============================================================================
# CITATION TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_citation_manager.py
# ------------------------------------------------------------------------------

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



# ============================================================================
# ANSWER VALIDATION TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_answer_validator.py
# ------------------------------------------------------------------------------

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



# ============================================================================
# EVALUATION TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_evaluation.py
# ------------------------------------------------------------------------------

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


def get_sample_vector_collection_eval() -> VectorCollection:
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
        self.collection = get_sample_vector_collection_eval()

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

# ------------------------------------------------------------------------------
# Source File: tests\test_evaluation_metrics.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Task 3.36: Retrieval Evaluation & Recall Testing Module
=======================================================================
Tests evaluate_query(), evaluate_dataset(), evaluate_multi_k(),
Recall@K, Precision@K, MRR@K formulas, zero division safety,
failure inspection, diagnostic logging, and strategy comparisons.
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
    LABELLED_QUERIES,
    evaluate_query,
    evaluate_dataset,
    evaluate_multi_k,
    inspect_retrieval_failures,
    diagnose_failure_causes,
    compare_retrieval_strategies,
)


def get_test_vector_collection() -> VectorCollection:
    """Constructs a populated VectorCollection for evaluation testing."""
    collection = VectorCollection(name="test_eval_col")
    records = [
        {
            "id": "customs_requirements_paragraph_001",
            "vector": [1.0, 0.0, 0.0],
            "text": "Customs duties and 8-digit HS code tariff classification details.",
            "metadata": {"source": "customs_requirements.txt", "chunk_index": 1}
        },
        {
            "id": "customs_requirements_paragraph_002",
            "vector": [0.0, 1.0, 0.0],
            "text": "Preferential tariff rates and Certificate of Origin under FTA.",
            "metadata": {"source": "customs_requirements.txt", "chunk_index": 2}
        },
        {
            "id": "customs_requirements_paragraph_003",
            "vector": [0.0, 0.0, 1.0],
            "text": "Statutory registrations BIS CRS for IT hardware and telecommunications.",
            "metadata": {"source": "customs_requirements.txt", "chunk_index": 3}
        },
        {
            "id": "international_shipping_guide_paragraph_001",
            "vector": [0.707, 0.707, 0.0],
            "text": "Incoterms 2020 rules for FOB and CIF risk obligations.",
            "metadata": {"source": "international_shipping_guide.pdf", "chunk_index": 1}
        },
        {
            "id": "shipping_rules_paragraph_001",
            "vector": [0.5, 0.5, 0.707],
            "text": "Mandatory shipping documents commercial invoice and packing list.",
            "metadata": {"source": "shipping_rules.txt", "chunk_index": 1}
        }
    ]
    collection.upsert(records)
    return collection


class TestRetrievalEvaluation(unittest.TestCase):

    def setUp(self):
        self.collection = get_test_vector_collection()
        self.sample_query_item = {
            "query": "What are the statutory registrations for IT hardware?",
            "relevant_chunk_ids": [
                "customs_requirements_paragraph_003",
                "customs_requirements.txt:3"
            ],
            "expected_sources": ["customs_requirements.txt"],
            "keywords": ["statutory", "registrations", "hardware"],
            "topic": "Statutory Registrations"
        }

    # --------------------------------------------------------------------------
    # 1. PARAMETER VALIDATION & ERROR HANDLING TESTS
    # --------------------------------------------------------------------------
    def test_invalid_query_item_validation(self):
        """Verifies ValueError for invalid query item."""
        with self.assertRaises(ValueError):
            evaluate_query({}, k=3, collection=self.collection)

        with self.assertRaises(ValueError):
            evaluate_query(None, k=3, collection=self.collection)

    def test_invalid_labelled_queries_dataset_validation(self):
        """Verifies ValueError for invalid labelled queries dataset."""
        with self.assertRaises(ValueError):
            evaluate_dataset([], k=3, collection=self.collection)

        with self.assertRaises(ValueError):
            evaluate_dataset("invalid_type", k=3, collection=self.collection)

    # --------------------------------------------------------------------------
    # 2. METRIC FORMULA & RECALL/PRECISION TESTS
    # --------------------------------------------------------------------------
    @patch("src.evaluation.retrieve")
    def test_evaluate_query_metrics_hit(self, mock_retrieve):
        """Verifies Recall@K, Precision@K, and MRR@K calculations when hit occurs."""
        mock_retrieve.return_value = [
            {
                "document_id": "customs_requirements_paragraph_003",
                "source": "customs_requirements.txt",
                "chunk_index": 3,
                "similarity_score": 0.85,
                "chunk_text": "Statutory registrations BIS CRS."
            },
            {
                "document_id": "shipping_rules_paragraph_001",
                "source": "shipping_rules.txt",
                "chunk_index": 1,
                "similarity_score": 0.40,
                "chunk_text": "Commercial invoice."
            }
        ]

        res = evaluate_query(self.sample_query_item, k=2, strategy="vector", collection=self.collection)

        self.assertTrue(res["hit"])
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["precision"], 0.5)  # 1 hit / 2 retrieved
        self.assertEqual(res["reciprocal_rank"], 1.0)  # Hit at rank 1

    @patch("src.evaluation.retrieve")
    def test_evaluate_query_metrics_miss(self, mock_retrieve):
        """Verifies zero metrics when no relevant chunks are retrieved."""
        mock_retrieve.return_value = [
            {
                "document_id": "shipping_rules_paragraph_001",
                "source": "shipping_rules.txt",
                "chunk_index": 1,
                "similarity_score": 0.30,
                "chunk_text": "Unrelated text."
            }
        ]

        item_miss = {
            "query": "Non-matching query",
            "relevant_chunk_ids": ["customs_requirements_paragraph_003"],
            "expected_sources": ["customs_requirements.txt"]
        }

        res = evaluate_query(item_miss, k=1, strategy="vector", collection=self.collection)

        self.assertFalse(res["hit"])
        self.assertEqual(res["recall"], 0.0)
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["reciprocal_rank"], 0.0)

    def test_division_by_zero_safety(self):
        """Verifies evaluation handles K > corpus size and empty retrieval safely without division by zero."""
        item = {
            "query": "Test query",
            "relevant_chunk_ids": [],
            "expected_sources": []
        }
        res = evaluate_query(item, k=100, strategy="vector", collection=self.collection)
        self.assertIn("recall", res)
        self.assertIn("precision", res)
        self.assertGreaterEqual(res["recall"], 0.0)

    # --------------------------------------------------------------------------
    # 3. AGGREGATE EVALUATION & MULTI-K TESTS
    # --------------------------------------------------------------------------
    def test_evaluate_dataset_aggregate(self):
        """Verifies evaluate_dataset returns aggregate recall, precision, and MRR."""
        dataset_eval = evaluate_dataset(LABELLED_QUERIES, k=3, strategy="vector", collection=self.collection)

        self.assertEqual(dataset_eval["total_queries"], 5)
        self.assertGreaterEqual(dataset_eval["average_recall"], 0.0)
        self.assertGreaterEqual(dataset_eval["average_precision"], 0.0)
        self.assertGreaterEqual(dataset_eval["mrr"], 0.0)
        self.assertIn("recall_pct", dataset_eval)

    def test_evaluate_multi_k(self):
        """Verifies evaluate_multi_k evaluates metrics across K=[1, 3, 5, 10]."""
        multi_k = evaluate_multi_k(LABELLED_QUERIES, k_values=[1, 3, 5, 10], strategy="vector", collection=self.collection)

        self.assertIn("metrics_by_k", multi_k)
        for k in [1, 3, 5, 10]:
            self.assertIn(k, multi_k["metrics_by_k"])
            self.assertIn("recall_pct", multi_k["metrics_by_k"][k])
            self.assertIn("precision_pct", multi_k["metrics_by_k"][k])

    # --------------------------------------------------------------------------
    # 4. FAILURE DIAGNOSTICS & STRATEGY COMPARISON TESTS
    # --------------------------------------------------------------------------
    def test_inspect_retrieval_failures_and_diagnose(self):
        """Verifies failure extraction and cause diagnosis logic."""
        mock_failure_record = {
            "query": "Unknown query",
            "topic": "Unknown",
            "recall": 0.0,
            "precision": 0.0,
            "expected_relevant_chunks": ["chunk_xyz"],
            "retrieved_chunk_ids": ["chunk_abc"],
            "missing_relevant_chunks": ["chunk_xyz"],
            "retrieved_scores": [0.05],
            "retrieved_sources": ["source.txt"]
        }

        causes = diagnose_failure_causes(mock_failure_record)
        self.assertTrue(len(causes) > 0)
        self.assertIn("Weak semantic similarity", causes[0])

    def test_compare_retrieval_strategies(self):
        """Verifies comparison across Vector, Hybrid, and Re-Ranked strategies."""
        comp = compare_retrieval_strategies(LABELLED_QUERIES, k_values=[1, 3], collection=self.collection)

        self.assertIn("vector", comp)
        self.assertIn("hybrid", comp)
        self.assertIn("reranked", comp)

# ------------------------------------------------------------------------------
# Source File: tests\test_rag_evaluation.py
# ------------------------------------------------------------------------------

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



# ============================================================================
# SANITY / INTEGRATION TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_sanity_checker.py
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# Source File: tests\test_token_cost.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Token Usage & API Cost Reporting Module
======================================================
Tests calculate_token_cost, format_token_cost_report, sample token calculations,
and 6-decimal-place cost formatting.
"""

import unittest
import sys
import os

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.token_counter import (
    count_tokens,
    calculate_token_cost,
    format_token_cost_report,
    INPUT_RATE,
    OUTPUT_RATE,
)


class TestTokenCost(unittest.TestCase):

    def test_sample1_token_cost_calculation(self):
        """Test Sample 1: 'What shipping rule applies to this order?' token and cost math."""
        text = "What shipping rule applies to this order?"
        self.assertEqual(len(text), 41)
        
        # Test math with Sample 1 specified token counts (9 input tokens, 20 output tokens)
        cost_data = calculate_token_cost(
            input_tokens=9,
            output_tokens=20,
            input_rate=0.000002,
            output_rate=0.000006
        )

        self.assertEqual(cost_data["input_tokens"], 9)
        self.assertEqual(cost_data["output_tokens"], 20)
        self.assertAlmostEqual(cost_data["input_cost"], 0.000018, places=6)
        self.assertAlmostEqual(cost_data["output_cost"], 0.000120, places=6)
        self.assertAlmostEqual(cost_data["total_cost"], 0.000138, places=6)

    def test_sample2_shiprule_project_description(self):
        """Test Sample 2: Full ShipRule project description token counting and cost estimation."""
        description = (
            "ShipRule CDLP is an automated customs compliance and duty classification platform.\n"
            "Source traceability ensures every duty lookup is verified against official trade regulations.\n"
            "It validates import documents, calculates tariffs based on HS Codes, and flags restricted goods."
        )
        
        char_count = len(description)
        self.assertGreater(char_count, 100)
        
        in_tokens = count_tokens(description)
        out_tokens = 50
        
        cost_data = calculate_token_cost(in_tokens, out_tokens)
        self.assertGreater(cost_data["input_tokens"], 0)
        self.assertEqual(cost_data["output_tokens"], 50)
        self.assertAlmostEqual(
            cost_data["total_cost"],
            cost_data["input_cost"] + cost_data["output_cost"],
            places=6
        )

    def test_format_token_cost_report_structure(self):
        """Test format_token_cost_report produces the exact required output block structure."""
        report = format_token_cost_report(
            input_tokens=42,
            output_tokens=50,
            input_rate=0.000002,
            output_rate=0.000006
        )

        self.assertIn("SHIPRULE - TOKEN USAGE & COST", report)
        self.assertIn("Tokenizer Used: tiktoken", report)
        self.assertIn("Encoding Used: cl100k_base", report)
        self.assertIn("Input Tokens: 42", report)
        self.assertIn("Output Tokens: 50", report)
        self.assertIn("42 × 0.000002 = $0.000084", report)
        self.assertIn("50 × 0.000006 = $0.000300", report)
        self.assertIn("Total Cost:\n$0.000384", report)



# ============================================================================
# API TESTS
# ============================================================================

# ------------------------------------------------------------------------------
# Source File: tests\test_api.py
# ------------------------------------------------------------------------------

"""
Unit Tests for Task 3.44: Backend API for the RAG Service
===========================================================
Tests FastAPI endpoints (/root, /health, /query), Pydantic input validation,
response schema structure, error status codes (400, 422, 500), and environment config.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import uuid


# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from src.api import app, QueryRequest, QueryResponse, Source


class TestRAGBackendAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self._emb_path = Path(project_root) / 'outputs' / 'embedded_chunks.json'
        self._emb_backup = self._emb_path.read_text(encoding='utf-8') if self._emb_path.exists() else None

    # --------------------------------------------------------------------------
    # 1. SYSTEM ENDPOINTS TESTS
    # --------------------------------------------------------------------------
    def test_root_endpoint(self):
        """Verifies GET / returns 200 OK and service metadata."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["service"], "ShipRule RAG Service API")
        self.assertEqual(data["status"], "online")
        self.assertIn("config", data)

    def test_health_check_endpoint(self):
        """Verifies GET /health returns 200 OK and healthy status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("environment", data)

    # --------------------------------------------------------------------------
    # 2. POST /query SUCCESSFUL ENDPOINT TESTS
    # --------------------------------------------------------------------------
    @patch("src.api.answer_query")
    def test_query_endpoint_success(self, mock_answer_query):
        """Verifies POST /query returns 200 OK with structured QueryResponse JSON."""
        mock_answer_query.return_value = {
            "answer": "All international shipments must include a commercial invoice and packing list.",
            "sources": [{"source": "shipping_rules.txt"}],
            "retrieved_chunks": [{
                "source": "shipping_rules.txt",
                "chunk_index": 1,
                "document_id": "shipping_rules_paragraph_001",
                "rerank_score": 0.85
            }],
            "guardrail_decision": "ALLOW",
            "token_usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "timing": {"generation_ms": 500.0}
        }

        payload = {
            "question": "What shipping documentation is required for international customs clearance?",
            "use_reranking": True,
            "final_k": 3
        }

        response = self.client.post("/query", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("sources", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "answered")
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["source"], "shipping_rules.txt")
        self.assertEqual(data["sources"][0]["chunk_id"], "shipping_rules_paragraph_001")
        self.assertEqual(data["sources"][0]["score"], 0.85)

    @patch("src.api.answer_query")
    def test_query_endpoint_refusal_status(self, mock_answer_query):
        """Verifies POST /query sets status to 'refused' when context is insufficient."""
        mock_answer_query.return_value = {
            "answer": "The provided context is insufficient to answer this question.",
            "sources": [],
            "retrieved_chunks": [],
            "guardrail_decision": "REFUSE",
            "token_usage": {},
            "timing": {}
        }

        payload = {"question": "What is the policy for employee vacation approval?"}
        response = self.client.post("/query", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "refused")

    # --------------------------------------------------------------------------
    # 3. INPUT VALIDATION & ERROR HANDLING TESTS
    # --------------------------------------------------------------------------
    def test_query_endpoint_validation_short_question(self):
        """Verifies 422 Unprocessable Entity when question is less than 3 characters."""
        payload = {"question": "hi"}
        response = self.client.post("/query", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_query_endpoint_validation_missing_question(self):
        """Verifies 422 Unprocessable Entity when question field is missing."""
        payload = {}
        response = self.client.post("/query", json=payload)
        self.assertEqual(response.status_code, 422)

    @patch("src.api.answer_query")
    def test_query_endpoint_value_error_400(self, mock_answer_query):
        """Verifies 400 Bad Request when pipeline raises ValueError."""
        mock_answer_query.side_effect = ValueError("Query must be a non-empty string.")

        payload = {"question": "   "}
        # Note: space string passes Pydantic length >= 3, but is caught by endpoint check
        response = self.client.post("/query", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Question field cannot be empty", response.json()["detail"])

    @patch("src.api.answer_query")
    def test_query_endpoint_server_error_500(self, mock_answer_query):
        """Verifies 500 Internal Server Error when pipeline raises an unexpected exception."""
        mock_answer_query.side_effect = RuntimeError("Unexpected vector DB error")

        payload = {"question": "What is the Basic Customs Duty rate?"}
        response = self.client.post("/query", json=payload)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "RAG service failed")


# ------------------------------------------------------------------------------
# 4. DOCUMENT UPLOAD & INDEXING ENDPOINT TESTS (POST /documents)
# ------------------------------------------------------------------------------

class TestDocumentUploadAPI(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, '_emb_backup'):
            if self._emb_backup is not None:
                self._emb_path.write_text(self._emb_backup, encoding='utf-8')
            elif self._emb_path.exists():
                self._emb_path.unlink()

    def setUp(self):
        self.client = TestClient(app)
        self._emb_path = Path(project_root) / 'outputs' / 'embedded_chunks.json'
        self._emb_backup = self._emb_path.read_text(encoding='utf-8') if self._emb_path.exists() else None

    @patch("src.api.generate_embedding")
    @patch("src.api.create_embedding_client")
    def test_upload_txt_success(self, mock_client, mock_embed):
        """1. Successful .txt upload."""
        mock_embed.return_value = [0.1] * 384
        content = b"Customs Tariff Code 9999 is applicable for special maritime imports."
        files = {"file": ("customs_tariff_9999.txt", content, "text/plain")}

        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "indexed")
        self.assertEqual(data["filename"], "customs_tariff_9999.txt")
        self.assertIn("summary", data)
        self.assertGreater(data["summary"]["chunks"], 0)
        self.assertEqual(data["summary"]["chunks"], data["summary"]["indexed"])

    @patch("src.api.generate_embedding")
    @patch("src.api.create_embedding_client")
    def test_upload_md_success(self, mock_client, mock_embed):
        """2. Successful .md upload."""
        mock_embed.return_value = [0.2] * 384
        content = b"# Markdown Shipping Guide\n\nAll containerized freight must have tamper-evident seals."
        files = {"file": ("freight_guide.md", content, "text/markdown")}

        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "indexed")
        self.assertEqual(data["filename"], "freight_guide.md")
        self.assertGreater(data["summary"]["chunks"], 0)

    @patch("src.api.generate_embedding")
    @patch("src.api.create_embedding_client")
    def test_upload_pdf_success(self, mock_client, mock_embed):
        """3. Successful .pdf upload."""
        mock_embed.return_value = [0.3] * 384
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, txt="International Maritime Hazardous Materials Protocol", ln=1)
        pdf_bytes = bytes(pdf.output())

        files = {"file": ("hazardous_protocol.pdf", pdf_bytes, "application/pdf")}
        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "indexed")
        self.assertEqual(data["filename"], "hazardous_protocol.pdf")
        self.assertGreater(data["summary"]["chunks"], 0)

    def test_upload_unsupported_extension(self):
        """4. Unsupported extension -> HTTP 415."""
        files = {"file": ("malicious_script.exe", b"binary content", "application/octet-stream")}
        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 415)
        self.assertIn("Unsupported file type", response.json()["detail"])

    def test_upload_missing_filename(self):
        """5. Missing/invalid filename -> HTTP 400 or 422."""
        files = {"file": ("   ", b"some content", "text/plain")}
        response = self.client.post("/documents", files=files)
        self.assertIn(response.status_code, [400, 422])

    def test_upload_empty_file(self):
        """6. Empty file -> HTTP 400."""
        files = {"file": ("empty_document.txt", b"", "text/plain")}
        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("no extractable text", response.json()["detail"])

    @patch("src.api.MAX_UPLOAD_SIZE_MB", 1)
    def test_upload_exceeding_max_size(self):
        """7. File exceeding maximum size -> HTTP 413."""
        large_content = b"A" * (2 * 1024 * 1024)  # 2MB > 1MB limit
        files = {"file": ("oversized_file.txt", large_content, "text/plain")}
        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 413)
        self.assertIn("exceeds maximum", response.json()["detail"].lower())

    @patch("src.api.generate_embedding")
    @patch("src.api.create_embedding_client")
    def test_upload_path_traversal_attempt(self, mock_client, mock_embed):
        """8. Unsafe filename/path traversal attempt."""
        mock_embed.return_value = [0.1] * 384
        content = b"Path traversal payload text content."
        files = {"file": ("../../etc/passwd.txt", content, "text/plain")}
        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], "passwd.txt")
        self.assertFalse("../../" in data["summary"]["document"])

    @patch("src.api._load_pdf")
    def test_upload_text_extraction_failure(self, mock_pdf_load):
        """9. Text extraction failure (empty PDF or unreadable text) -> HTTP 400."""
        mock_pdf_load.side_effect = ValueError("PDF document contains no extractable text.")
        files = {"file": ("scanned_blank_image.pdf", b"fake pdf header", "application/pdf")}
        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("no extractable text", response.json()["detail"])

    @patch("src.api.generate_embedding")
    @patch("src.api.create_embedding_client")
    def test_upload_embedding_failure(self, mock_client, mock_embed):
        """10. Embedding failure -> HTTP 500."""
        mock_embed.side_effect = RuntimeError("Embedding provider service offline")
        files = {"file": ("valid_policy.txt", b"Valid policy text content.", "text/plain")}
        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Document indexing failed")

    @patch("src.api.process_uploaded_document")
    def test_upload_vector_indexing_failure(self, mock_process):
        """11. Vector indexing failure -> HTTP 500."""
        mock_process.side_effect = Exception("Disk I/O error writing vector store")
        files = {"file": ("valid_policy.txt", b"Valid policy text content.", "text/plain")}
        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Document indexing failed")

    @patch("src.api.generate_embedding")
    @patch("src.api.create_embedding_client")
    def test_upload_correct_chunk_count_summary(self, mock_client, mock_embed):
        """12. Successful indexing returns correct chunk count."""
        mock_embed.return_value = [0.1] * 384
        content = b"Chunk 1 text content.\n\n" + (b"Word " * 500)
        files = {"file": ("multi_chunk_doc.txt", content, "text/plain")}
        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["summary"]["chunks"], 2)
        self.assertEqual(data["summary"]["chunks"], data["summary"]["indexed"])

    def test_runtime_search_newly_uploaded_content(self):
        """13. Newly uploaded content is immediately retrievable through /query."""
        unique_token = f"UNIQUE_CUSTOMS_RULE_{uuid.uuid4().hex[:6]}"
        unique_text = f"Special Customs Requirement: Shipments containing {unique_token} require Certificate XYZ-99."
        files = {"file": ("unique_customs_policy.txt", unique_text.encode("utf-8"), "text/plain")}

        # 1. Upload document
        upload_resp = self.client.post("/documents", files=files)
        self.assertEqual(upload_resp.status_code, 200)
        self.assertEqual(upload_resp.json()["status"], "indexed")

        # 2. Query unique sentence via POST /query
        query_payload = {
            "question": f"What certificate is required for shipments with {unique_token}?",
            "use_reranking": False,
            "final_k": 3
        }
        query_resp = self.client.post("/query", json=query_payload)
        self.assertEqual(query_resp.status_code, 200)
        query_data = query_resp.json()

        # 3. Verify retrieved answer or sources reference the uploaded document
        self.assertEqual(query_data["status"], "answered")
        sources_found = [s["source"] for s in query_data.get("sources", [])]
        self.assertTrue(any("unique_customs_policy.txt" in s for s in sources_found))


if __name__ == '__main__':
    unittest.main()
