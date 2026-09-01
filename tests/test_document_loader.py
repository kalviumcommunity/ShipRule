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



if __name__ == "__main__":
    unittest.main()
