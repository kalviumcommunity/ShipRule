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


if __name__ == "__main__":
    unittest.main()
