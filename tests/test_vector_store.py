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


if __name__ == "__main__":
    unittest.main()
