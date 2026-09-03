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


if __name__ == "__main__":
    unittest.main()
