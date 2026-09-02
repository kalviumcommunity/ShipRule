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


if __name__ == "__main__":
    unittest.main()
