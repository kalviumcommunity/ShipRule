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


if __name__ == "__main__":
    unittest.main()
