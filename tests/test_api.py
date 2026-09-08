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

    def setUp(self):
        self.client = TestClient(app)

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


if __name__ == "__main__":
    unittest.main()

