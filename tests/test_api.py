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


if __name__ == "__main__":
    unittest.main()
