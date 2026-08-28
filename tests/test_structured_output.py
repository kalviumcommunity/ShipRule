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


if __name__ == "__main__":
    unittest.main()
