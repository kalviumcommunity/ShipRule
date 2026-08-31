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


if __name__ == "__main__":
    unittest.main()
