"""
Unit Tests for Strict Scope Guard Module
==========================================
Tests classification of user queries into IN-SCOPE vs OUT-OF-SCOPE.
"""

import unittest
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.scope_guard import is_in_scope, OUT_OF_SCOPE_RESPONSE


class TestScopeGuard(unittest.TestCase):

    def test_out_of_scope_queries(self):
        """Test general knowledge, programming, weather, entertainment, and people queries are rejected."""
        out_of_scope_samples = [
            "who is abhiram kollepara",
            "Who is Elon Musk?",
            "Write me a Python program.",
            "What's the weather today?",
            "Tell me a joke.",
            "What is the capital of India?",
            "Explain quantum physics.",
            "who won the world cup",
            "recipe for chocolate cake",
        ]

        for query in out_of_scope_samples:
            with self.subTest(query=query):
                self.assertFalse(
                    is_in_scope(query),
                    f"Query should be classified as OUT-OF-SCOPE: '{query}'"
                )

    def test_in_scope_queries(self):
        """Test logistics, customs duties, HS codes, and import/export queries are accepted."""
        in_scope_samples = [
            "What is the HS code for importing laptops?",
            "What documents are required for importing electronics?",
            "What customs duty applies to this shipment?",
            "What is the import procedure for goods from China to India?",
            "What documents are required for customs clearance?",
            "How is customs duty calculated?",
            "What are the rules for importing this product?",
            "What is the difference between CIF and FOB for customs purposes?",
            "How do I clear freight at the port of origin?",
            "Is a BIS certificate mandatory for courier shipments?",
        ]

        for query in in_scope_samples:
            with self.subTest(query=query):
                self.assertTrue(
                    is_in_scope(query),
                    f"Query should be classified as IN-SCOPE: '{query}'"
                )

    def test_exact_fixed_response_content(self):
        """Verify the exact out-of-scope fixed response text matches specifications."""
        expected_text = (
            "I don't know. I'm only able to help with logistics, customs duties, "
            "import documentation, HS codes, and shipment-related rules."
        )
        self.assertEqual(OUT_OF_SCOPE_RESPONSE, expected_text)


if __name__ == "__main__":
    unittest.main()
