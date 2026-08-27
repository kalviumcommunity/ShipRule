"""
Unit Tests for Token Usage & API Cost Reporting Module
======================================================
Tests calculate_token_cost, format_token_cost_report, sample token calculations,
and 6-decimal-place cost formatting.
"""

import unittest
import sys
import os

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.token_counter import (
    count_tokens,
    calculate_token_cost,
    format_token_cost_report,
    INPUT_RATE,
    OUTPUT_RATE,
)


class TestTokenCost(unittest.TestCase):

    def test_sample1_token_cost_calculation(self):
        """Test Sample 1: 'What shipping rule applies to this order?' token and cost math."""
        text = "What shipping rule applies to this order?"
        self.assertEqual(len(text), 41)
        
        # Test math with Sample 1 specified token counts (9 input tokens, 20 output tokens)
        cost_data = calculate_token_cost(
            input_tokens=9,
            output_tokens=20,
            input_rate=0.000002,
            output_rate=0.000006
        )

        self.assertEqual(cost_data["input_tokens"], 9)
        self.assertEqual(cost_data["output_tokens"], 20)
        self.assertAlmostEqual(cost_data["input_cost"], 0.000018, places=6)
        self.assertAlmostEqual(cost_data["output_cost"], 0.000120, places=6)
        self.assertAlmostEqual(cost_data["total_cost"], 0.000138, places=6)

    def test_sample2_shiprule_project_description(self):
        """Test Sample 2: Full ShipRule project description token counting and cost estimation."""
        description = (
            "ShipRule CDLP is an automated customs compliance and duty classification platform.\n"
            "Source traceability ensures every duty lookup is verified against official trade regulations.\n"
            "It validates import documents, calculates tariffs based on HS Codes, and flags restricted goods."
        )
        
        char_count = len(description)
        self.assertGreater(char_count, 100)
        
        in_tokens = count_tokens(description)
        out_tokens = 50
        
        cost_data = calculate_token_cost(in_tokens, out_tokens)
        self.assertGreater(cost_data["input_tokens"], 0)
        self.assertEqual(cost_data["output_tokens"], 50)
        self.assertAlmostEqual(
            cost_data["total_cost"],
            cost_data["input_cost"] + cost_data["output_cost"],
            places=6
        )

    def test_format_token_cost_report_structure(self):
        """Test format_token_cost_report produces the exact required output block structure."""
        report = format_token_cost_report(
            input_tokens=42,
            output_tokens=50,
            input_rate=0.000002,
            output_rate=0.000006
        )

        self.assertIn("SHIPRULE - TOKEN USAGE & COST", report)
        self.assertIn("Tokenizer Used: tiktoken", report)
        self.assertIn("Encoding Used: cl100k_base", report)
        self.assertIn("Input Tokens: 42", report)
        self.assertIn("Output Tokens: 50", report)
        self.assertIn("42 × 0.000002 = $0.000084", report)
        self.assertIn("50 × 0.000006 = $0.000300", report)
        self.assertIn("Total Cost:\n$0.000384", report)


if __name__ == "__main__":
    unittest.main()
