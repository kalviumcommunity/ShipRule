"""
Unit Tests for Text Extraction & Cleaning Pipeline Module
===========================================================
Tests Unicode NFKC normalization, line ending normalization (\r\n to \n),
page footer pattern removal, whitespace collapsing, consecutive newline reduction,
and preservation of meaningful content (headings, numbers, code blocks, tables).
"""

import unittest
import sys
import os

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.text_cleaner import clean, format_cleaning_summary


class TestTextCleaner(unittest.TestCase):

    def test_unicode_normalization_nfkc(self):
        """Test Unicode normalization converts fullwidth/compat characters using NFKC."""
        raw_text = "Fullwidth numbers \uff11\uff12\uff13 and ligature \ufb01le."
        cleaned = clean(raw_text)
        self.assertEqual(cleaned, "Fullwidth numbers 123 and ligature file.")

    def test_windows_line_ending_normalization(self):
        """Test normalizing \\r\\n and \\r to standard \\n."""
        raw_text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        cleaned = clean(raw_text)
        self.assertEqual(cleaned, "Line 1\nLine 2\nLine 3\nLine 4")

    def test_page_footer_removal(self):
        """Test removal of page footer patterns like 'Page 3 of 12' and 'Page 10 of 25'."""
        raw_text = (
            "Customs documentation guidelines.\n"
            "Page 3 of 12\n"
            "All shipments require invoice.\n"
            "Page 10 of 25\n"
            "End of section."
        )
        cleaned = clean(raw_text)
        self.assertNotIn("Page 3 of 12", cleaned)
        self.assertNotIn("Page 10 of 25", cleaned)
        self.assertIn("Customs documentation guidelines.", cleaned)
        self.assertIn("All shipments require invoice.", cleaned)
        self.assertIn("End of section.", cleaned)

    def test_case_insensitive_page_footer_removal(self):
        """Test footer pattern removal handles varied casing."""
        raw_text = "Content before.\npage 1 of 5\npage 100 OF 200\nContent after."
        cleaned = clean(raw_text)
        self.assertNotIn("page 1 of 5", cleaned)
        self.assertNotIn("page 100 OF 200", cleaned)
        self.assertIn("Content before.", cleaned)
        self.assertIn("Content after.", cleaned)

    def test_collapse_repeated_spaces_and_tabs(self):
        """Test collapsing repeated spaces and tabs into a single space while keeping newlines."""
        raw_text = "Word1    Word2\t\tWord3   \t   Word4\nLine 2   Word5"
        cleaned = clean(raw_text)
        self.assertEqual(cleaned, "Word1 Word2 Word3 Word4\nLine 2 Word5")

    def test_collapse_consecutive_newlines(self):
        """Test collapsing 3 or more consecutive newlines into a maximum of 2."""
        raw_text = "Header\n\n\n\n\nParagraph 1\n\n\nParagraph 2"
        cleaned = clean(raw_text)
        self.assertEqual(cleaned, "Header\n\nParagraph 1\n\nParagraph 2")

    def test_strip_leading_trailing_whitespace(self):
        """Test stripping leading and trailing whitespace."""
        raw_text = "   \n\n   Clean text inside.   \n\n  "
        cleaned = clean(raw_text)
        self.assertEqual(cleaned, "Clean text inside.")

    def test_preserve_headings_and_punctuation(self):
        """Test preservation of markdown headings and punctuation."""
        raw_text = (
            "# Main Title: Customs Duty & Tariff Rules\n\n"
            "## Section 1.2: Compliance Verification!\n"
            "Is origin certificate required? Yes (under FTA rules)."
        )
        cleaned = clean(raw_text)
        self.assertIn("# Main Title: Customs Duty & Tariff Rules", cleaned)
        self.assertIn("## Section 1.2: Compliance Verification!", cleaned)
        self.assertIn("Is origin certificate required? Yes (under FTA rules).", cleaned)

    def test_preserve_numbers_and_tariff_codes(self):
        """Test preservation of numbers, HS codes, and monetary values."""
        raw_text = "Harmonized Tariff Code: 8471.30.10. Total declared value: $12,500.50 (Duty rate: 7.5%)."
        cleaned = clean(raw_text)
        self.assertEqual(
            cleaned,
            "Harmonized Tariff Code: 8471.30.10. Total declared value: $12,500.50 (Duty rate: 7.5%)."
        )

    def test_preserve_code_blocks(self):
        """Test preservation of code blocks and structural syntax."""
        raw_text = (
            "```python\n"
            "def calculate_duty(value, rate):\n"
            "    return value * rate\n"
            "```"
        )
        cleaned = clean(raw_text)
        self.assertIn("```python", cleaned)
        self.assertIn("def calculate_duty(value, rate):", cleaned)
        self.assertIn("return value * rate", cleaned)
        self.assertIn("```", cleaned)

    def test_preserve_table_content(self):
        """Test preservation of table headers, borders, and cell values."""
        raw_text = (
            "| HS Code | Commodity Description | Duty Rate |\n"
            "| ------- | --------------------- | --------- |\n"
            "| 8471.30 | Laptops & Notebooks   | 0%        |\n"
            "| 8517.12 | Smartphones           | 20%       |"
        )
        cleaned = clean(raw_text)
        self.assertIn("| HS Code | Commodity Description | Duty Rate |", cleaned)
        self.assertIn("| 8471.30 | Laptops & Notebooks | 0% |", cleaned)
        self.assertIn("| 8517.12 | Smartphones | 20% |", cleaned)

    def test_no_overcleaning_of_meaningful_text(self):
        """Ensure sentences mentioning numbers or 'Page' in normal prose are not erased."""
        raw_text = "Refer to Section 3 of the agreement. See Page 5 for details on shipping rates."
        cleaned = clean(raw_text)
        # "Section 3" and "See Page 5 for details" should NOT be removed as page footers
        self.assertIn("Refer to Section 3 of the agreement.", cleaned)
        self.assertIn("See Page 5 for details on shipping rates.", cleaned)

    def test_edge_cases_empty_and_none(self):
        """Test edge cases with empty inputs, None, or pure whitespace."""
        self.assertEqual(clean(""), "")
        self.assertEqual(clean(None), "")
        self.assertEqual(clean("   \t \n\r\n "), "")

    def test_format_cleaning_summary(self):
        """Test summary output formatting helper."""
        raw = "Line 1\r\nPage 1 of 5\r\nLine 2"
        cleaned = clean(raw)
        summary = format_cleaning_summary("document.pdf", raw, cleaned)
        self.assertIn("document.pdf: 27 -> 14 chars", summary)
        self.assertIn("BEFORE:", summary)
        self.assertIn("AFTER :", summary)


if __name__ == "__main__":
    unittest.main()
