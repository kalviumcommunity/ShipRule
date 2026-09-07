"""
Unit Tests for src/context_assembler.py (Context Assembly & Prompt Augmentation)
================================================================================
Tests:
TEST 1: Retrieved chunks are formatted cleanly.
TEST 2: Sequential source markers [1], [2], [3] are generated.
TEST 3: Source metadata (source, section, chunk, page, country, etc.) appears in formatted context.
TEST 4: Context stays strictly within configured token budget.
TEST 5: Large chunks exceeding remaining budget are excluded, while fitting subsequent chunks are retained.
TEST 6: Grounding instructions are present in prompt.
TEST 7: Insufficient-context instruction and handling is present.
TEST 8: Augmented prompt contains instructions, context, source markers, user question, and budget check.
"""

import unittest
import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.context_assembler import (
    format_chunk,
    assemble_context,
    build_augmented_prompt,
    format_budget_report,
    DEFAULT_GROUNDING_INSTRUCTIONS,
    AssembledContext
)
from src.token_counter import count_tokens


class TestContextAssembler(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            {
                "id": "chunk_01",
                "source": "customs_reg_india.json",
                "section": "Customs Record - Laptop Computers (India)",
                "chunk_index": 0,
                "page": 1,
                "country": "India",
                "hs_code": "8471.30",
                "chunk_text": "Customs Record for Laptop Computers in India: Basic Customs Duty is 0%, IGST is 18%. Mandatory BIS CRS registration required."
            },
            {
                "id": "chunk_02",
                "source": "cdlp_prd_v1.0.md",
                "section": "6. Dataset & Data Source Documentation",
                "chunk_index": 0,
                "page": 4,
                "chunk_text": "CDLP Dataset & Schema: Customs Regulation Dataset includes country, HS code, duty rate, and mandatory document requirements."
            },
            {
                "id": "chunk_03",
                "source": "international_shipping_guide.pdf",
                "section": "Incoterms 2020",
                "chunk_index": 1,
                "page": 2,
                "chunk_text": "Under CIF terms, the seller must deliver goods on board and arrange minimum insurance coverage to the destination port."
            }
        ]

    # --------------------------------------------------------------------------
    # TEST 1: RETRIEVED CHUNKS ARE FORMATTED
    # --------------------------------------------------------------------------
    def test_retrieved_chunks_are_formatted(self):
        """TEST 1: Verifies retrieved chunks are properly formatted with headers and text bodies."""
        chunk = self.sample_chunks[0]
        formatted = format_chunk(chunk, marker_index=1)

        self.assertTrue(isinstance(formatted, str))
        self.assertIn("[1] Source: customs_reg_india.json", formatted)
        self.assertIn("Customs Record for Laptop Computers in India", formatted)

    # --------------------------------------------------------------------------
    # TEST 2: SOURCE MARKERS [1], [2], [3] ARE GENERATED
    # --------------------------------------------------------------------------
    def test_source_markers_generated_sequentially(self):
        """TEST 2: Verifies unique sequential source markers [1], [2], [3] are attached to chunks."""
        result = assemble_context(
            retrieved_chunks=self.sample_chunks,
            user_question="Importing laptops in India",
            max_context_tokens=4096,
            response_reserve_tokens=500
        )

        self.assertIn("[1] Source: customs_reg_india.json", result.context)
        self.assertIn("[2] Source: cdlp_prd_v1.0.md", result.context)
        self.assertIn("[3] Source: international_shipping_guide.pdf", result.context)

        self.assertEqual(len(result.source_mapping), 3)
        self.assertEqual(result.source_mapping[0]["marker"], "[1]")
        self.assertEqual(result.source_mapping[1]["marker"], "[2]")
        self.assertEqual(result.source_mapping[2]["marker"], "[3]")

    # --------------------------------------------------------------------------
    # TEST 3: SOURCE METADATA APPEARS IN FORMATTED CONTEXT
    # --------------------------------------------------------------------------
    def test_source_metadata_appears_in_formatted_context(self):
        """TEST 3: Verifies metadata fields (source, section, chunk, page, country, hs_code) appear accurately."""
        chunk_with_meta = {
            "metadata": {
                "source": "customs_act_1962.pdf",
                "section": "Section 14 Valuation",
                "chunk_index": 5,
                "page": 12,
                "country": "India",
                "hs_code": "8471",
                "source_agency": "CBIC",
                "source_url": "https://cbic.gov.in",
                "last_confirmed_date": "2026-01-15"
            },
            "chunk_text": "Valuation of imported goods shall be based on transaction value."
        }

        formatted = format_chunk(chunk_with_meta, marker_index=1)
        self.assertIn("[1] Source: customs_act_1962.pdf", formatted)
        self.assertIn("Section: Section 14 Valuation", formatted)
        self.assertIn("Chunk: 5", formatted)
        self.assertIn("Page: 12", formatted)
        self.assertIn("Country: India", formatted)
        self.assertIn("HS Code: 8471", formatted)
        self.assertIn("Agency: CBIC", formatted)
        self.assertIn("URL: https://cbic.gov.in", formatted)
        self.assertIn("Date: 2026-01-15", formatted)
        self.assertIn("Valuation of imported goods shall be based on transaction value.", formatted)

    # --------------------------------------------------------------------------
    # TEST 4: CONTEXT STAYS WITHIN TOKEN BUDGET
    # --------------------------------------------------------------------------
    def test_context_stays_within_token_budget(self):
        """TEST 4: Verifies total estimated tokens never exceed configured max budget."""
        result = assemble_context(
            retrieved_chunks=self.sample_chunks,
            user_question="What are the documentation rules for shipping to India?",
            max_context_tokens=1000,
            response_reserve_tokens=300
        )

        inst_tok = count_tokens(DEFAULT_GROUNDING_INSTRUCTIONS)
        q_tok = count_tokens("What are the documentation rules for shipping to India?")
        ctx_tok = count_tokens(result.context)
        reserve_tok = 300

        total_est = inst_tok + q_tok + ctx_tok + reserve_tok
        self.assertLessEqual(total_est, 1000)
        self.assertEqual(result.budget_info["budget_check"], "PASS")

    # --------------------------------------------------------------------------
    # TEST 5: LARGE CHUNKS EXCLUDED / SAFE BUDGET CONSTRAINT
    # --------------------------------------------------------------------------
    def test_large_chunks_excluded_when_budget_insufficient(self):
        """TEST 5: Large chunks exceeding remaining budget are excluded; subsequent fitting chunks fit."""
        huge_chunk = {
            "source": "huge_regulations.txt",
            "section": "Mega Policy",
            "chunk_index": 1,
            "page": 1,
            "chunk_text": "Extremely large policy content. " * 300  # ~1200 tokens
        }
        small_chunk_1 = {
            "source": "small_doc_1.txt",
            "section": "Brief Section 1",
            "chunk_index": 1,
            "page": 1,
            "chunk_text": "Compact rule 1: Invoice required."
        }
        small_chunk_2 = {
            "source": "small_doc_2.txt",
            "section": "Brief Section 2",
            "chunk_index": 2,
            "page": 1,
            "chunk_text": "Compact rule 2: Packing list required."
        }

        # Budget of 350 tokens with 100 reserve and ~100 instruction leaves ~150 available context tokens
        result = assemble_context(
            retrieved_chunks=[huge_chunk, small_chunk_1, small_chunk_2],
            user_question="Requirements?",
            max_context_tokens=350,
            response_reserve_tokens=100
        )

        # Huge chunk should be skipped, small chunks should be included
        self.assertNotIn("huge_regulations.txt", result.context)
        self.assertIn("small_doc_1.txt", result.context)
        self.assertIn("small_doc_2.txt", result.context)
        self.assertLessEqual(result.budget_info["total_estimated_tokens"], 350)
        self.assertEqual(result.budget_info["budget_check"], "PASS")

    # --------------------------------------------------------------------------
    # TEST 6: GROUNDING INSTRUCTIONS ARE PRESENT
    # --------------------------------------------------------------------------
    def test_grounding_instructions_are_present(self):
        """TEST 6: Grounding instructions are present in default configuration and prompt."""
        self.assertIn("grounded customs-information assistant", DEFAULT_GROUNDING_INSTRUCTIONS)
        self.assertIn("ONLY the information contained in the provided context", DEFAULT_GROUNDING_INSTRUCTIONS)
        self.assertIn("Do not invent customs rules", DEFAULT_GROUNDING_INSTRUCTIONS)
        self.assertIn("Never invent or fabricate source markers", DEFAULT_GROUNDING_INSTRUCTIONS)

        result = assemble_context(
            retrieved_chunks=self.sample_chunks,
            user_question="Explain customs rules"
        )
        self.assertIn("grounded customs-information assistant", result.augmented_prompt)

    # --------------------------------------------------------------------------
    # TEST 7: INSUFFICIENT-CONTEXT INSTRUCTION IS PRESENT
    # --------------------------------------------------------------------------
    def test_insufficient_context_instruction_is_present(self):
        """TEST 7: Prompt instructs LLM to state context insufficiency when unable to answer."""
        expected_phrase = "'The provided context is insufficient to answer this question.'"
        self.assertIn(expected_phrase, DEFAULT_GROUNDING_INSTRUCTIONS)

        empty_result = assemble_context(
            retrieved_chunks=[],
            user_question="What is the customs duty for a product not in records?"
        )
        self.assertEqual(empty_result.selected_chunk_count, 0)
        self.assertEqual(empty_result.context, "")
        self.assertIn(expected_phrase, empty_result.augmented_prompt)

    # --------------------------------------------------------------------------
    # TEST 8: AUGMENTED PROMPT CONTAINS ALL REQUIRED SECTIONS
    # --------------------------------------------------------------------------
    def test_augmented_prompt_contains_all_required_sections(self):
        """TEST 8: Augmented prompt contains instructions, context, source markers, question, budget."""
        result = assemble_context(
            retrieved_chunks=self.sample_chunks,
            user_question="What are the import requirements for laptops in India?",
            max_context_tokens=4096,
            response_reserve_tokens=500
        )

        prompt = result.augmented_prompt
        self.assertIn("SYSTEM / INSTRUCTIONS", prompt)
        self.assertIn("grounded customs-information assistant", prompt)
        self.assertIn("CONTEXT", prompt)
        self.assertIn("[1] Source: customs_reg_india.json", prompt)
        self.assertIn("[2] Source: cdlp_prd_v1.0.md", prompt)
        self.assertIn("[3] Source: international_shipping_guide.pdf", prompt)
        self.assertIn("USER QUESTION", prompt)
        self.assertIn("What are the import requirements for laptops in India?", prompt)
        self.assertIn("CONTEXT BUDGET", prompt)
        self.assertIn("Budget Check: PASS", prompt)

    # --------------------------------------------------------------------------
    # TEST 9: ASSEMBLED CONTEXT UNPACKING & DICT ACCESS COMPATIBILITY
    # --------------------------------------------------------------------------
    def test_assembled_context_unpacking_and_dict_access(self):
        """TEST 9: AssembledContext supports 4-tuple unpacking and dictionary/attribute access."""
        result = assemble_context(
            retrieved_chunks=self.sample_chunks[:1],
            user_question="test query"
        )

        # 4-tuple unpacking
        ctx, count, tok, sources = result
        self.assertEqual(ctx, result.context)
        self.assertEqual(count, result.selected_chunk_count)
        self.assertEqual(tok, result.token_count)
        self.assertEqual(sources, result.source_mapping)

        # Dict access
        self.assertEqual(result["context"], result.context)
        self.assertEqual(result["selected_chunk_count"], 1)
        self.assertEqual(result.get("token_count"), result.token_count)
        self.assertIn("[1]", result["sources"][0]["marker"])


if __name__ == "__main__":
    unittest.main()
