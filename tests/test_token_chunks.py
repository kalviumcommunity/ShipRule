"""
Unit Tests for Concept 13 — Token-Aware Chunk Sizing & Overlap
===============================================================
Tests token-based chunking with tiktoken (cl100k_base), metadata generation,
overlap configurations, edge cases, error handling, ordering, and context preservation.
"""

import unittest
import sys
import os
import tiktoken

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.document_loader import token_chunks, chunk_document, chunk_documents
from src.token_counter import get_tokenizer


class TestTokenAwareChunking(unittest.TestCase):

    def setUp(self):
        self.enc = get_tokenizer("cl100k_base")

    def test_1_empty_text(self):
        """1. Test empty text returns an empty list safely."""
        chunks_empty = token_chunks("", size=400, overlap=60)
        self.assertEqual(chunks_empty, [])

        chunks_spaces = token_chunks("   \n\t  ", size=400, overlap=60)
        self.assertEqual(chunks_spaces, [])

    def test_2_short_text(self):
        """2. Test short text smaller than overlap size produces a single chunk."""
        text = "Commercial invoice required."
        tokens = self.enc.encode(text)
        chunks = token_chunks(text, size=400, overlap=60)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], 1)
        self.assertEqual(chunks[0]["token_count"], len(tokens))
        self.assertEqual(chunks[0]["start_token"], 0)
        self.assertEqual(chunks[0]["end_token"], len(tokens))
        self.assertEqual(chunks[0]["text"], text)

    def test_3_text_smaller_than_chunk_size(self):
        """3. Test text larger than overlap but smaller than chunk size produces single chunk."""
        words = ["customs"] * 150  # ~150 tokens
        text = " ".join(words)
        tokens = self.enc.encode(text)

        chunks = token_chunks(text, size=400, overlap=60)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["token_count"], len(tokens))
        self.assertEqual(chunks[0]["start_token"], 0)
        self.assertEqual(chunks[0]["end_token"], len(tokens))

    def test_4_text_larger_than_chunk_size(self):
        """4. Test text larger than chunk size splits into multiple chunks with overlap."""
        words = ["duty"] * 500  # ~500 tokens
        text = " ".join(words)

        chunks = token_chunks(text, size=400, overlap=60)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["token_count"], 400)
        self.assertEqual(chunks[0]["start_token"], 0)
        self.assertEqual(chunks[0]["end_token"], 400)
        self.assertEqual(chunks[1]["start_token"], 340)  # 400 - 60

    def test_5_exact_chunk_size_text(self):
        """5. Test text matching exact chunk size (400 tokens) produces exactly 1 chunk."""
        token_id = self.enc.encode(" duty")[0]
        exact_text = self.enc.decode([token_id] * 400)
        tokens = self.enc.encode(exact_text)

        self.assertEqual(len(tokens), 400)

        chunks = token_chunks(exact_text, size=400, overlap=60)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["token_count"], 400)
        self.assertEqual(chunks[0]["start_token"], 0)
        self.assertEqual(chunks[0]["end_token"], 400)

    def test_6_overlap_zero(self):
        """6. Test chunking with overlap = 0 (no overlap between chunks)."""
        words = ["tariff"] * 600
        text = " ".join(words)
        tokens = self.enc.encode(text)

        chunks = token_chunks(text, size=400, overlap=0)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["start_token"], 0)
        self.assertEqual(chunks[0]["end_token"], 400)
        self.assertEqual(chunks[1]["start_token"], 400)
        self.assertEqual(chunks[1]["end_token"], len(tokens))

    def test_7_overlap_sixty(self):
        """7. Test chunking with overlap = 60 verifies shared token window."""
        words = ["logistics"] * 600
        text = " ".join(words)

        chunks = token_chunks(text, size=400, overlap=60)
        self.assertGreaterEqual(len(chunks), 2)

        c1 = chunks[0]
        c2 = chunks[1]

        # Check overlapping token indices
        self.assertEqual(c1["start_token"], 0)
        self.assertEqual(c1["end_token"], 400)
        self.assertEqual(c2["start_token"], 340)
        self.assertEqual(c1["end_token"] - c2["start_token"], 60)

    def test_8_invalid_overlap_raises_value_error(self):
        """8. Test invalid overlap parameters raise ValueError."""
        text = "Test text for validation."

        # Overlap equal to size
        with self.assertRaises(ValueError):
            token_chunks(text, size=400, overlap=400)

        # Overlap greater than size
        with self.assertRaises(ValueError):
            token_chunks(text, size=400, overlap=450)

        # Negative overlap
        with self.assertRaises(ValueError):
            token_chunks(text, size=400, overlap=-10)

        # Non-positive chunk size
        with self.assertRaises(ValueError):
            token_chunks(text, size=0, overlap=10)

        with self.assertRaises(ValueError):
            token_chunks(text, size=-100, overlap=10)

    def test_9_correct_token_counts(self):
        """9. Test that chunk token counts match actual tiktoken counts, not character counts."""
        text = "Pneumonoultramicroscopicsilicovolcanoconiosis antidisestablishmentarianism 12345 !@#$%"
        char_length = len(text)
        actual_token_count = len(self.enc.encode(text))

        # Ensure token count differs from character length
        self.assertNotEqual(char_length, actual_token_count)

        chunks = token_chunks(text, size=10, overlap=2)
        total_tokens_counted = sum(c["token_count"] for c in chunks)
        for c in chunks:
            # Each chunk's text decoded back and re-encoded should match token_count
            self.assertEqual(c["token_count"], len(self.enc.encode(c["text"])))

    def test_10_correct_ordering_of_chunks(self):
        """10. Test that chunks maintain strictly incremental chunk_id and start_token sequence."""
        words = ["document"] * 1000
        text = " ".join(words)

        chunks = token_chunks(text, size=300, overlap=50)
        self.assertGreater(len(chunks), 1)

        prev_id = 0
        prev_start = -1

        for c in chunks:
            self.assertEqual(c["chunk_id"], prev_id + 1)
            self.assertGreater(c["start_token"], prev_start)
            prev_id = c["chunk_id"]
            prev_start = c["start_token"]

    def test_11_boundary_context_preservation(self):
        """11. Test boundary context preservation with overlap vs no overlap."""
        prefix = "Word " * 380
        critical = "CRITICAL BOUNDARY SENTENCE FOR CUSTOMS VERIFICATION."
        suffix = " Word" * 380
        text = prefix + critical + suffix

        # Case A: Overlap = 0
        chunks_zero = token_chunks(text, size=400, overlap=0)
        self.assertGreaterEqual(len(chunks_zero), 2)

        # Case B: Overlap = 60
        chunks_overlap = token_chunks(text, size=400, overlap=60)
        self.assertGreaterEqual(len(chunks_overlap), 2)

        # With 60 overlap, chunk 2 should start earlier and capture context surrounding critical sentence
        self.assertLess(chunks_overlap[1]["start_token"], chunks_zero[1]["start_token"])
        self.assertEqual(chunks_zero[1]["start_token"] - chunks_overlap[1]["start_token"], 60)

    def test_12_no_accidental_infinite_loops(self):
        """12. Test that edge-case parameters terminate quickly without infinite loops."""
        text = "Fast loop test." * 50

        # Step size = 1 (size=2, overlap=1)
        chunks = token_chunks(text, size=2, overlap=1)
        self.assertGreater(len(chunks), 1)

        # Large text with small chunk size
        chunks_small = token_chunks(text, size=10, overlap=5)
        self.assertGreater(len(chunks_small), 1)

    def test_document_loader_integration(self):
        """Test chunk_document and chunk_documents propagate token metadata properly."""
        doc = {
            "source": "customs_guide.txt",
            "text": "Import requirements for laptops. " * 50
        }
        chunks = chunk_document(doc, max_chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertIn("id", c)
            self.assertIn("text", c)
            self.assertIn("metadata", c)
            meta = c["metadata"]
            self.assertEqual(meta["source"], "customs_guide.txt")
            self.assertIn("token_count", meta)
            self.assertIn("start_token", meta)
            self.assertIn("end_token", meta)
            self.assertIn("overlap", meta)


if __name__ == "__main__":
    unittest.main()
