import unittest
from chunking import chunk_text

class TestChunking(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(chunk_text(""), [])

    def test_short_string(self):
        text = "Hello world!"
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
        self.assertEqual(chunks, [text])

    def test_exact_chunk_size(self):
        text = "1234567890"
        chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
        self.assertEqual(chunks, [text])

    def test_multiple_chunks_with_overlap(self):
        text = "1234567890" # 10 chars
        # size 5, overlap 2. 
        # chunk 1: text[0:5] -> "12345"
        # start shifts by: 5 - 2 = 3. New start is 3.
        # chunk 2: text[3:8] -> "45678"
        # start shifts to 6.
        # chunk 3: text[6:10] -> "7890"
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=2)
        self.assertEqual(chunks, ["12345", "45678", "7890"])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=-1)
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=5, chunk_overlap=6)

if __name__ == "__main__":
    unittest.main()
