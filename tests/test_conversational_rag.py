"""
Unit Tests for Conversational RAG & Follow-Up Context Module (KDU 3.42)
=======================================================================
Tests history tracking across turns, follow-up query rewriting, strength-checked
context retrieval using rewritten queries, and multi-turn grounded Q&A.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.conversational_rag import (
    rewrite_followup,
    retrieval_is_strong,
    conversational_answer,
    ConversationalRAGManager,
    format_history_for_rewriter
)


class TestConversationalRAG(unittest.TestCase):

    def setUp(self):
        self.sample_history = [
            {"role": "user", "content": "What evidence is required for project submission?"},
            {"role": "assistant", "content": "The submission needs a PR link, sample output, and a video explanation with source evidence."}
        ]

    # --------------------------------------------------------------------------
    # TASK 1: History Tracking Tests
    # --------------------------------------------------------------------------
    def test_history_tracking_accrual(self):
        """Verifies that user questions and assistant answers accrue properly in history."""
        manager = ConversationalRAGManager(collection=MagicMock(), max_history_tokens=1000)
        self.assertEqual(len(manager.history), 0)

        with patch("src.conversational_rag.rewrite_followup", return_value="What evidence is required for project submission?"), \
             patch("src.conversational_rag.retrieve_context", return_value=([{"metadata": {"source": "test.md"}, "rerank_score": 0.9}], {})), \
             patch("src.conversational_rag.retrieval_is_strong", return_value=True), \
             patch("src.conversational_rag.generate_answer", return_value={"answer": "A PR link and video explanation.", "sources": ["test.md"]}):

            res1 = manager.ask("What evidence is required for project submission?")
            self.assertEqual(len(manager.history), 2)
            self.assertEqual(manager.history[0]["role"], "user")
            self.assertEqual(manager.history[1]["role"], "assistant")

            res2 = manager.ask("What about the video?")
            self.assertEqual(len(manager.history), 4)
            self.assertEqual(manager.history[2]["content"], "What about the video?")

    def test_history_reset(self):
        """Verifies history reset clears stored turns."""
        manager = ConversationalRAGManager(collection=MagicMock())
        manager.history = list(self.sample_history)
        self.assertEqual(len(manager.history), 2)
        manager.reset_history()
        self.assertEqual(len(manager.history), 0)

    # --------------------------------------------------------------------------
    # TASK 2: Follow-up Question Rewriting Tests
    # --------------------------------------------------------------------------
    def test_rewrite_followup_empty_history_returns_original(self):
        """Verifies that an empty history returns the original question intact."""
        q = "What is the duty rate for cars in Italy?"
        rewritten = rewrite_followup(history=[], question=q)
        self.assertEqual(rewritten, q)

    @patch("src.conversational_rag.call_llm")
    def test_rewrite_followup_with_history(self, mock_call_llm):
        """Verifies that follow-up questions are passed to LLM and stripped of prefixes."""
        mock_call_llm.return_value = 'Standalone Query: "What video explanation is required for project submission?"'
        
        rewritten = rewrite_followup(
            history=self.sample_history,
            question="What about the video?"
        )
        self.assertEqual(rewritten, "What video explanation is required for project submission?")
        mock_call_llm.assert_called_once()

    # --------------------------------------------------------------------------
    # TASK 3: Strength-checked Retrieval with Rewritten Query
    # --------------------------------------------------------------------------
    def test_retrieval_is_strong_valid_chunks(self):
        """Verifies retrieval_is_strong returns True for relevant non-empty chunks."""
        chunks = [
            {
                "chunk_text": "Required documents: Commercial invoice, packing list.",
                "rerank_score": 0.85,
                "distance": 0.3
            }
        ]
        self.assertTrue(retrieval_is_strong(chunks))

    def test_retrieval_is_strong_empty_chunks(self):
        """Verifies retrieval_is_strong returns False when no chunks are retrieved."""
        self.assertFalse(retrieval_is_strong([]))
        self.assertFalse(retrieval_is_strong(None))

    # --------------------------------------------------------------------------
    # TASK 4: Multi-turn Conversational Answer Flow & Grounded Q&A
    # --------------------------------------------------------------------------
    @patch("src.conversational_rag.generate_answer")
    @patch("src.conversational_rag.retrieve_context")
    @patch("src.conversational_rag.rewrite_followup")
    def test_conversational_answer_strong_retrieval(self, mock_rewrite, mock_retrieve, mock_gen_answer):
        """Verifies end-to-end multi-turn answer flow when retrieval is strong."""
        mock_rewrite.return_value = "What video explanation is required for project submission?"
        mock_retrieve.return_value = (
            [{"metadata": {"source": "guidelines.md"}, "rerank_score": 0.9, "chunk_text": "Video walkthrough required."}],
            {}
        )
        mock_gen_answer.return_value = {
            "answer": "The submission requires a 3-5 minute screen-share video.",
            "sources": ["guidelines.md"]
        }

        history = list(self.sample_history)
        res = conversational_answer(history=history, user_question="What about the video?")

        self.assertEqual(res["rewritten_query"], "What video explanation is required for project submission?")
        self.assertIn("3-5 minute screen-share video", res["answer"])
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(len(history), 4)

    @patch("src.conversational_rag.retrieve_context")
    @patch("src.conversational_rag.rewrite_followup")
    def test_conversational_answer_weak_retrieval_refusal(self, mock_rewrite, mock_retrieve):
        """Verifies safe refusal answer when retrieval yields no relevant chunks."""
        mock_rewrite.return_value = "What is the deadline for space shuttles?"
        mock_retrieve.return_value = ([], {})

        history = []
        res = conversational_answer(history=history, user_question="What about the deadline for space shuttles?")

        self.assertEqual(res["answer"], "I don't have enough reliable context to answer that.")
        self.assertEqual(len(res["sources"]), 0)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["content"], "I don't have enough reliable context to answer that.")


if __name__ == "__main__":
    unittest.main()
