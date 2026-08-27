"""
Unit Tests for Context Window & Message History Management Module
==================================================================
Tests total_tokens calculation, trim_history, summarize_history,
prepare_context, and ContextManager class under various context budget conditions.
"""

import unittest
import sys
import os

# Ensure project root is in sys.path for running tests
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.context_manager import (
    total_tokens,
    trim_history,
    summarize_history,
    prepare_context,
    ContextManager,
    _truncate_text_to_tokens,
)
from src.token_counter import count_tokens


class TestContextManager(unittest.TestCase):

    def setUp(self):
        self.system_prompt = "You are an AI Support Assistant for Customs Duty & Documentation Lookup Platform."
        self.sample_user_msg = "What are the required import documents for laptop computers to India?"
        self.sample_assistant_msg = "You need a Commercial Invoice, Bill of Lading, and BIS Registration Certificate."

    def test_total_tokens_calculation(self):
        """Test total_tokens includes message content, role keys, and ChatML overhead."""
        messages = [
            {"role": "system", "content": "System prompt text."},
            {"role": "user", "content": "Hello model!"}
        ]
        tok = total_tokens(messages)
        self.assertGreater(tok, 0)
        # Overhead per msg (4*2=8) + primer (2) + content tok + role tok ("system", "user" = 2 tok)
        expected_total = count_tokens("System prompt text.") + count_tokens("Hello model!") + 10 + count_tokens("system") + count_tokens("user")
        self.assertEqual(tok, expected_total)

    def test_normal_multiturn_conversation(self):
        """Test multi-turn conversation within token budget preserves all history."""
        cm = ContextManager(system_prompt=self.system_prompt, max_context_tokens=1000, response_reserve_tokens=100)
        
        res1 = cm.ask("What is ShipRule?", llm_fn=lambda msgs: "ShipRule is a customs platform.")
        self.assertEqual(len(cm.history), 3) # sys + user + assistant
        
        res2 = cm.ask("How does it verify duties?", llm_fn=lambda msgs: "It uses source traceability.")
        self.assertEqual(len(cm.history), 5) # sys + 2 user + 2 assistant
        self.assertEqual(res2["strategy_applied"], "none")

    def test_system_message_preservation(self):
        """Test system prompt is always preserved even when severe trimming occurs."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "Message 1 " * 50},
            {"role": "assistant", "content": "Response 1 " * 50},
            {"role": "user", "content": "Message 2 " * 50},
            {"role": "assistant", "content": "Response 2 " * 50},
            {"role": "user", "content": "Message 3 " * 50},
        ]
        
        # Set tight budget so trimming must happen
        sys_tokens = total_tokens([messages[0]])
        tight_budget = sys_tokens + 150
        
        trimmed_msgs, final_tokens, strategy = trim_history(messages, budget=tight_budget, preserve_recent=1)
        
        # Verify system message is at index 0 and unmodified
        self.assertEqual(trimmed_msgs[0]["role"], "system")
        self.assertEqual(trimmed_msgs[0]["content"], self.system_prompt)
        self.assertLessEqual(final_tokens, tight_budget)

    def test_trim_behavior_pair_removal(self):
        """Test trim strategy removes oldest user/assistant pairs first."""
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User question 1"},
            {"role": "assistant", "content": "Assistant answer 1"},
            {"role": "user", "content": "User question 2"},
            {"role": "assistant", "content": "Assistant answer 2"},
            {"role": "user", "content": "User question 3"},
        ]
        
        budget = total_tokens(messages) - 35
        trimmed, final_tok, strategy = trim_history(messages, budget=budget)
        
        self.assertEqual(strategy, "trim")
        self.assertEqual(trimmed[0]["role"], "system")
        # Turn 1 should be removed
        self.assertNotIn("User question 1", [m["content"] for m in trimmed])
        self.assertIn("User question 3", trimmed[-1]["content"])

    def test_summarize_behavior(self):
        """Test summarize_history condenses older turns while retaining recent turns intact."""
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Older query 1: Tell me about HS Code 8471 for computers and peripherals import details. " * 3},
            {"role": "assistant", "content": "Older answer 1: HS Code 8471 covers automatic data processing machines and units thereof. " * 3},
            {"role": "user", "content": "Older query 2: What about BIS certificate requirement for imported electronics? " * 3},
            {"role": "assistant", "content": "Older answer 2: BIS registration is compulsory in India for safety standard compliance. " * 3},
            {"role": "user", "content": "Recent query: What is the tariff percentage?"},
            {"role": "assistant", "content": "Recent answer: Basic customs duty rate is 7.5%."},
            {"role": "user", "content": "Current question: Any extra surcharge?"}
        ]

        full_tokens = total_tokens(messages)

        summarized, final_tok, strategy = summarize_history(messages, budget=full_tokens - 40, preserve_recent=1)

        self.assertEqual(strategy, "summarize")
        self.assertEqual(summarized[0]["role"], "system")
        # Verify a summary message exists in history
        summary_found = any("Context Summary" in m.get("content", "") for m in summarized)
        self.assertTrue(summary_found)
        self.assertLessEqual(final_tok, full_tokens - 40)

    def test_long_user_message_handling(self):
        """Test an extraordinarily long user message exceeding budget is handled gracefully."""
        huge_user_msg = "Customs classification text. " * 300
        cm = ContextManager(system_prompt=self.system_prompt, max_context_tokens=300, response_reserve_tokens=50)

        prep = cm.get_prepared_payload(user_message=huge_user_msg)
        
        self.assertLessEqual(prep["total_tokens"], prep["budget"])
        self.assertIn("Truncated", prep["messages"][-1]["content"])

    def test_rag_context_budget_consumption(self):
        """Test when RAG retrieved documents consume the budget, context is trimmed to fit."""
        huge_rag_context = "Document chunk rule line. " * 400
        user_question = "What is the tariff for laptops?"

        prep = prepare_context(
            history=[],
            retrieved_context=huge_rag_context,
            user_message=user_question,
            max_tokens=500,
            reserve_tokens=100,
            system_prompt=self.system_prompt
        )

        self.assertLessEqual(prep["total_tokens"], prep["budget"])
        self.assertIn("Truncated", prep["messages"][-1]["content"])
        self.assertIn("Question:\nWhat is the tariff for laptops?", prep["messages"][-1]["content"])

    def test_multiple_consecutive_trimming_operations(self):
        """Test history maintains continuity and stability across 10 consecutive turns under tight budget."""
        cm = ContextManager(
            system_prompt=self.system_prompt,
            max_context_tokens=250,
            response_reserve_tokens=50,
            strategy="trim"
        )

        for i in range(10):
            res = cm.ask(
                user_message=f"Turn {i}: Query about customs document rule #{i}",
                llm_fn=lambda msgs: f"Response to turn {i}."
            )
            # Verify budget constraint maintained at every turn
            self.assertLessEqual(res["total_tokens"], res["budget"])
            self.assertEqual(cm.history[0]["role"], "system")

    def test_conversation_continuity_after_pruning(self):
        """Test conversation flow remains valid role-alternating structure after trimming."""
        cm = ContextManager(
            system_prompt=self.system_prompt,
            max_context_tokens=350,
            response_reserve_tokens=50,
            strategy="trim"
        )

        for i in range(5):
            cm.ask(f"Question {i} " * 15, llm_fn=lambda msgs: f"Answer {i} " * 15)

        # Inspect history role ordering: System, User, Assistant, User, Assistant ...
        roles = [m["role"] for m in cm.history]
        self.assertEqual(roles[0], "system")
        for idx in range(1, len(roles)):
            if idx % 2 == 1:
                self.assertEqual(roles[idx], "user")
            else:
                self.assertEqual(roles[idx], "assistant")


if __name__ == "__main__":
    unittest.main()
