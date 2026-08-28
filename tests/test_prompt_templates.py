"""
Unit Tests for Prompt Templates & Reusable Prompt Design Module
===============================================================
Tests template definition with named placeholders, runtime dynamic value injection,
template reuse across features, separation from business logic, and error handling for missing values.
"""

import unittest
import sys
import os

# Ensure project root is in sys.path for test runner
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.prompt_templates import (
    PromptTemplate,
    render,
    ANSWER_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    BATCH_EVAL_TEMPLATE,
    STRUCTURED_JSON_TEMPLATE,
)


class TestPromptTemplates(unittest.TestCase):

    def test_define_template_with_named_placeholders(self):
        """Task 1: Test creating a PromptTemplate object and extracting named placeholders."""
        template_str = "Context:\n{context}\n\nQuestion: {question}"
        pt = PromptTemplate(template_str)
        
        self.assertEqual(pt.placeholders, ["context", "question"])
        self.assertEqual(str(pt), template_str)

    def test_inject_dynamic_values_at_runtime(self):
        """Task 2: Test injecting dynamic runtime values into template placeholders."""
        pt = PromptTemplate("System Role: {role} for domain: {domain}")
        rendered = pt.render(role="Compliance Officer", domain="Customs Duty")
        
        self.assertEqual(rendered, "System Role: Compliance Officer for domain: Customs Duty")

    def test_standalone_render_function(self):
        """Test standalone render helper function with string and PromptTemplate instances."""
        rendered_str = render("Hello {name}, your query is {query}", name="Alice", query="HS Code 8471")
        self.assertEqual(rendered_str, "Hello Alice, your query is HS Code 8471")

        rendered_pt = render(ANSWER_TEMPLATE, context="Doc snippet 1", question="Duty rate?")
        self.assertIn("Context:\nDoc snippet 1", rendered_pt)
        self.assertIn("Question:\nDuty rate?", rendered_pt)

    def test_reuse_template_across_multiple_features(self):
        """Task 3: Test reusing the same template structure across Chat and Batch features."""
        pt = ANSWER_TEMPLATE

        # Feature 1: Interactive RAG Chat Feature
        chat_render = render(
            pt,
            context="ShipRule platform CDLP customs duty policy for electronics.",
            question="What is the duty rate for laptops?"
        )
        self.assertIn("CDLP customs compliance assistant", chat_render)
        self.assertIn("laptops?", chat_render)

        # Feature 2: Batch Evaluator / CLI Feature
        batch_render = render(
            pt,
            context="Batch document payload item #104.",
            question="Verify compliance for HS code 8471.30."
        )
        self.assertIn("CDLP customs compliance assistant", batch_render)
        self.assertIn("HS code 8471.30", batch_render)

    def test_missing_placeholder_raises_error(self):
        """Test rendering with missing required placeholders raises ValueError."""
        pt = PromptTemplate("Context: {context} | Question: {question}")
        
        with self.assertRaises(ValueError) as ctx:
            pt.render(context="Only context provided")
        self.assertIn("Missing required placeholder values: question", str(ctx.exception))

    def test_invalid_template_creation_raises_error(self):
        """Test instantiating PromptTemplate with empty or non-string raises ValueError."""
        with self.assertRaises(ValueError):
            PromptTemplate("")

        with self.assertRaises(ValueError):
            PromptTemplate("   ")

    def test_predefined_standard_templates_exist(self):
        """Task 4: Test standard templates exist and are stored separately from logic."""
        self.assertIsInstance(ANSWER_TEMPLATE, PromptTemplate)
        self.assertIsInstance(SYSTEM_PROMPT_TEMPLATE, PromptTemplate)
        self.assertIsInstance(BATCH_EVAL_TEMPLATE, PromptTemplate)
        self.assertIsInstance(STRUCTURED_JSON_TEMPLATE, PromptTemplate)


if __name__ == "__main__":
    unittest.main()
