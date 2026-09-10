"""
Prompt Templates & Reusable Prompt Design Module
=================================================
Centralized repository of prompt templates, placeholder rendering functions,
and dynamic value injection logic for ShipRule / CDLP platform.

Keeps prompts separate from business logic to ensure consistency across features.
"""

import re
from typing import Dict, Any, List, Union


class PromptTemplate:
    """
    Represents a prompt template with named placeholders.
    Supports template validation, placeholder discovery, and dynamic injection.
    """

    def __init__(self, template_text: str):
        if not isinstance(template_text, str) or not template_text.strip():
            raise ValueError("Template text must be a non-empty string.")
        self.template_text = template_text
        self.placeholders = self._extract_placeholders(template_text)

    @staticmethod
    def _extract_placeholders(text: str) -> List[str]:
        """Extracts named placeholder variables in {variable_name} format."""
        # Find all {var_name} matches excluding escaped double braces {{ ... }}
        pattern = r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})"
        return list(dict.fromkeys(re.findall(pattern, text)))

    def render(self, **values: Any) -> str:
        """
        Injects dynamic values into the named placeholders of the template.
        Raises ValueError if required placeholders are missing.
        """
        missing = [p for p in self.placeholders if p not in values]
        if missing:
            raise ValueError(f"Missing required placeholder values: {', '.join(missing)}")
        return self.template_text.format(**values)

    def __str__(self) -> str:
        return self.template_text

    def __repr__(self) -> str:
        return f"<PromptTemplate placeholders={self.placeholders}>"


def render(template: Union[str, PromptTemplate], **values: Any) -> str:
    """
    Standalone render helper function to inject dynamic values into a template.
    Accepts either a PromptTemplate instance or a raw template string with placeholders.
    """
    if isinstance(template, PromptTemplate):
        return template.render(**values)
    elif isinstance(template, str):
        pt = PromptTemplate(template)
        return pt.render(**values)
    else:
        raise TypeError("Template must be a PromptTemplate instance or a string.")


def render_prompt(context: str, question: str) -> str:
    """
    Renders the standard shipping and customs prompt with context and question.
    """
    return ANSWER_TEMPLATE.render(context=context, question=question)


# =====================================================================
# Centralized Standard Prompt Templates (Separated from Business Logic)
# =====================================================================

SHIPPING_PROMPT_TEMPLATE = PromptTemplate(
    "You are a helpful shipping and customs assistant.\n\n"
    "Use the following retrieved context to answer the user's question.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Instructions:\n"
    "- Answer only using the provided context.\n"
    "- If the answer is not available in the context, clearly state that verified information is not available.\n"
    "- Do not invent shipping, customs, or regulatory information.\n\n"
    "Answer:"
)

ANSWER_TEMPLATE = PromptTemplate(
    "You are a CDLP customs compliance assistant. Answer ONLY from the context provided.\n"
    "Respond ONLY with a valid JSON object matching this schema:\n"
    "{{\"answer\": \"<factual response>\", \"sources\": [{{\"source\": \"<document>\", \"page\": \"<page>\"}}], \"confidence\": \"high|medium|low\", \"has_answer\": true|false}}\n"
    "If the answer is not available in the context, set \"answer\" to \"I don't have enough information in the provided shipping rules to answer this question.\", \"sources\" to [], \"confidence\" to \"low\", and \"has_answer\" to false.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}"
)

SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    "You are an AI assistant specialized in {domain}.\n"
    "Role: {role}\n"
    "Constraints: {constraints}"
)

BATCH_EVAL_TEMPLATE = PromptTemplate(
    "=== CDLP Batch Evaluation Item ===\n"
    "Category: {category}\n"
    "Target Domain: {domain}\n"
    "Retrieved Context:\n{context}\n\n"
    "User Query: {question}"
)

STRUCTURED_JSON_TEMPLATE = PromptTemplate(
    "You are a CDLP customs compliance assistant. Respond ONLY with a valid JSON object.\n"
    "JSON Schema: {{\"answer\": \"<factual response>\", \"sources\": [{{\"source\": \"<document>\", \"page\": \"<page>\"}}], \"confidence\": \"high|medium|low\", \"has_answer\": true|false}}\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

