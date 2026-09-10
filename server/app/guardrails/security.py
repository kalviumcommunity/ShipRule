"""
ShipRule CDLP - Security & Sensitive Information Guardrail Module
==================================================================
Protects internal administrative credentials, access keys, system prompts,
environment secrets, database configurations, and guards against prompt-injection
attacks prior to retrieval or LLM completion.

STRICT PRINCIPLE:
Guardrails take highest priority. Never leak protected credentials, admin keys,
passwords, JWT secrets, or system prompts under any phrasing or injection attempt.
"""

import re
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

SECURITY_REFUSAL_MESSAGE = (
    "I'm sorry, but I cannot provide internal administrative credentials, "
    "access keys, system prompts, or protected system configuration."
)

# Sensitive patterns targeting admin keys, credentials, system instructions, or prompt injection
SENSITIVE_PATTERNS = [
    # Admin access keys and passwords
    r"\bkrishna7\b",
    r"\bAbhi@2006\b",
    r"\badmin\s*key\b",
    r"\bmaster\s*key\b",
    r"\baccess\s*key\b",
    r"\badmin\s*password\b",
    r"\bmaster\s*password\b",
    r"\blogin\s*credentials?\b",
    r"\bbackend\s*credentials?\b",
    r"\bdatabase\s*credentials?\b",
    r"\bjwt\s*secret\b",
    r"\bgroq_api_key\b",
    r"\.?env\b",
    r"env\s+secrets?",

    # Prompt injection & system prompt extraction
    r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions",
    r"reveal\s+(?:the\s+)?system\s+prompt",
    r"show\s+(?:me\s+)?(?:the\s+)?system\s+(?:prompt|instructions)",
    r"what\s+is\s+(?:your\s+)?system\s+prompt",
    r"print\s+(?:the\s+)?system\s+prompt",
    r"display\s+(?:the\s+)?system\s+prompt",
    r"act\s+as\s+(?:an\s+)?admin",
    r"bypass\s+security",
    r"show\s+(?:me\s+)?admin\s+credentials",
    r"what\s+is\s+(?:the\s+)?admin\s+(?:key|password|route)"
]


def check_security_and_system_protection(query: str) -> Dict[str, Any]:
    """
    Evaluates a user query for security risks, prompt injection attempts,
    or requests for protected system secrets / admin credentials.

    Args:
        query: User search question string.

    Returns:
        Structured security evaluation dictionary:
        {
            "is_security_risk": bool,
            "risk_type": Optional[str],
            "decision": str ("ALLOW" | "REFUSE"),
            "message": str
        }
    """
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "is_security_risk": False,
            "risk_type": None,
            "decision": "ALLOW",
            "message": ""
        }

    clean_query = query.strip().lower()

    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, clean_query, re.IGNORECASE):
            logger.warning(f"Security guardrail triggered for query pattern '{pattern}': {query[:50]}")
            return {
                "is_security_risk": True,
                "risk_type": "sensitive_info_or_injection_attempt",
                "decision": "REFUSE",
                "message": SECURITY_REFUSAL_MESSAGE
            }

    return {
        "is_security_risk": False,
        "risk_type": None,
        "decision": "ALLOW",
        "message": ""
    }


def sanitize_answer_content(answer_text: str) -> str:
    """
    Scans generated RAG answer output for accidental credential, API key,
    or secret token leaks and redacts them prior to client delivery.

    Args:
        answer_text: Generated answer text.

    Returns:
        Sanitized answer text string.
    """
    if not answer_text or not isinstance(answer_text, str):
        return ""

    sanitized = answer_text
    # Redact sensitive credentials if present in generated text
    sanitized = re.sub(r"krishna7", "[REDACTED_ADMIN_KEY]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"Abhi@2006", "[REDACTED_PASSWORD]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"gsk_[a-zA-Z0-9_\-]+", "[REDACTED_API_KEY]", sanitized)

    return sanitized
