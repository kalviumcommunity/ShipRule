"""
ShipRule CDLP - Semantic Scope Guard Module
============================================
Ensures the assistant ONLY answers queries related to logistics, customs duties,
import/export documentation, HS codes, shipment rules, tariffs, and customs clearance.
Refuses out-of-scope questions before vector retrieval or LLM execution.
"""

import os
import json
import logging
import re
from typing import Optional, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_RESPONSE = (
    "I'm sorry, I can only help with questions related to customs duties, shipping rules, "
    "import/export regulations, trade compliance, HS codes, and shipment documentation. "
    "Please feel free to ask me anything related to these topics."
)

# Out-of-scope intent patterns (general knowledge, political, sports, recipes, general coding)
OUT_OF_SCOPE_PATTERNS = [
    r"^\s*who\s+is\b",
    r"^\s*who\s+was\b",
    r"\bprime\s+minister\b",
    r"\bpresident\b",
    r"^\s*write\s+(?:me\s+)?(?:a\s+)?(?:python|javascript|code|program|script|c\+\+|java)\b",
    r"^\s*tell\s+me\s+a\s+joke\b",
    r"^\s*what(?:'s|\s+is)\s+the\s+weather\b",
    r"^\s*what(?:'s|\s+is)\s+the\s+capital\b",
    r"^\s*explain\s+quantum\b",
    r"^\s*recipe\s+for\b",
    r"^\s*joke\b",
    r"^\s*who\s+won\b",
    r"\bmovie\b",
    r"\bsong\b",
]

# Supported domain intent concepts (customs, tariffs, HS codes, shipping, import/export, trade)
IN_SCOPE_PATTERNS = [
    r"\bcustoms?\b", r"\bduty\b", r"\bduties\b", r"\btariff\b", r"\btariffs\b",
    r"\bhs\s*codes?\b", r"\bhscode\b", r"\bhs-code\b", r"\bhs_code\b",
    r"\bimport\b", r"\bimports\b", r"\bimporting\b", r"\bimported\b", r"\bimporter\b",
    r"\bexport\b", r"\bexports\b", r"\bexporting\b", r"\bexported\b", r"\bexporter\b",
    r"\bshipment\b", r"\bshipments\b", r"\bshipping\b", r"\bship\b", r"\bfreight\b",
    r"\blogistics\b", r"\bcargo\b", r"\bclearance\b", r"\bbill of lading\b",
    r"\bcommercial invoice\b", r"\bcertificate of origin\b", r"\bbis certificate\b",
    r"\bbis registration\b", r"\bcif\b", r"\bfob\b", r"\bincoterms?\b", r"\bdeclaration\b",
    r"\bdgft\b", r"\btrade regulation\b", r"\bcourier\b", r"\bconsignee\b", r"\bconsignor\b",
    r"\bport of entry\b", r"\bport of origin\b", r"\bexcise\b", r"\btaxation\b", r"\btaxes?\b",
    r"\bborder\b", r"\brestricted goods\b", r"\bimport license\b", r"\bcustoms officer\b",
    r"\bcdlp\b", r"\bprd\b", r"\bpersona\b", r"\brequirement\b", r"\bsla\b", r"\bmvp\b",
    r"\bgoal\b", r"\bmetrics?\b", r"\bverification\b", r"\bgap\b", r"\btraceability\b",
    r"\bdocuments?\b", r"\bpaperwork\b", r"\bpapers?\b", r"\brates?\b", r"\bvaluation\b",
    r"\bfta\b", r"\btrade\b", r"\bproduct\b", r"\bgoods\b", r"\bcarrier\b", r"\btransport\b",
    r"\bregulations?\b"
]


def is_in_scope(user_query: str, client: Optional[Any] = None) -> bool:
    """
    Classifies a user query as IN-SCOPE (True) or OUT-OF-SCOPE (False)
    based on semantic intent.

    Args:
        user_query: User question string.
        client: Optional Groq client instance.

    Returns:
        Boolean indicating if query is in scope.
    """
    if not user_query or not isinstance(user_query, str) or not user_query.strip():
        return False

    query_clean = user_query.strip().lower()

    # 1. Check explicit out-of-scope query intent patterns first
    for pat in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, query_clean):
            # Exception: if query explicitly mentions customs/shipping/import/export/duty/tariff
            if not any(re.search(in_pat, query_clean) for in_pat in [r"\bcustoms?\b", r"\bduty\b", r"\bduties\b", r"\btariff\b", r"\bimport\b", r"\bexport\b", r"\bshipping\b", r"\bshipment\b"]):
                return False

    # 2. Check for explicit customs / shipping domain patterns
    has_domain_concept = any(re.search(pat, query_clean) for pat in IN_SCOPE_PATTERNS)
    if has_domain_concept:
        return True

    # 3. Dynamic LLM semantic domain scope classification if API is available
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if (api_key or client) and not (api_key and api_key.startswith("gsk_local_key") and client is None):
        try:
            from groq import Groq
            groq_client = client or Groq(api_key=api_key)
            llm_model = os.getenv("CHAT_MODEL", "groq/compound-mini")

            system_instruction = (
                "You are a strict domain scope classifier for ShipRule CDLP.\n"
                "ShipRule is ONLY intended to answer questions regarding customs duties, tariffs, HS codes, "
                "import/export regulations, shipping documents, trade compliance, logistics, freight, and customs clearance.\n"
                "Classify whether the user query is IN SCOPE or OUT OF SCOPE.\n"
                "Respond ONLY with valid JSON: {\"is_in_scope\": true} or {\"is_in_scope\": false}."
            )
            user_prompt = f"User Query: {query_clean}"

            resp = groq_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=50,
                response_format={"type": "json_object"} if hasattr(groq_client, "chat") else None
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            if "is_in_scope" in data:
                return bool(data["is_in_scope"])
        except Exception as e:
            logger.warning(f"LLM scope classification encountered error: {e}. Falling back to default scope guard.")

    # Default fallback: if no domain concept found, reject out-of-domain queries
    return False
