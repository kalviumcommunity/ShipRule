"""
ShipRule CDLP - Strict Scope Guard Module
=========================================
Ensures the assistant ONLY answers queries related to logistics, customs duties,
import/export documentation, HS codes, shipment rules, tariffs, and customs clearance.
Refuses out-of-scope questions before vector retrieval or LLM execution.
"""

import re

OUT_OF_SCOPE_RESPONSE = (
    "I don't know. I'm only able to help with logistics, customs duties, "
    "import documentation, HS codes, and shipment-related rules."
)

# Domain keywords related to logistics, customs duties, tariffs, and shipping rules
LOGISTICS_KEYWORDS = [
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
    r"\bgoal\b", r"\bmetrics?\b", r"\bverification\b", r"\bgap\b", r"\btraceability\b"
]

# Explicit out-of-scope question patterns
OUT_OF_SCOPE_PATTERNS = [
    r"^\s*who\s+is\b",
    r"^\s*who\s+was\b",
    r"^\s*write\s+(?:me\s+)?(?:a\s+)?(?:python|javascript|code|program|script|c\+\+|java)\b",
    r"^\s*tell\s+me\s+a\s+joke\b",
    r"^\s*what(?:'s|\s+is)\s+the\s+weather\b",
    r"^\s*what(?:'s|\s+is)\s+the\s+capital\b",
    r"^\s*explain\s+quantum\b",
    r"^\s*recipe\s+for\b",
    r"^\s*joke\b",
]


def is_in_scope(user_query: str) -> bool:
    """
    Classifies a user query as IN-SCOPE (True) or OUT-OF-SCOPE (False).
    Must return True ONLY for questions related to logistics, customs duties,
    import/export documentation, HS codes, tariffs, and shipment rules.
    """
    if not user_query or not user_query.strip():
        return False

    query_lower = user_query.strip().lower()

    # 1. Check for explicit logistics/customs domain keywords
    has_logistics_keyword = any(re.search(pat, query_lower) for pat in LOGISTICS_KEYWORDS)

    # 2. Check for explicit out-of-scope query patterns
    is_out_of_scope_pattern = any(re.search(pat, query_lower) for pat in OUT_OF_SCOPE_PATTERNS)

    if is_out_of_scope_pattern and not has_logistics_keyword:
        return False

    if has_logistics_keyword:
        return True

    # Reject non-logistics / general knowledge queries by default
    return False
