"""
ShipRule CDLP - Answer Grounding & Source Accuracy Validator
============================================================
Provides answer-grounding validation and source citation verification for RAG responses.

Key capabilities:
1. Extracts citation markers (e.g. [1], [2], [3]) from generated AI responses.
2. Compares extracted citations against available source markers from context assembly.
3. Detects and flags unsupported or hallucinated citation markers.
4. Validates absence of unsupported claims when context is missing.
5. Generates standardized ASCII validation reports and structured validation payloads.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Union

logger = logging.getLogger(__name__)

INSUFFICIENT_CONTEXT_PHRASE = "The provided context is insufficient to answer this question."


def extract_citation_markers(text: str) -> List[str]:
    """
    Extracts all citation marker strings (e.g., '[1]', '[2]', '[12]') from text.
    Maintains first-occurrence ordering while removing duplicates.

    Args:
        text: Response or context text string.

    Returns:
        List of unique citation marker strings in order of appearance (e.g. ['[1]', '[2]']).
    """
    if not text or not isinstance(text, str):
        return []

    pattern = r"\[\s*(\d+)\s*\]"
    matches = re.findall(pattern, text)

    unique_markers = []
    seen = set()
    for m in matches:
        marker = f"[{m}]"
        if marker not in seen:
            seen.add(marker)
            unique_markers.append(marker)

    return unique_markers


def format_validation_report(
    answer_generated: bool,
    used_markers: List[str],
    available_markers: List[str],
    unsupported_markers: List[str],
    source_validation: str,
    grounding_status: str
) -> str:
    """Formats standardized ASCII validation report block."""
    used_str = ", ".join(used_markers) if used_markers else "None"
    avail_str = ", ".join(available_markers) if available_markers else "None"
    unsupported_str = ", ".join(unsupported_markers) if unsupported_markers else "None"

    lines = [
        "[Answer Validation]",
        f"Answer Generated       : {'YES' if answer_generated else 'NO'}",
        f"Source Markers Used    : {used_str}",
        f"Available Source Marks : {avail_str}",
        f"Unsupported Markers    : {unsupported_str}",
        f"Source Validation      : {source_validation}",
        f"Grounding Status       : {grounding_status}"
    ]
    return "\n".join(lines)


def validate_answer_sources(
    answer: str,
    available_sources: Union[List[str], List[Dict[str, Any]], Set[str]],
    is_retrieval_mode: bool = True
) -> Dict[str, Any]:
    """
    Validates that citations in the generated answer match available source markers.

    Args:
        answer: The generated answer string.
        available_sources: List of citation marker strings (e.g. ['[1]', '[2]'])
                           OR list of source mapping dictionaries containing 'marker' or 'citation_id'.
        is_retrieval_mode: Boolean indicating if retrieval context was provided.

    Returns:
        Structured validation dictionary with:
        - is_valid (bool): True if no unsupported citations exist.
        - validation_status (str): 'PASS' or 'FAIL'.
        - grounding_status (str): 'GROUNDED', 'REVIEW REQUIRED', or 'NO RETRIEVAL CONTEXT'.
        - used_markers (List[str]): Markers found in the answer.
        - available_markers (List[str]): Markers provided in the context.
        - unsupported_markers (List[str]): Markers in answer not in available_markers.
        - is_insufficient_context_fallback (bool): True if answer states context insufficiency.
        - report (str): Standard ASCII validation report.
    """
    answer_str = str(answer or "").strip()
    has_answer = bool(answer_str)

    # 1. Normalize available source markers to list of strings like ['[1]', '[2]']
    norm_available: List[str] = []
    if available_sources:
        if isinstance(available_sources, (list, tuple, set)):
            for item in available_sources:
                if isinstance(item, dict):
                    m = item.get("marker") or item.get("citation_id")
                    if m:
                        norm_available.append(str(m).strip())
                elif isinstance(item, (str, int)):
                    s = str(item).strip()
                    if not s.startswith("[") and s.isdigit():
                        s = f"[{s}]"
                    norm_available.append(s)

    # Deduplicate while preserving order
    available_markers = list(dict.fromkeys(norm_available))

    # 2. Extract citations used in answer
    used_markers = extract_citation_markers(answer_str)

    # 3. Detect unsupported citation markers
    avail_set = set(available_markers)
    unsupported_markers = [m for m in used_markers if m not in avail_set]

    # 4. Check for insufficient context fallback phrase
    is_fallback = INSUFFICIENT_CONTEXT_PHRASE.lower() in answer_str.lower() or "insufficient" in answer_str.lower()

    # 5. Determine Validation and Grounding Status
    if not is_retrieval_mode:
        validation_status = "PASS" if not unsupported_markers else "FAIL"
        grounding_status = "NO RETRIEVAL CONTEXT" if not unsupported_markers else "REVIEW REQUIRED"
    elif unsupported_markers:
        validation_status = "FAIL"
        grounding_status = "REVIEW REQUIRED"
    else:
        validation_status = "PASS"
        grounding_status = "GROUNDED"

    is_valid = (validation_status == "PASS")

    report_str = format_validation_report(
        answer_generated=has_answer,
        used_markers=used_markers,
        available_markers=available_markers,
        unsupported_markers=unsupported_markers,
        source_validation=validation_status,
        grounding_status=grounding_status
    )

    return {
        "is_valid": is_valid,
        "validation_status": validation_status,
        "grounding_status": grounding_status,
        "used_markers": used_markers,
        "available_markers": available_markers,
        "unsupported_markers": unsupported_markers,
        "is_insufficient_context_fallback": is_fallback,
        "report": report_str
    }
