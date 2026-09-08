"""
ShipRule CDLP - Citation Manager & Source Verifiability Module
==============================================================
Provides end-to-end citation mapping, source registry management, and citation
verifiability for the Customs Duty & Documentation Lookup Platform (CDLP).

Key Capabilities:
1. Build citation registries mapping sequentially indexed markers ([1], [2], [3]...)
   to real chunk metadata and original retrieved text.
2. Trace citations back to exact source documents, chunk IDs, chunk indexes, sections, and pages.
3. Validate used citations in generated answers against available citation registries.
4. Detect fabricated or unsupported citations and reject invalid references.
5. Provide detailed user verification views for inspecting source citations.
6. Export standardized citation mapping JSON and verification reports.
"""

import os
import sys
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)


def extract_citations(text: str) -> List[str]:
    """
    Extracts citation markers (e.g., '[1]', '[2]', '[3]') from text.
    Maintains first-occurrence order and removes duplicates.

    Args:
        text: Input string containing citations.

    Returns:
        List of unique citation marker strings (e.g., ['[1]', '[2]']).
    """
    if not text or not isinstance(text, str):
        return []

    pattern = r"\[\s*(\d+)\s*\]"
    matches = re.findall(pattern, text)

    unique_citations = []
    seen = set()
    for m in matches:
        marker = f"[{m}]"
        if marker not in seen:
            seen.add(marker)
            unique_citations.append(marker)

    return unique_citations


def build_citation_registry(
    retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Constructs a citation registry mapping sequential markers ([1], [2], ...)
    to actual retrieved chunk metadata and original retrieved text.

    Args:
        retrieved_chunks: List of retrieved candidate chunks from ChromaDB / retrieval.

    Returns:
        Dictionary mapping citation markers (e.g. '[1]') to source chunk metadata.
    """
    registry: Dict[str, Dict[str, Any]] = {}
    if not retrieved_chunks or not isinstance(retrieved_chunks, list):
        return registry

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        if not isinstance(chunk, dict):
            continue

        marker = f"[{idx}]"
        meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}

        # 1. Source Document
        source = chunk.get("source") or meta.get("source") or "unknown_source"

        # 2. Chunk ID
        chunk_id = (
            chunk.get("chunk_id")
            or chunk.get("id")
            or chunk.get("embedding_id")
            or meta.get("chunk_id")
            or meta.get("embedding_id")
            or f"chunk_{idx:03d}"
        )

        # 3. Chunk Index
        chunk_idx = chunk.get("chunk_index")
        if chunk_idx is None:
            chunk_idx = chunk.get("chunk")
        if chunk_idx is None and meta:
            chunk_idx = meta.get("chunk_index", meta.get("chunk"))
        if chunk_idx is None:
            chunk_idx = idx - 1
        try:
            chunk_idx = int(chunk_idx)
        except (ValueError, TypeError):
            pass

        # 4. Section (null if not provided or N/A)
        section = chunk.get("section") or meta.get("section")
        if section and str(section).strip().lower() in ("n/a", "none", "null", ""):
            section = None

        # 5. Page (null if not provided or N/A)
        raw_page = chunk.get("page")
        if raw_page is None and meta:
            raw_page = meta.get("page")
        page = None
        if raw_page is not None and str(raw_page).strip().lower() not in ("n/a", "none", "null", ""):
            try:
                page = int(str(raw_page).strip())
            except ValueError:
                page = str(raw_page).strip()

        # 6. Document Type
        doc_type = (
            chunk.get("document_type")
            or chunk.get("doc_type")
            or meta.get("document_type")
            or meta.get("doc_type")
        )
        if doc_type and str(doc_type).strip().lower() in ("n/a", "none", "null", ""):
            doc_type = None

        # 7. Original Text (Preserve full retrieved text without truncation)
        original_text = chunk.get("chunk_text") or chunk.get("text") or ""

        # Construct registry entry
        entry: Dict[str, Any] = {
            "source": str(source),
            "chunk_id": str(chunk_id),
            "chunk_index": chunk_idx,
            "section": str(section) if section is not None else None,
            "page": page,
            "original_text": str(original_text)
        }

        if doc_type:
            entry["document_type"] = str(doc_type)

        # Include optional metadata if present and valid
        for key in ["country", "hs_code", "source_agency", "source_url", "last_confirmed_date"]:
            val = chunk.get(key) or meta.get(key)
            if val is not None and str(val).strip().lower() not in ("n/a", "none", "null", ""):
                entry[key] = str(val).strip()

        # Additional retrieval metadata
        if "rank" in chunk:
            entry["rank"] = chunk["rank"]
        else:
            entry["rank"] = idx

        if "rerank_score" in chunk:
            entry["score"] = chunk["rerank_score"]
        elif "similarity_score" in chunk:
            entry["score"] = chunk["similarity_score"]

        registry[marker] = entry

    return registry


def verify_citation(
    citation_marker: str,
    registry: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Verifies a single citation marker against the citation registry.

    Checks:
    1. Marker exists in registry.
    2. Marker maps to a retrieved chunk.
    3. Source metadata exists.
    4. Original retrieved text exists.

    Args:
        citation_marker: String marker like '[1]'.
        registry: Citation registry dictionary.

    Returns:
        Structured verification result dictionary.
    """
    marker = str(citation_marker).strip()
    if not marker.startswith("[") and marker.isdigit():
        marker = f"[{marker}]"

    if not registry or marker not in registry:
        return {
            "citation": marker,
            "status": "INVALID",
            "is_verified": False,
            "reason": "Citation does not exist in retrieved context",
            "source": None,
            "chunk_id": None,
            "chunk_index": None,
            "section": None,
            "page": None,
            "original_text_available": False
        }

    entry = registry[marker]
    source = entry.get("source")
    chunk_id = entry.get("chunk_id")
    chunk_idx = entry.get("chunk_index")
    section = entry.get("section")
    page = entry.get("page")
    text = entry.get("original_text", "")

    return {
        "citation": marker,
        "status": "VERIFIED",
        "is_verified": True,
        "reason": None,
        "source": source,
        "chunk_id": chunk_id,
        "chunk_index": chunk_idx,
        "section": section,
        "page": page,
        "original_text_available": bool(text and text.strip())
    }


def verify_answer_citations(
    answer: str,
    registry: Dict[str, Dict[str, Any]],
    is_retrieval_mode: bool = True
) -> Dict[str, Any]:
    """
    Verifies all citations contained in a generated answer against the citation registry.

    Detects fabricated/unknown citations, validates supported claims, and outputs
    overall verification status and formatted reports.

    Args:
        answer: Generated answer text.
        registry: Citation registry dictionary mapping markers to metadata.
        is_retrieval_mode: True if vector retrieval was used.

    Returns:
        Comprehensive verification dictionary with status, reports, and mappings.
    """
    answer_text = str(answer or "").strip()
    used_citations = extract_citations(answer_text)
    available_citations = list(registry.keys()) if registry else []

    verified_details: List[Dict[str, Any]] = []
    unsupported_citations: List[str] = []

    for marker in used_citations:
        res = verify_citation(marker, registry)
        verified_details.append(res)
        if not res["is_verified"]:
            unsupported_citations.append(marker)

    # Determine status
    if not is_retrieval_mode:
        if unsupported_citations:
            verification_status = "FAIL"
            grounding_status = "FAILED"
        else:
            verification_status = "PASS"
            grounding_status = "NO RETRIEVAL CONTEXT"
    elif unsupported_citations:
        verification_status = "FAIL"
        grounding_status = "FAILED"
    else:
        verification_status = "PASS"
        grounding_status = "GROUNDED"

    is_valid = (verification_status == "PASS")

    # Build ASCII verification report
    report_lines = ["[Source Verification]"]
    if not used_citations:
        report_lines.append("Citations           : NONE")
        report_lines.append(f"Verification Status : {verification_status}")
        report_lines.append(f"Grounding Status    : {grounding_status}")
    else:
        for item in verified_details:
            c = item["citation"]
            st = item["status"]
            if st == "VERIFIED":
                report_lines.append(f"Citation {c:<10} : {st}")
            else:
                report_lines.append(f"Citation {c:<10} : {st} ({item['reason']})")
        report_lines.append(f"Verification Status : {verification_status}")
        report_lines.append(f"Grounding Status    : {grounding_status}")

    report_str = "\n".join(report_lines)

    return {
        "is_valid": is_valid,
        "verification_status": verification_status,
        "grounding_status": grounding_status,
        "used_citations": used_citations,
        "available_citations": available_citations,
        "unsupported_citations": unsupported_citations,
        "verified_details": verified_details,
        "report": report_str
    }


def format_citation_registry(registry: Dict[str, Dict[str, Any]]) -> str:
    """
    Formats the citation registry block for display and inspection.
    """
    if not registry:
        return "Citation Registry: None"

    lines = ["Citation Registry:"]
    for marker, data in registry.items():
        src = data.get("source", "unknown")
        c_id = data.get("chunk_id", "N/A")
        c_idx = data.get("chunk_index", "N/A")
        sec = data.get("section", "N/A")
        page = data.get("page", "N/A")

        lines.append(f"{marker} → {src}")
        lines.append(f"  Chunk ID   : {c_id}")
        lines.append(f"  Chunk Index: {c_idx}")
        lines.append(f"  Section    : {sec}")
        lines.append(f"  Page       : {page}")

    return "\n".join(lines)


def format_citation_details(
    citation_marker: str,
    registry: Dict[str, Dict[str, Any]]
) -> str:
    """
    Formats the detailed developer verification view for inspecting the exact source
    and original retrieved text behind a specific citation.
    """
    marker = str(citation_marker).strip()
    if not marker.startswith("[") and marker.isdigit():
        marker = f"[{marker}]"

    ver = verify_citation(marker, registry)
    if not ver["is_verified"]:
        return (
            "========================================\n"
            "CITATION DETAILS\n"
            f"{marker}\n"
            f"Status       : INVALID\n"
            f"Reason       : {ver['reason']}\n"
            "========================================"
        )

    data = registry[marker]
    lines = [
        "========================================",
        "CITATION DETAILS",
        marker,
        f"Source Document        : {data.get('source', 'unknown')}",
        f"Chunk ID               : {data.get('chunk_id', 'N/A')}",
        f"Chunk Index            : {data.get('chunk_index', 'N/A')}",
        f"Section                : {data.get('section', 'N/A')}",
        f"Page                   : {data.get('page', 'N/A')}",
        "Original Retrieved Text:",
        data.get("original_text", "").strip(),
        "Verification:",
        "PASS",
        "========================================"
    ]
    return "\n".join(lines)


def format_sample_cited_answer(
    question: str,
    answer: str,
    registry: Dict[str, Dict[str, Any]],
    verification_result: Dict[str, Any]
) -> str:
    """
    Generates standardized format for outputs/sample_cited_answer.txt.
    """
    lines = [
        "========================================",
        "SHIPRULE SAMPLE CITED ANSWER",
        "Question:",
        question.strip(),
        "",
        "RETRIEVED SOURCES"
    ]

    for marker, data in registry.items():
        lines.append(f"{marker} Source: {data.get('source', 'unknown')}")
        lines.append(f"Chunk ID: {data.get('chunk_id', 'N/A')}")
        lines.append(f"Chunk Index: {data.get('chunk_index', 'N/A')}")
        lines.append(f"Section: {data.get('section', 'N/A')}")
        lines.append(f"Page: {data.get('page', 'N/A')}")
        lines.append("Original Text:")
        lines.append(data.get("original_text", "").strip())
        lines.append("")

    lines.append("GENERATED ANSWER")
    lines.append(answer.strip())
    lines.append("")

    lines.append("CITATION MAPPING")
    for marker, data in registry.items():
        src = data.get("source", "unknown")
        c_id = data.get("chunk_id", "N/A")
        c_idx = data.get("chunk_index", "N/A")
        sec = data.get("section") or "General"
        page = data.get("page")
        loc_str = f"{sec}/Page {page}" if page is not None else sec
        lines.append(f"{marker} → {src} → {c_id} (Index {c_idx}) → {loc_str}")
    lines.append("")

    lines.append("SOURCE VERIFICATION")
    for item in verification_result.get("verified_details", []):
        c = item["citation"]
        st = item["status"]
        lines.append(f"{c} {st}")

    lines.append("Overall Verification:")
    lines.append(verification_result.get("verification_status", "PASS"))
    lines.append("========================================")

    return "\n".join(lines)


def export_citation_mapping_json(
    registry: Dict[str, Dict[str, Any]],
    output_path: str
) -> None:
    """
    Exports clean citation-to-source mapping dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
