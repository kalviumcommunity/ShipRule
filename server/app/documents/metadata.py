"""
Chunk Metadata & Source Tracking Module (KDU 3.22)
===================================================
Provides standardized document chunking, metadata tagging, consistent schema validation,
and source traceability for the Customs Duty & Documentation Lookup Platform (CDLP).

Key Capabilities:
1. Store source identifier (filename, doc ID, source URL).
2. Attach detailed metadata (section, page, chunk_index, char_start/char_end positions).
3. Enforce a consistent metadata dictionary schema across the entire corpus.
4. Trace any retrieved chunk back to its exact source document, section, and position.
"""

from typing import List, Dict, Any, Tuple, Optional


REQUIRED_METADATA_KEYS = [
    "source",
    "section",
    "chunk_index",
    "char_start",
    "char_end",
    "doc_type"
]

OPTIONAL_METADATA_KEYS = [
    "page",
    "country",
    "hs_code",
    "last_confirmed_date",
    "source_agency",
    "source_url"
]


def create_metadata_dict(
    source: str,
    section: str = "General",
    chunk_index: int = 0,
    char_start: int = 0,
    char_end: int = 0,
    doc_type: str = "PRD",
    page: int = 1,
    country: str = "N/A",
    hs_code: str = "N/A",
    last_confirmed_date: str = "N/A",
    source_agency: str = "N/A",
    source_url: str = "N/A"
) -> Dict[str, Any]:
    """
    Constructs a consistent metadata dictionary for a chunk across the corpus.
    Guarantees that every chunk has the exact same fields in its metadata dictionary.
    """
    return {
        "source": str(source),
        "section": str(section),
        "chunk_index": int(chunk_index),
        "char_start": int(char_start),
        "char_end": int(char_end),
        "doc_type": str(doc_type),
        "page": int(page),
        "country": str(country),
        "hs_code": str(hs_code),
        "last_confirmed_date": str(last_confirmed_date),
        "source_agency": str(source_agency),
        "source_url": str(source_url)
    }


def validate_chunk_metadata_schema(chunk_item: Dict[str, Any]) -> bool:
    """
    Validates that a chunk object strictly contains 'text' and a 'metadata' dict
    with all required schema fields present.
    """
    if not isinstance(chunk_item, dict):
        return False
    if "text" not in chunk_item or "metadata" not in chunk_item:
        return False
    metadata = chunk_item["metadata"]
    if not isinstance(metadata, dict):
        return False
    for req_key in REQUIRED_METADATA_KEYS:
        if req_key not in metadata:
            return False
    return True


def tag_chunks(
    source: str,
    raw_chunks: List[Tuple[str, int, int]],
    section: str = "General",
    doc_type: str = "PRD",
    page: int = 1,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Attaches metadata to a list of raw text chunks (text, char_start, char_end).

    Args:
        source: Document identifier (e.g., 'cdlp_prd_v1.0.md')
        raw_chunks: List of tuples (chunk_text, char_start, char_end)
        section: Document section title
        doc_type: Category of document ('PRD', 'RegulationData', etc.)
        page: Page number if applicable
        extra_metadata: Optional dict for country, hs_code, source_url, source_agency, etc.

    Returns:
        List of chunk dicts in consistent shape:
        [{ 'text': '...', 'metadata': { ... } }]
    """
    extra = extra_metadata or {}
    tagged = []
    for idx, (text_content, char_start, char_end) in enumerate(raw_chunks):
        meta = create_metadata_dict(
            source=source,
            section=section,
            chunk_index=idx,
            char_start=char_start,
            char_end=char_end,
            doc_type=doc_type,
            page=page,
            country=extra.get("country", "N/A"),
            hs_code=extra.get("hs_code", "N/A"),
            last_confirmed_date=extra.get("last_confirmed_date", "N/A"),
            source_agency=extra.get("source_agency", "N/A"),
            source_url=extra.get("source_url", "N/A")
        )
        tagged.append({
            "text": text_content,
            "metadata": meta
        })
    return tagged


def chunk_document_with_metadata(
    text: str,
    source: str,
    section: str = "General",
    doc_type: str = "PRD",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    page: int = 1,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Splits text into character chunks with overlap, tracking start/end positions,
    and returns chunk objects with consistent metadata attached.
    """
    if not text or not text.strip():
        return []

    raw_chunks: List[Tuple[str, int, int]] = []
    step = max(1, chunk_size - chunk_overlap)
    pos = 0
    text_length = len(text)

    while pos < text_length:
        end_pos = min(pos + chunk_size, text_length)
        chunk_text = text[pos:end_pos]
        raw_chunks.append((chunk_text, pos, end_pos))
        if end_pos >= text_length:
            break
        pos += step

    return tag_chunks(
        source=source,
        raw_chunks=raw_chunks,
        section=section,
        doc_type=doc_type,
        page=page,
        extra_metadata=extra_metadata
    )


def trace_chunk_source(chunk_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traces any retrieved chunk back to its exact source document, section, and position.

    Args:
        chunk_item: Either a chunk dict {'text': ..., 'metadata': {...}} or metadata dict.

    Returns:
        Structured traceback info with formatted citation string.
    """
    if "metadata" in chunk_item:
        meta = chunk_item["metadata"]
        text_snippet = chunk_item.get("text", "")
    else:
        meta = chunk_item
        text_snippet = ""

    source = meta.get("source", "Unknown Source")
    section = meta.get("section", "General")
    chunk_index = meta.get("chunk_index", 0)
    char_start = meta.get("char_start", 0)
    char_end = meta.get("char_end", 0)
    doc_type = meta.get("doc_type", "Document")
    page = meta.get("page", 1)
    agency = meta.get("source_agency", "N/A")
    url = meta.get("source_url", "N/A")
    country = meta.get("country", "N/A")
    hs_code = meta.get("hs_code", "N/A")

    formatted_citation = (
        f"[Source: {source} | Section: {section} | Chunk #{chunk_index} "
        f"(Page {page}, pos: {char_start}-{char_end})"
    )
    if country != "N/A":
        formatted_citation += f" | Country: {country}"
    if hs_code != "N/A":
        formatted_citation += f" | HS: {hs_code}"
    if agency != "N/A":
        formatted_citation += f" | Agency: {agency}"
    if url != "N/A":
        formatted_citation += f" | URL: {url}"
    formatted_citation += "]"

    return {
        "source": source,
        "section": section,
        "chunk_index": chunk_index,
        "char_start": char_start,
        "char_end": char_end,
        "doc_type": doc_type,
        "page": page,
        "country": country,
        "hs_code": hs_code,
        "source_agency": agency,
        "source_url": url,
        "snippet": text_snippet[:100] + ("..." if len(text_snippet) > 100 else ""),
        "formatted_citation": formatted_citation
    }
