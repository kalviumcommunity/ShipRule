"""
ShipRule CDLP - Text Extraction & Cleaning Pipeline Module
============================================================
Reusable text-cleaning pipeline that normalizes Unicode, normalizes line endings,
strips page footer patterns (e.g. Page 3 of 12), collapses excess whitespace, and
prepares extracted document text for downstream chunking, embedding, and RAG retrieval.
"""

import re
import unicodedata


def clean(text: str) -> str:
    """
    Cleans and normalizes raw document text for the RAG pipeline.

    Operations:
    1. Unicode NFKC normalization.
    2. Normalize Windows/Mac line endings (\\r\\n, \\r) to \\n.
    3. Remove page footer patterns (e.g., 'Page 3 of 12', 'Page 10 of 25').
    4. Collapse repeated horizontal spaces and tabs into a single space.
    5. Strip spaces at line margins.
    6. Collapse 3 or more consecutive newlines into a maximum of 2.
    7. Strip overall leading/trailing whitespace.
    8. Preserve headings, numbers, punctuation, code blocks, and table content.

    Args:
        text: Raw extracted document string.

    Returns:
        Cleaned, normalized string ready for RAG processing.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    if not text:
        return ""

    # 1. Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Line ending normalization (\r\n and \r to \n)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Remove page footer patterns (e.g., "Page 3 of 12", "Page 10 of 25")
    # Matches "Page <digits> of <digits>" ignoring case
    text = re.sub(r"(?i)\bPage\s+\d+\s+of\s+\d+\b", "", text)

    # 4. Collapse repeated spaces and tabs into a single space on each line
    text = re.sub(r"[^\S\n]+", " ", text)

    # 5. Trim leading/trailing spaces from each line
    text = re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.MULTILINE)

    # 6. Collapse 3 or more consecutive newlines into a maximum of 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 7. Strip leading and trailing whitespace
    return text.strip()


def format_cleaning_summary(source: str, raw_text: str, cleaned_text: str, sample_len: int = 140) -> str:
    """
    Formats a clear BEFORE -> AFTER cleaning demonstration summary for a document.

    Example output line:
    document.pdf: 2450 -> 2180 chars
    """
    raw_len = len(raw_text) if raw_text else 0
    cleaned_len = len(cleaned_text) if cleaned_text else 0

    before_snippet = (raw_text[:sample_len] + "...") if len(raw_text) > sample_len else raw_text
    after_snippet = (cleaned_text[:sample_len] + "...") if len(cleaned_text) > sample_len else cleaned_text

    # Replace newlines in snippet previews for clean output printing
    before_preview = before_snippet.replace("\r", "\\r").replace("\n", "\\n")
    after_preview = after_snippet.replace("\r", "\\r").replace("\n", "\\n")

    lines = [
        f"{source}: {raw_len} -> {cleaned_len} chars",
        f"  BEFORE: {before_preview}",
        f"  AFTER : {after_preview}"
    ]
    return "\n".join(lines)
