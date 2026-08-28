"""
ShipRule CDLP - Document Intake & Loader Module
================================================
Robust multi-format document loader supporting plain-text conversion (PDF, TXT, MD),
source identity preservation, and graceful handling of missing, corrupt, or unsupported files.
"""

import os
import sys
import re
import logging
from typing import Dict, List, Optional, Any

# Ensure UTF-8 output handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md"}

logger = logging.getLogger(__name__)


def _normalize_sample_text(text: str, max_chars: int = 140) -> str:
    """Normalizes excessive whitespace and returns a clean preview snippet."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) > max_chars:
        return normalized[:max_chars].rstrip() + "..."
    return normalized


def _load_txt(file_path: str) -> str:
    """Loads plain text or markdown file with UTF-8 encoding and fallback."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def _load_pdf(file_path: str) -> str:
    """Extracts text from PDF file using pypdf."""
    import pypdf

    reader = pypdf.PdfReader(file_path)
    pages_text = []
    for page_idx, page in enumerate(reader.pages):
        page_content = page.extract_text()
        if page_content:
            pages_text.append(page_content.strip())

    extracted = "\n\n".join(pages_text).strip()
    if not extracted:
        raise ValueError("PDF document contains no extractable text.")
    return extracted


def load_document(file_path: str, verbose: bool = False) -> Optional[Dict[str, str]]:
    """
    Loads a single document file into a common plain-text representation.
    Handles missing, corrupt, and unsupported files gracefully without crashing.

    Returns:
        Dict with keys {"source": str, "text": str} on success, or None on failure.
    """
    filename = os.path.basename(file_path) if file_path else "unknown"

    # 1. Missing file check
    if not file_path or not os.path.exists(file_path):
        warning_msg = f"[WARNING] File not found: {filename} - skipping."
        print(warning_msg, flush=True)
        return None

    # 2. Unsupported format check
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        warning_msg = f"[WARNING] Unsupported file format: {filename} - skipping."
        print(warning_msg, flush=True)
        return None

    # 3. Read & Extract text with error handling for corrupt / unreadable files
    try:
        if ext in {".txt", ".md"}:
            text = _load_txt(file_path)
        elif ext == ".pdf":
            text = _load_pdf(file_path)
        else:
            warning_msg = f"[WARNING] Unsupported file format: {filename} - skipping."
            print(warning_msg, flush=True)
            return None

        if not text or not text.strip():
            warning_msg = f"[WARNING] Could not read: {filename} - skipping.\nReason: File is empty or contains no extractable text."
            print(warning_msg, flush=True)
            return None

        doc_payload = {
            "source": filename,
            "text": text
        }

        if verbose:
            sample_snippet = _normalize_sample_text(text)
            print(f"[LOADED]", flush=True)
            print(f"Source: {filename}", flush=True)
            print(f"Text Length: {len(text)} characters", flush=True)
            print(f"Sample: {sample_snippet}\n", flush=True)
            print("--------------------------------------------------\n", flush=True)

        return doc_payload

    except Exception as e:
        warning_msg = f"[WARNING] Could not read: {filename} - skipping.\nReason: {str(e)}"
        print(warning_msg, flush=True)
        return None


def load_documents(file_paths: List[str], verbose: bool = True) -> List[Dict[str, str]]:
    """
    Loads a list of file paths into a common plain-text representation.
    Prints document intake results and an intake summary.

    Returns:
        List of document dicts: [{"source": "...", "text": "..."}, ...]
    """
    if verbose:
        print("==================================================", flush=True)
        print("SHIPRULE DOCUMENT LOADER", flush=True)
        print("==================================================\n", flush=True)

    loaded_documents: List[Dict[str, str]] = []
    total_attempted = len(file_paths)
    skipped_count = 0

    for path in file_paths:
        doc = load_document(path, verbose=verbose)
        if doc is not None:
            loaded_documents.append(doc)
        else:
            skipped_count += 1
            if verbose:
                print("", flush=True)

    if verbose:
        print("==================================================", flush=True)
        print("INTAKE SUMMARY", flush=True)
        print("==================================================", flush=True)
        print(f"Files attempted: {total_attempted}", flush=True)
        print(f"Files successfully loaded: {len(loaded_documents)}", flush=True)
        print(f"Files skipped: {skipped_count}", flush=True)
        print("==================================================", flush=True)

    return loaded_documents


def load_directory(directory_path: str, verbose: bool = True) -> List[Dict[str, str]]:
    """
    Scans a directory and loads all supported files.
    """
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        print(f"[WARNING] Directory not found: {directory_path} - skipping.", flush=True)
        return []

    file_paths = []
    for root, _, files in os.walk(directory_path):
        for f in sorted(files):
            if not f.startswith("."):
                file_paths.append(os.path.join(root, f))

    return load_documents(file_paths, verbose=verbose)


def run_intake_demonstration():
    """Runs the document intake demonstration against sample corpus and test edge cases."""
    corpus_dir = os.path.join(project_root, "data", "sample_corpus")

    # Order sample files explicitly for consistent demonstration
    sample_names = [
        "shipping_rules.txt",
        "customs_requirements.txt",
        "international_shipping_guide.pdf"
    ]
    test_files = [os.path.join(corpus_dir, name) for name in sample_names if os.path.exists(os.path.join(corpus_dir, name))]

    # Add missing and unsupported files to demonstrate robust error handling
    test_files.extend([
        os.path.join(corpus_dir, "missing_document.pdf"),
        os.path.join(corpus_dir, "unsupported_document.docx"),
    ])

    dummy_unsupported = os.path.join(corpus_dir, "unsupported_document.docx")
    dummy_created = False
    if not os.path.exists(dummy_unsupported):
        with open(dummy_unsupported, "w", encoding="utf-8") as f:
            f.write("Dummy Word Document binary content")
        dummy_created = True

    try:
        load_documents(test_files, verbose=True)
    finally:
        if dummy_created and os.path.exists(dummy_unsupported):
            try:
                os.remove(dummy_unsupported)
            except Exception:
                pass


if __name__ == "__main__":
    run_intake_demonstration()
