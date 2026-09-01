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

from src.text_cleaner import clean
import tiktoken
from src.token_counter import get_tokenizer

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
    """Extracts text from PDF file using pypdf, including page markers."""
    import pypdf

    reader = pypdf.PdfReader(file_path)
    pages_text = []
    for page_idx, page in enumerate(reader.pages):
        page_content = page.extract_text()
        if page_content and page_content.strip():
            pages_text.append(f"[Page {page_idx + 1}]\n{page_content.strip()}")

    extracted = "\n\n".join(pages_text).strip()
    if not extracted:
        raise ValueError("PDF document contains no extractable text.")
    return extracted


def token_chunks(
    text: str,
    size: int = 400,
    overlap: int = 60,
    encoding_name: str = "cl100k_base"
) -> List[Dict[str, Any]]:
    """
    Splits input text into token-based overlapping chunks using tiktoken.

    Args:
        text: Input string to chunk.
        size: Maximum token count per chunk (default 400).
        overlap: Token overlap count between adjacent chunks (default 60).
        encoding_name: Name of the tiktoken encoding (default 'cl100k_base').

    Returns:
        List of dictionaries with keys: chunk_id, text, token_count, start_token, end_token, overlap.

    Raises:
        ValueError: If size <= 0, overlap < 0, or overlap >= size.
    """
    if size <= 0:
        raise ValueError(f"Chunk size must be greater than 0, got {size}")
    if overlap < 0:
        raise ValueError(f"Overlap cannot be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(f"Overlap ({overlap}) must be strictly less than chunk size ({size})")

    if not text or not text.strip():
        return []

    try:
        enc = get_tokenizer(encoding_name)
    except Exception:
        enc = None
    if enc is None:
        enc = tiktoken.get_encoding(encoding_name)

    token_ids = enc.encode(text)
    total_tokens = len(token_ids)

    if total_tokens == 0:
        return []

    step = size - overlap
    chunks = []
    chunk_id = 1
    start = 0

    while start < total_tokens:
        end = min(start + size, total_tokens)
        chunk_token_ids = token_ids[start:end]
        chunk_text = enc.decode(chunk_token_ids)

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "token_count": len(chunk_token_ids),
            "start_token": start,
            "end_token": end,
            "overlap": overlap
        })

        if end == total_tokens:
            break

        start += step
        chunk_id += 1

    return chunks


def chunk_document(doc: Dict[str, str], max_chunk_size: int = 400, overlap: int = 60) -> List[Dict[str, Any]]:
    """
    Splits a document's cleaned text into structured, token-aware overlapping chunks.
    Preserves source name, detected page number, chunk index, and token metadata.
    """
    if not doc or not doc.get("text"):
        return []

    source = doc.get("source", "unknown")
    text = doc["text"]

    raw_token_chunks = token_chunks(text, size=max_chunk_size, overlap=overlap)

    chunks = []
    for idx, tc in enumerate(raw_token_chunks, start=1):
        chunk_text = tc["text"]
        page_num = "1"
        page_match = re.search(r"\[Page\s+(\d+)\]", chunk_text)
        if page_match:
            page_num = page_match.group(1)
            chunk_text = re.sub(r"\[Page\s+\d+\]\n?", "", chunk_text).strip()

        chunks.append({
            "id": f"{source}_chunk_{idx}",
            "text": chunk_text,
            "metadata": {
                "source": source,
                "page": str(page_num),
                "chunk_id": idx,
                "token_count": tc["token_count"],
                "start_token": tc["start_token"],
                "end_token": tc["end_token"],
                "overlap": tc["overlap"]
            }
        })

    return chunks


def chunk_documents(docs: List[Dict[str, str]], max_chunk_size: int = 400, overlap: int = 60) -> List[Dict[str, Any]]:
    """Chunks a list of loaded document dicts using token-aware chunking."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, max_chunk_size=max_chunk_size, overlap=overlap))
    return all_chunks



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

        raw_text = text or ""
        cleaned_text = clean(raw_text)

        if not cleaned_text:
            warning_msg = f"[WARNING] Could not read: {filename} - skipping.\nReason: File is empty or contains no extractable text."
            print(warning_msg, flush=True)
            return None

        doc_payload = {
            "source": filename,
            "text": cleaned_text
        }

        if verbose:
            sample_before = _normalize_sample_text(raw_text)
            sample_after = _normalize_sample_text(cleaned_text)
            print(f"[LOADED & CLEANED]", flush=True)
            print(f"Source: {filename}", flush=True)
            print(f"{filename}: {len(raw_text)} -> {len(cleaned_text)} chars", flush=True)
            print(f"BEFORE Sample: {sample_before}", flush=True)
            print(f"AFTER Sample : {sample_after}\n", flush=True)
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

    import io

    # Intercept output to write to outputs/document_intake_results.txt
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "document_intake_results.txt")

    buffer = io.StringIO()

    class DualWriter:
        def __init__(self, original_stdout, string_buffer):
            self.stdout = original_stdout
            self.buffer = string_buffer

        def write(self, s):
            self.stdout.write(s)
            self.buffer.write(s)

        def flush(self):
            self.stdout.flush()
            self.buffer.flush()

    dual_writer = DualWriter(sys.stdout, buffer)
    old_stdout = sys.stdout
    sys.stdout = dual_writer

    try:
        load_documents(test_files, verbose=True)
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(buffer.getvalue())
    finally:
        sys.stdout = old_stdout
        if dummy_created and os.path.exists(dummy_unsupported):
            try:
                os.remove(dummy_unsupported)
            except Exception:
                pass



if __name__ == "__main__":
    run_intake_demonstration()
