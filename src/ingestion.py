"""
ShipRule CDLP - Corpus Preparation & Ingestion Validation Pipeline
===================================================================
Connects document discovery, multi-format loading, text cleaning, chunking,
chunk metadata tagging, reconciliation validation, and report generation
into an automated, resumable ingestion pipeline.
"""

import os
import sys
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple

# Ensure UTF-8 output handling on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.text_cleaner import clean
from src.document_loader import _load_txt, _load_pdf, SUPPORTED_EXTENSIONS
from src.chunker import chunk_document_by_strategy


# ==============================================================================
# 1. FILE DISCOVERY & HASHING UTILITIES
# ==============================================================================

def compute_file_hash(file_path: str) -> str:
    """Computes SHA-256 hash of a file for change detection and resumability."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def discover_files(directory_path: str, recursive: bool = True) -> List[str]:
    """
    Recursively discovers all valid files in the given directory.
    Ignores hidden files (starting with '.') and version control folders.
    """
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return []

    discovered = []
    if recursive:
        for root, dirs, files in os.walk(directory_path):
            # Exclude hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in sorted(files):
                if not f.startswith("."):
                    discovered.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(directory_path)):
            full_p = os.path.join(directory_path, f)
            if os.path.isfile(full_p) and not f.startswith("."):
                discovered.append(full_p)

    return sorted(discovered)


# ==============================================================================
# 2. SINGLE FILE PROCESSOR
# ==============================================================================

def process_document_file(
    file_path: str,
    strategy: str = "paragraph",
    **chunk_kwargs
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """
    Loads, cleans, validates, and chunks an individual document file.

    Returns:
        manifest_entry: Dict with file_path, file_name, document_type, status,
                        character_count, chunk_count, error_message, file_hash
        chunks: List of generated unified chunk dictionaries
        log_message: Formatted single-line progress log string
    """
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_path)
    ext_lower = ext.lower()
    doc_type = ext_lower.lstrip(".") if ext_lower else "unknown"

    manifest_entry = {
        "file_path": file_path,
        "file_name": file_name,
        "document_type": doc_type,
        "status": "FAILED",
        "character_count": 0,
        "chunk_count": 0,
        "error_message": None,
        "file_hash": None
    }

    # 1. Existence check
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        manifest_entry["status"] = "FAILED"
        manifest_entry["error_message"] = "File not found or unreadable"
        log_msg = f"FAILED | {manifest_entry['error_message']}"
        return manifest_entry, [], log_msg

    try:
        manifest_entry["file_hash"] = compute_file_hash(file_path)
    except Exception as e:
        manifest_entry["file_hash"] = None

    # 2. Supported format check
    if ext_lower not in SUPPORTED_EXTENSIONS:
        manifest_entry["status"] = "SKIPPED"
        manifest_entry["error_message"] = f"Unsupported file format '{ext}'"
        log_msg = f"SKIPPED | Unsupported file format '{ext}'"
        return manifest_entry, [], log_msg

    # 3. Read & clean text
    try:
        if ext_lower in {".txt", ".md"}:
            raw_text = _load_txt(file_path)
        elif ext_lower == ".pdf":
            raw_text = _load_pdf(file_path)
        else:
            manifest_entry["status"] = "SKIPPED"
            manifest_entry["error_message"] = f"Unsupported file format '{ext}'"
            log_msg = f"SKIPPED | Unsupported format '{ext}'"
            return manifest_entry, [], log_msg

        cleaned_text = clean(raw_text) if raw_text else ""

        if not cleaned_text or not cleaned_text.strip():
            manifest_entry["status"] = "SKIPPED"
            manifest_entry["error_message"] = "File is empty or contains no extractable text"
            log_msg = f"SKIPPED | File is empty or contains no extractable text"
            return manifest_entry, [], log_msg

        doc_payload = {
            "source": file_name,
            "text": cleaned_text
        }

        # 4. Apply chunking strategy
        chunks = chunk_document_by_strategy(doc_payload, strategy=strategy, **chunk_kwargs)

        if not chunks:
            manifest_entry["status"] = "FAILED"
            manifest_entry["error_message"] = "Chunking strategy produced 0 chunks"
            log_msg = f"FAILED | Chunking produced 0 chunks"
            return manifest_entry, [], log_msg

        manifest_entry["status"] = "SUCCESS"
        manifest_entry["character_count"] = len(cleaned_text)
        manifest_entry["chunk_count"] = len(chunks)
        manifest_entry["error_message"] = None
        log_msg = f"SUCCESS | {len(chunks)} chunks ({len(cleaned_text):,} chars)"
        return manifest_entry, chunks, log_msg

    except Exception as e:
        manifest_entry["status"] = "FAILED"
        manifest_entry["error_message"] = str(e) or "Unable to process document"
        log_msg = f"FAILED | {manifest_entry['error_message']}"
        return manifest_entry, [], log_msg


# ==============================================================================
# 3. VALIDATION & RECONCILIATION
# ==============================================================================

REQUIRED_METADATA_FIELDS = {
    "chunk_id",
    "source",
    "document_type",
    "strategy",
    "chunk_index",
    "character_count",
    "chunk_text"
}


def validate_chunk_metadata(chunks: List[Dict[str, Any]]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates that every generated chunk contains all required metadata fields,
    has non-empty text, correct character counts, positive 1-indexed chunk index,
    and a globally unique chunk ID.
    """
    errors = []
    seen_ids = set()
    invalid_chunk_ids = []

    for i, c in enumerate(chunks):
        c_id = c.get("chunk_id")
        
        # Check required keys
        missing_keys = REQUIRED_METADATA_FIELDS - set(c.keys())
        if missing_keys:
            errors.append(f"Chunk at index {i} missing fields: {missing_keys}")
            if c_id:
                invalid_chunk_ids.append(c_id)
            continue

        if not c_id:
            errors.append(f"Chunk at index {i} has empty chunk_id")
            continue

        # Check uniqueness
        if c_id in seen_ids:
            errors.append(f"Duplicate chunk_id detected: '{c_id}'")
            invalid_chunk_ids.append(c_id)
        seen_ids.add(c_id)

        # Check content integrity
        text = c.get("chunk_text", "")
        if not text or not text.strip():
            errors.append(f"Chunk '{c_id}' contains empty text")
            invalid_chunk_ids.append(c_id)

        char_count = c.get("character_count", 0)
        if char_count != len(text):
            errors.append(f"Chunk '{c_id}' character_count ({char_count}) mismatch with text length ({len(text)})")
            invalid_chunk_ids.append(c_id)

        if c.get("chunk_index", 0) < 1:
            errors.append(f"Chunk '{c_id}' chunk_index must be >= 1")
            invalid_chunk_ids.append(c_id)

    is_valid = len(errors) == 0
    details = {
        "total_chunks_validated": len(chunks),
        "unique_chunk_ids": len(seen_ids),
        "invalid_chunk_count": len(set(invalid_chunk_ids)),
        "invalid_chunk_ids": list(set(invalid_chunk_ids)),
        "error_details": errors[:10]  # sample errors
    }
    return is_valid, invalid_chunk_ids, details


def validate_corpus_reconciliation(
    manifest: List[Dict[str, Any]],
    total_discovered_files: int
) -> Tuple[bool, str, Dict[str, int]]:
    """
    Enforces the fundamental reconciliation check:
    total_discovered_files == successfully_processed + failed + skipped
    Ensures zero documents are silently dropped by the pipeline.
    """
    success_count = sum(1 for m in manifest if m.get("status") == "SUCCESS")
    failure_count = sum(1 for m in manifest if m.get("status") == "FAILED")
    skipped_count = sum(1 for m in manifest if m.get("status") == "SKIPPED")

    total_accounted = success_count + failure_count + skipped_count
    counts = {
        "total_discovered": total_discovered_files,
        "successful": success_count,
        "failed": failure_count,
        "skipped": skipped_count,
        "total_accounted": total_accounted
    }

    if total_accounted != total_discovered_files:
        diff = total_discovered_files - total_accounted
        msg = f"RECONCILIATION FAILED: {diff} file(s) were silently dropped! ({total_discovered_files} discovered != {total_accounted} accounted)"
        return False, msg, counts

    msg = f"{total_discovered_files} files = {success_count} successful + {failure_count} failed + {skipped_count} skipped. No documents were silently dropped."
    return True, msg, counts


# ==============================================================================
# 4. CORPUS INGESTION PIPELINE CLASS & RUNNER
# ==============================================================================

class CorpusIngestionPipeline:
    """
    End-to-end ingestion pipeline with tracking, reconciliation validation,
    failure isolation, resumability, and output generation.
    """

    def __init__(
        self,
        corpus_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        strategy: str = "paragraph",
        resumable: bool = False,
        verbose: bool = True
    ):
        self.corpus_dir = corpus_dir or os.path.join(project_root, "data", "sample_corpus")
        self.output_dir = output_dir or os.path.join(project_root, "outputs")
        self.strategy = strategy
        self.resumable = resumable
        self.verbose = verbose

    def run(self) -> Dict[str, Any]:
        """Executes the complete corpus ingestion pipeline."""
        os.makedirs(self.output_dir, exist_ok=True)
        log_lines = []

        def log_print(msg: str = ""):
            log_lines.append(msg)
            if self.verbose:
                print(msg, flush=True)

        log_print("=================================================")
        log_print("SHIPRULE CORPUS INGESTION PIPELINE")
        log_print("=================================================\n")

        # 1. Discover all files recursively
        discovered_paths = discover_files(self.corpus_dir, recursive=True)
        total_files = len(discovered_paths)

        log_print(f"Discovered {total_files} file(s) in '{self.corpus_dir}'")
        log_print("-" * 49)

        # 2. Resumability check
        previous_manifest_map = {}
        previous_chunks_map = {}
        manifest_file = os.path.join(self.output_dir, "corpus_manifest.json")
        chunks_file = os.path.join(self.output_dir, "processed_chunks.json")

        if self.resumable and os.path.exists(manifest_file) and os.path.exists(chunks_file):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    old_manifest = json.load(f)
                    for item in old_manifest:
                        if item.get("status") == "SUCCESS" and item.get("file_hash"):
                            previous_manifest_map[item["file_path"]] = item

                with open(chunks_file, "r", encoding="utf-8") as f:
                    old_chunks = json.load(f)
                    for c in old_chunks:
                        src = c.get("source")
                        if src:
                            previous_chunks_map.setdefault(src, []).append(c)
                log_print(f"[Resumable Mode] Loaded state for {len(previous_manifest_map)} previously successful file(s).")
            except Exception as e:
                log_print(f"[Resumable Mode] Could not load previous state ({e}). Running full ingestion.")

        manifest: List[Dict[str, Any]] = []
        all_chunks: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        # 3. Process each discovered file with isolation
        for idx, fpath in enumerate(discovered_paths, start=1):
            fname = os.path.basename(fpath)

            # Check if resumable skip applies
            if self.resumable and fpath in previous_manifest_map:
                try:
                    curr_hash = compute_file_hash(fpath)
                    if curr_hash == previous_manifest_map[fpath].get("file_hash"):
                        entry = previous_manifest_map[fpath]
                        manifest.append(entry)
                        reused = previous_chunks_map.get(fname, [])
                        all_chunks.extend(reused)
                        log_print(f"[{idx}/{total_files}] Processing {fname} ... REUSED (Cached) | {len(reused)} chunks")
                        continue
                except Exception:
                    pass  # Recompute if error

            # Process file
            entry, chunks, log_msg = process_document_file(fpath, strategy=self.strategy)
            manifest.append(entry)
            
            if entry["status"] == "SUCCESS":
                all_chunks.extend(chunks)
            elif entry["status"] == "FAILED":
                failures.append(entry)

            log_print(f"[{idx}/{total_files}] Processing {fname} ... {log_msg}")

        log_print("\n" + "=" * 49)

        # 4. Validations
        reconciliation_passed, recon_msg, recon_counts = validate_corpus_reconciliation(
            manifest, total_files
        )
        metadata_passed, invalid_ids, meta_details = validate_chunk_metadata(all_chunks)

        overall_passed = reconciliation_passed and metadata_passed and (recon_counts["successful"] > 0 or total_files == 0)

        # 5. Statistics
        total_success = recon_counts["successful"]
        total_failed = recon_counts["failed"]
        total_skipped = recon_counts["skipped"]
        total_chunks = len(all_chunks)
        avg_chunks = round(total_chunks / total_success, 1) if total_success > 0 else 0.0
        total_chars = sum(m.get("character_count", 0) for m in manifest if m.get("status") == "SUCCESS")

        # 6. Print Report Summary
        log_print("SHIPRULE CORPUS INGESTION REPORT")
        log_print("=================================================")
        log_print(f"Files discovered: {total_files}")
        log_print(f"Successfully processed: {total_success}")
        log_print(f"Failed: {total_failed}")
        log_print(f"Skipped: {total_skipped}")
        log_print(f"Total chunks generated: {total_chunks}")
        log_print(f"Average chunks/document: {avg_chunks}")
        log_print(f"Total characters processed: {total_chars:,}")
        log_print(f"Validation: {'PASSED' if overall_passed else 'FAILED'}\n")
        log_print("Reconciliation Check:")
        log_print(recon_msg)
        log_print(f"Metadata Validation: {'PASSED' if metadata_passed else f'FAILED (Invalid Chunks: {len(invalid_ids)})'}")
        log_print("=================================================\n")

        # 7. Sample Chunk Inspection
        if all_chunks:
            log_print("--- SAMPLE CHUNK INSPECTION ---")
            sample_count = min(2, len(all_chunks))
            for i in range(sample_count):
                sample_chunk = all_chunks[i]
                snippet = sample_chunk["chunk_text"]
                if len(snippet) > 130:
                    snippet = snippet[:130].replace("\n", " ").strip() + "..."
                else:
                    snippet = snippet.replace("\n", " ").strip()
                log_print(f"Chunk ID: {sample_chunk['chunk_id']}")
                log_print(f"Source: {sample_chunk['source']}")
                log_print(f"Type: {sample_chunk['document_type']}")
                log_print(f"Strategy: {sample_chunk['strategy']}")
                log_print(f"Characters: {sample_chunk['character_count']}")
                log_print(f"Preview:\n\"{snippet}\"\n")

        # 8. Save all 5 Output Files
        manifest_out = os.path.join(self.output_dir, "corpus_manifest.json")
        report_out = os.path.join(self.output_dir, "ingestion_report.json")
        failures_out = os.path.join(self.output_dir, "ingestion_failures.json")
        chunks_out = os.path.join(self.output_dir, "processed_chunks.json")
        log_out = os.path.join(self.output_dir, "ingestion_log.txt")

        with open(manifest_out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        with open(chunks_out, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        with open(failures_out, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)

        report_payload = {
            "corpus_directory": self.corpus_dir,
            "chunking_strategy": self.strategy,
            "statistics": {
                "files_discovered": total_files,
                "successfully_processed": total_success,
                "failed": total_failed,
                "skipped": total_skipped,
                "total_chunks_generated": total_chunks,
                "average_chunks_per_document": avg_chunks,
                "total_characters_processed": total_chars
            },
            "validation": {
                "overall_status": "PASSED" if overall_passed else "FAILED",
                "reconciliation_status": "PASSED" if reconciliation_passed else "FAILED",
                "reconciliation_details": recon_msg,
                "metadata_status": "PASSED" if metadata_passed else "FAILED",
                "metadata_details": meta_details
            },
            "output_files": {
                "manifest": "outputs/corpus_manifest.json",
                "report": "outputs/ingestion_report.json",
                "failures": "outputs/ingestion_failures.json",
                "processed_chunks": "outputs/processed_chunks.json",
                "log": "outputs/ingestion_log.txt"
            }
        }

        with open(report_out, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2, ensure_ascii=False)

        with open(log_out, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")

        log_print("[OUTPUT] Ingestion artifacts saved to:")
        log_print(f"  • {manifest_out}")
        log_print(f"  • {report_out}")
        log_print(f"  • {failures_out}")
        log_print(f"  • {chunks_out}")
        log_print(f"  • {log_out}\n")

        return report_payload


def run_ingestion_pipeline(
    corpus_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    strategy: str = "paragraph",
    resumable: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """Convenience functional wrapper for CorpusIngestionPipeline."""
    pipeline = CorpusIngestionPipeline(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        strategy=strategy,
        resumable=resumable,
        verbose=verbose
    )
    return pipeline.run()


if __name__ == "__main__":
    run_ingestion_pipeline()
