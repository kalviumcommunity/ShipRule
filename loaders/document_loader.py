"""
ShipRule CDLP - Document Intake & Loader Module
================================================
Loaders package adapter for src.document_loader.
"""

from src.text_cleaner import clean
from src.document_loader import (
    load_document,
    load_documents,
    load_directory,
    run_intake_demonstration,
    SUPPORTED_EXTENSIONS,
    _normalize_sample_text,
    _load_txt,
    _load_pdf,
)

if __name__ == "__main__":
    run_intake_demonstration()
