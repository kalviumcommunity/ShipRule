"""
ShipRule CDLP - Loaders Module
"""

from loaders.document_loader import (
    load_document,
    load_documents,
    load_directory,
    SUPPORTED_EXTENSIONS,
)

__all__ = [
    "load_document",
    "load_documents",
    "load_directory",
    "SUPPORTED_EXTENSIONS",
]
