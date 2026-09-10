"""
ShipRule CDLP - API Package Re-exports
"""

from main import app
from app.core.config import settings
from app.api.routes.query import QueryRequest, QueryResponse, Source, query_rag
from app.api.routes.documents import (
    DocumentUploadResponse,
    IndexingSummary,
    process_uploaded_document,
    upload_document,
    MAX_UPLOAD_SIZE_MB,
)
from app.documents.loader import token_chunks, _load_pdf
from app.documents.cleaner import clean
from app.rag.embeddings import create_embedding_client, generate_embedding
from app.rag.pipeline import answer_query

SUPPORTED_EXTENSIONS = settings.SUPPORTED_EXTENSIONS

__all__ = [
    "app",
    "QueryRequest",
    "QueryResponse",
    "Source",
    "DocumentUploadResponse",
    "IndexingSummary",
    "process_uploaded_document",
    "upload_document",
    "_load_pdf",
    "clean",
    "token_chunks",
    "create_embedding_client",
    "generate_embedding",
    "answer_query",
    "query_rag",
    "MAX_UPLOAD_SIZE_MB",
    "SUPPORTED_EXTENSIONS",
]
