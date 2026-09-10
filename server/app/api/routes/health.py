"""
ShipRule CDLP - System & Health Routes
======================================
Provides health check and system status endpoints.
"""

from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/", tags=["System"])
def root():
    """Root endpoint returning API metadata and health status."""
    return {
        "service": settings.PROJECT_NAME,
        "status": "online",
        "version": settings.VERSION,
        "config": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "chat_model": settings.CHAT_MODEL,
            "collection_name": settings.COLLECTION_NAME
        }
    }


@router.get("/health", tags=["System"])
def health_check():
    """Health check endpoint to verify backend API service operational state."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "environment": {
            "embedding_model_configured": bool(settings.EMBEDDING_MODEL),
            "chat_model_configured": bool(settings.CHAT_MODEL),
            "vector_db_path": settings.VECTOR_DB_URL,
            "collection_name": settings.COLLECTION_NAME,
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB
        }
    }
