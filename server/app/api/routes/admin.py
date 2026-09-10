"""
ShipRule CDLP - Admin Panel & System Control API Routes
========================================================
Provides protected endpoints for document deletion, RAG settings configuration (Top-K),
and system status management for Master Admin (/master).
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status as http_status
from pydantic import BaseModel, Field

from app.core.config import settings, UPLOADS_DIR, OUTPUTS_DIR
from app.core.logging import logger
from app.api.dependencies import require_admin

router = APIRouter(prefix="/admin", tags=["Master Admin"])


class AdminSettingsUpdate(BaseModel):
    """Payload model for updating Master Admin settings."""
    default_top_k: Optional[int] = Field(None, ge=1, le=10, description="Configured Top-K retrieval parameter")


@router.get("/settings")
def get_admin_settings(admin: dict = Depends(require_admin)):
    """Returns current Master Admin configuration settings."""
    return {
        "default_top_k": settings.DEFAULT_TOP_K,
        "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chat_model": settings.CHAT_MODEL,
        "collection_name": settings.COLLECTION_NAME
    }


@router.post("/settings")
def update_admin_settings(payload: AdminSettingsUpdate, admin: dict = Depends(require_admin)):
    """Updates system-wide RAG configuration parameters such as default Top-K."""
    if payload.default_top_k is not None:
        settings.DEFAULT_TOP_K = payload.default_top_k
        logger.info(f"Admin updated DEFAULT_TOP_K to {settings.DEFAULT_TOP_K}")

    return {
        "status": "updated",
        "settings": {
            "default_top_k": settings.DEFAULT_TOP_K,
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB
        }
    }


@router.delete("/documents/{stored_filename}")
def delete_document(stored_filename: str, admin: dict = Depends(require_admin)):
    """
    Deletes an uploaded document file from uploads/ and removes its chunks from the vector index.
    """
    # Sanitize filename
    stored_path = (UPLOADS_DIR / Path(stored_filename).name).resolve()
    if not str(stored_path).startswith(str(UPLOADS_DIR.resolve())):
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename or path traversal attempt"
        )

    deleted_file = False
    if stored_path.exists():
        try:
            stored_path.unlink()
            deleted_file = True
            logger.info(f"Deleted uploaded document file: {stored_filename}")
        except Exception as exc:
            logger.error(f"Failed to delete file {stored_filename}: {exc}")
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document file"
            )

    # Clean from outputs/embedded_chunks.json
    embedded_file = OUTPUTS_DIR / "embedded_chunks.json"
    removed_chunks = 0
    if embedded_file.exists():
        try:
            with open(embedded_file, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
                if isinstance(existing_chunks, list):
                    filtered_chunks = []
                    rel_document_path = f"uploads/{stored_filename}"
                    for chunk in existing_chunks:
                        if chunk.get("source") == rel_document_path or chunk.get("filename") == stored_filename:
                            removed_chunks += 1
                        else:
                            filtered_chunks.append(chunk)
                    with open(embedded_file, "w", encoding="utf-8") as f_out:
                        json.dump(filtered_chunks, f_out, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Error removing chunks for {stored_filename}: {exc}")

    if not deleted_file and removed_chunks == 0:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    return {
        "status": "deleted",
        "stored_filename": stored_filename,
        "removed_chunks": removed_chunks
    }
