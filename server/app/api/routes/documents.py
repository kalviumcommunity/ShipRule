"""
ShipRule CDLP - Document Management API Routes
===============================================
Provides document upload, ingestion, cleaning, chunking, embedding, indexing, and listing endpoints.
"""

import os
import re
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, status as http_status
from pydantic import BaseModel, Field

from app.core.config import settings, SERVER_DIR, UPLOADS_DIR, OUTPUTS_DIR
from app.core.logging import logger
from app.documents.loader import token_chunks, _load_pdf
from app.documents.cleaner import clean
from app.rag.embeddings import create_embedding_client, generate_embedding

MAX_UPLOAD_SIZE_MB = settings.MAX_UPLOAD_SIZE_MB

router = APIRouter()


class IndexingSummary(BaseModel):
    """Structured indexing summary for an uploaded document."""
    document: str = Field(..., description="Stored document path relative to uploads directory.")
    chunks: int = Field(..., description="Number of text chunks generated from the document.")
    indexed: int = Field(..., description="Number of vector embeddings successfully indexed.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "document": "uploads/new-policy.md",
                "chunks": 12,
                "indexed": 12
            }
        }
    }


class DocumentUploadResponse(BaseModel):
    """Response model returned following successful document upload and indexing."""
    status: str = Field(default="indexed", description="Indexing status result.")
    filename: str = Field(..., description="Original name of the uploaded document file.")
    summary: IndexingSummary = Field(..., description="Detailed indexing metrics summary.")


class DocumentItem(BaseModel):
    """Document info model for document listing."""
    filename: str
    stored_filename: str
    size_bytes: int
    created_at: str
    document_type: str


def process_uploaded_document(file_path: Path, original_filename: str) -> Dict[str, Any]:
    """
    Extracts text, cleans, chunks, tags metadata, generates embeddings,
    and updates vector storage for an uploaded document.
    """
    ext = file_path.suffix.lower()

    # 1. Load raw text
    try:
        if ext in {".txt", ".md"}:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read().strip()
        elif ext == ".pdf":
            raw_text = _load_pdf(str(file_path))
        else:
            raise ValueError(f"Unsupported file extension '{ext}'")
    except Exception as e:
        if "contains no extractable text" in str(e).lower() or isinstance(e, ValueError):
            raise ValueError("Document contains no extractable text")
        raise RuntimeError(f"Failed to read file: {e}")

    if not raw_text or not raw_text.strip():
        raise ValueError("Document contains no extractable text")

    # 2. Clean text
    cleaned_text = clean(raw_text)
    if not cleaned_text or not cleaned_text.strip():
        raise ValueError("Document contains no extractable text")

    # 3. Chunk text
    raw_token_chunks = token_chunks(cleaned_text, size=400, overlap=60)
    if not raw_token_chunks:
        raise ValueError("Document contains no extractable text")

    # 4. Attach metadata
    rel_document_path = f"uploads/{file_path.name}"
    tagged_chunks = []
    for idx, tc in enumerate(raw_token_chunks, start=1):
        chunk_text_item = tc["text"]
        page_num = "1"
        page_match = re.search(r"\[Page\s+(\d+)\]", chunk_text_item)
        if page_match:
            page_num = page_match.group(1)
            chunk_text_item = re.sub(r"\[Page\s+\d+\]\n?", "", chunk_text_item).strip()

        chunk_id = f"{file_path.name}_chunk_{idx}"
        emb_id = f"emb_{file_path.name}_{idx:03d}"

        tagged_chunks.append({
            "id": chunk_id,
            "chunk_id": chunk_id,
            "embedding_id": emb_id,
            "source": rel_document_path,
            "filename": original_filename,
            "chunk_index": idx,
            "document_type": ext.lstrip("."),
            "strategy": "token",
            "page": str(page_num),
            "chunk_text": chunk_text_item,
            "character_count": len(chunk_text_item),
            "token_count": tc.get("token_count", 0),
            "metadata": {
                "source": rel_document_path,
                "filename": original_filename,
                "chunk_index": idx,
                "page": str(page_num),
                "document_type": ext.lstrip("."),
                "chunk_id": chunk_id,
                "strategy": "token"
            }
        })

    # 5. Generate embeddings
    client = create_embedding_client()
    embedded_chunks = []
    for item in tagged_chunks:
        vec = generate_embedding(client, item["chunk_text"], model=settings.EMBEDDING_MODEL)
        item["embedding"] = vec
        item["vector"] = vec
        item["vector_dimension"] = len(vec)
        item["embedding_model"] = settings.EMBEDDING_MODEL
        embedded_chunks.append(item)

    # 6. Index into outputs/embedded_chunks.json
    embedded_chunks_file = OUTPUTS_DIR / "embedded_chunks.json"

    existing_chunks = []
    if embedded_chunks_file.exists():
        try:
            with open(embedded_chunks_file, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
                if not isinstance(existing_chunks, list):
                    existing_chunks = []
        except Exception:
            existing_chunks = []

    new_ids = {c["id"] for c in embedded_chunks}
    updated_list = [c for c in existing_chunks if c.get("id") not in new_ids]
    updated_list.extend(embedded_chunks)

    with open(embedded_chunks_file, "w", encoding="utf-8") as f:
        json.dump(updated_list, f, indent=2, ensure_ascii=False)

    return {
        "document": rel_document_path,
        "chunks": len(tagged_chunks),
        "indexed": len(embedded_chunks)
    }


@router.get("/documents", response_model=List[DocumentItem], tags=["Document Management"])
def list_documents():
    """Returns list of uploaded documents in the uploads directory."""
    documents = []
    if UPLOADS_DIR.exists():
        for file_path in UPLOADS_DIR.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in settings.SUPPORTED_EXTENSIONS:
                stat = file_path.stat()
                filename = file_path.name
                parts = file_path.name.split("_", 2)
                if len(parts) >= 3:
                    filename = parts[2]
                
                documents.append(DocumentItem(
                    filename=filename,
                    stored_filename=file_path.name,
                    size_bytes=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    document_type=file_path.suffix.lstrip(".")
                ))
    return sorted(documents, key=lambda x: x.created_at, reverse=True)


from app.api.dependencies import require_admin

@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=http_status.HTTP_200_OK,
    tags=["Document Management"]
)
async def upload_document(
    file: UploadFile = File(...),
    admin: dict = Depends(require_admin)
):
    """
    Uploads, validates, cleans, chunks, embeds, and indexes a .txt, .md, or .pdf document.
    Makes the uploaded document immediately searchable through the /query endpoint.
    """
    logger.info(f"Received document upload request for filename: {getattr(file, 'filename', None)}")

    if not file or not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Missing or invalid filename"
        )

    raw_filename = file.filename.strip()

    ext = Path(raw_filename).suffix.lower()
    if ext not in settings.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=http_status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type"
        )

    original_filename = Path(raw_filename).name
    if not original_filename or original_filename in (".", ".."):
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Missing or invalid filename"
        )

    try:
        content = await file.read()
    except Exception as exc:
        logger.error(f"Failed to read upload file stream: {exc}")
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file"
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Document contains no extractable text"
        )

    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=http_status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed upload size of {MAX_UPLOAD_SIZE_MB}MB"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    sanitized_basename = re.sub(r"[^\w\.-]", "_", original_filename)
    stored_filename = f"{timestamp}_{unique_id}_{sanitized_basename}"
    stored_filepath = (UPLOADS_DIR / stored_filename).resolve()

    if not str(stored_filepath).startswith(str(UPLOADS_DIR.resolve())):
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Unsafe filename or path traversal attempt"
        )

    try:
        stored_filepath.write_bytes(content)
    except Exception as exc:
        logger.error(f"Failed to write stored upload file: {exc}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document indexing failed"
        )

    try:
        summary_result = process_uploaded_document(stored_filepath, original_filename)
    except ValueError as val_err:
        logger.warning(f"Document extraction validation error: {val_err}")
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Document processing or indexing failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document indexing failed"
        )

    return DocumentUploadResponse(
        status="indexed",
        filename=original_filename,
        summary=IndexingSummary(
            document=summary_result["document"],
            chunks=summary_result["chunks"],
            indexed=summary_result["indexed"]
        )
    )
