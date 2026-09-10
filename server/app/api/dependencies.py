"""
ShipRule CDLP - API Dependencies & Authentication
=================================================
Provides authentication, JWT token verification, role-based access control (RBAC),
and dependency injection helpers for FastAPI routes.
"""

import time
from typing import Optional, Dict, Any
import jwt
from fastapi import Depends, HTTPException, Header, status as http_status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings
from app.rag.retrieval import load_indexed_vector_collection

security = HTTPBearer(auto_error=False)


def create_access_token(data: Dict[str, Any], expires_in_seconds: int = 86400) -> str:
    """Generates a signed JWT access token containing user payload and expiration."""
    payload = data.copy()
    payload["exp"] = int(time.time()) + expires_in_seconds
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a signed JWT access token."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return payload
    except Exception:
        return None


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    """
    Dependency helper that validates the user JWT token or X-Admin-Key header.
    Returns user payload dictionary containing role and email.
    """
    # Allow X-Admin-Key fallback header for administrative tests / scripts
    if x_admin_key and x_admin_key == settings.ADMIN_ACCESS_KEY:
        return {"role": "admin", "email": "admin@shiprule.local"}

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required"
        )

    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token"
        )

    return payload


def require_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    """
    Role-based access control dependency requiring Master Admin privilege.
    Validates token role == 'admin' or valid X-Admin-Key header.
    """
    if x_admin_key and x_admin_key == settings.ADMIN_ACCESS_KEY:
        return {"role": "admin", "email": "admin@shiprule.local"}

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required"
        )

    payload = decode_access_token(credentials.credentials)
    if not payload or payload.get("role") != "admin":
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Master Admin access required"
        )

    return payload


def get_vector_collection():
    """Dependency helper to load and provide vector store collection."""
    return load_indexed_vector_collection()
