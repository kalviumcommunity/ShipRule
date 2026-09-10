"""
ShipRule CDLP - Single Backend Entry Point
==========================================
Main FastAPI application entry point for the ShipRule RAG Service.
Registers health, query, and document routers, configures CORS, and handles startup.
"""

import sys
import os
from pathlib import Path

# Ensure server root is in sys.path
SERVER_ROOT = Path(__file__).resolve().parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.api.routes.documents import router as documents_router
from app.api.routes.auth import router as auth_router
from app.api.routes.admin import router as admin_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API exposing Customs Duty & Shipping Documentation RAG pipeline with grounded answers and source citations.",
    version=settings.VERSION
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CLIENT_URL, "http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(health_router)
app.include_router(query_router)
app.include_router(documents_router)


if __name__ == "__main__":
    print(f"[INFO] Starting {settings.PROJECT_NAME} backend on http://127.0.0.1:8000...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
