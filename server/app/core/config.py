"""
ShipRule CDLP - Core Configuration Settings
============================================
Centralizes environment configuration, API keys, path definitions,
and model settings for the ShipRule RAG service.
Reads all settings directly from the server .env configuration file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
SERVER_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = SERVER_DIR / "data"
UPLOADS_DIR = SERVER_DIR / "uploads"
OUTPUTS_DIR = SERVER_DIR / "outputs"

# Create directories if they do not exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env
load_dotenv(dotenv_path=SERVER_DIR / ".env", override=True)

class Settings:
    """Application settings and configuration parameters imported from .env."""
    PROJECT_NAME: str = "ShipRule RAG Service API"
    VERSION: str = "1.0.0"
    
    # API & Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY", "")
    
    # Model Configurations
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "groq/compound-mini")
    
    # Vector DB Configuration
    VECTOR_DB_URL: str = os.getenv("VECTOR_DB_URL", str(DATA_DIR / "vector_db"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_chunks")
    
    # Upload & Security Limits
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    SUPPORTED_EXTENSIONS: set = {".txt", ".md", ".pdf"}
    
    # MongoDB Atlas Database Configuration (Loaded from .env)
    MONGODB_URI: str = os.getenv(
        "MONGODB_URI",
        "mongodb+srv://abhikollepara333:imY1MzMs9ZKcX6ql@cluster0.vmr7w.mongodb.net/Shiprule?retryWrites=true&w=majority&appName=Cluster0"
    )
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "Shiprule")

    # Authentication & Secrets (Loaded from .env)
    ADMIN_ACCESS_KEY: str = os.getenv("ADMIN_ACCESS_KEY", "krishna7")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "shiprule_secret_jwt_key_2026")
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "3"))

    # CORS & Client Connection Configuration (Loaded from .env)
    CLIENT_URL: str = os.getenv("CLIENT_URL", "http://localhost:3000")


settings = Settings()
