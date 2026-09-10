"""
ShipRule CDLP - MongoDB User Authentication & Storage Module
============================================================
Handles user registration, credential verification, and user management using MongoDB.
Includes automatic fallback to memory/file store if local MongoDB daemon is offline.
"""

import os
import hashlib
import secrets
from typing import Optional, Dict, Any
from app.core.config import settings
from app.core.logging import logger

try:
    import pymongo
    PYMONGO_AVAILABLE = True
except ImportError:
    pymongo = None
    PYMONGO_AVAILABLE = False


# In-memory user store fallback when MongoDB server connection fails or is offline
_MEMORY_USERS: Dict[str, Dict[str, Any]] = {}

def get_mongo_client() -> Optional[Any]:
    """Attempts to connect to MongoDB client with short timeout."""
    if not PYMONGO_AVAILABLE:
        return None
    mongo_uri = getattr(settings, "MONGODB_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    try:
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=1500)
        # Verify connection status
        client.admin.command('ping')
        return client
    except Exception as e:
        logger.debug(f"MongoDB connection ping failed (using fallback memory store): {e}")
        return None


def get_users_collection():
    """Returns PyMongo collection instance or None if offline."""
    client = get_mongo_client()
    if client is not None:
        db_name = getattr(settings, "MONGODB_DB_NAME", os.getenv("MONGODB_DB_NAME", "shiprule_db"))
        return client[db_name]["users"]
    return None


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Generates salted SHA-256 password hash."""
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return f"{salt}${hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verifies plain password against salted stored hash."""
    if not stored_hash or '$' not in stored_hash:
        return False
    salt, hashed = stored_hash.split('$', 1)
    computed = hash_password(password, salt=salt)
    return computed == stored_hash


def create_user(email: str, password: str, full_name: str = "", role: str = "user") -> Dict[str, Any]:
    """
    Registers a new user in MongoDB (or fallback store).
    Raises ValueError if email already exists.
    """
    clean_email = email.strip().lower()
    existing = find_user_by_email(clean_email)
    if existing:
        raise ValueError("User with this email address already exists.")

    password_hash = hash_password(password)
    user_doc = {
        "email": clean_email,
        "password_hash": password_hash,
        "full_name": full_name.strip(),
        "role": role.strip().lower() or "user"
    }

    collection = get_users_collection()
    if collection is not None:
        try:
            collection.insert_one(user_doc.copy())
            logger.info(f"Successfully registered user '{clean_email}' in MongoDB.")
        except Exception as e:
            logger.warning(f"MongoDB insert error, falling back to memory store: {e}")
            _MEMORY_USERS[clean_email] = user_doc
    else:
        _MEMORY_USERS[clean_email] = user_doc
        logger.info(f"Registered user '{clean_email}' in memory fallback store.")

    return {
        "email": clean_email,
        "full_name": full_name.strip(),
        "role": user_doc["role"]
    }


def find_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Retrieves user document by email from MongoDB or fallback store."""
    clean_email = email.strip().lower()

    collection = get_users_collection()
    if collection is not None:
        try:
            doc = collection.find_one({"email": clean_email})
            if doc:
                return {
                    "email": doc["email"],
                    "password_hash": doc.get("password_hash", ""),
                    "full_name": doc.get("full_name", ""),
                    "role": doc.get("role", "user")
                }
        except Exception as e:
            logger.warning(f"MongoDB search error: {e}")

    if clean_email in _MEMORY_USERS:
        return _MEMORY_USERS[clean_email]

    return None
