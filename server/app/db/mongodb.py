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
_QUERY_LOGS: list = []

def get_mongo_client() -> Optional[Any]:
    """Attempts to connect to MongoDB client with short timeout."""
    if not PYMONGO_AVAILABLE:
        return None
    mongo_uri = getattr(settings, "MONGODB_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    try:
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        # Verify connection status
        client.admin.command('ping')
        return client
    except Exception as e:
        logger.debug(f"MongoDB connection ping failed (using fallback memory store): {e}")
        return None


def get_db() -> Optional[Any]:
    """Returns PyMongo database instance from default URI database name or fallback."""
    client = get_mongo_client()
    if client is not None:
        try:
            return client.get_default_database()
        except Exception:
            db_name = getattr(settings, "MONGODB_DB_NAME", os.getenv("MONGODB_DB_NAME", "Shiprule"))
            return client[db_name]
    return None


def get_users_collection():
    """Returns PyMongo users collection instance or None if offline."""
    db = get_db()
    if db is not None:
        return db["users"]
    return None


def get_query_logs_collection():
    """Returns PyMongo query_logs collection instance or None if offline."""
    db = get_db()
    if db is not None:
        return db["query_logs"]
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


def update_user_password(email: str, old_password: str, new_password: str) -> bool:
    """Verifies old password and updates user password in MongoDB / memory store."""
    clean_email = email.strip().lower()
    user = find_user_by_email(clean_email)
    if not user:
        raise ValueError("User not found.")

    if not verify_password(old_password, user.get("password_hash", "")):
        raise ValueError("Current password is incorrect.")

    if len(new_password) < 4:
        raise ValueError("New password must be at least 4 characters long.")

    new_hash = hash_password(new_password)

    collection = get_users_collection()
    if collection is not None:
        try:
            collection.update_one({"email": clean_email}, {"$set": {"password_hash": new_hash}})
        except Exception as e:
            logger.warning(f"MongoDB password update error: {e}")

    if clean_email in _MEMORY_USERS:
        _MEMORY_USERS[clean_email]["password_hash"] = new_hash

    return True


def list_all_registered_users() -> list:
    """Lists all registered users from MongoDB or fallback memory store with aggregated query/token metrics."""
    collection = get_users_collection()
    users = []
    logs_col = get_query_logs_collection()

    if collection is not None:
        try:
            cursor = list(collection.find({}, {"_id": 0, "password_hash": 0}))
            # Enrich with query count and token usage
            for u in cursor:
                email = u.get("email", "")
                if logs_col is not None and email:
                    try:
                        u_logs = list(logs_col.find({"user_email": email}, {"_id": 0}))
                        u["total_queries"] = len(u_logs)
                        u["total_tokens"] = sum(l.get("total_tokens", 0) for l in u_logs)
                        u["last_active"] = u_logs[0].get("timestamp") if u_logs else None
                    except Exception:
                        u["total_queries"] = 0
                        u["total_tokens"] = 0
                        u["last_active"] = None
                else:
                    u["total_queries"] = 0
                    u["total_tokens"] = 0
                    u["last_active"] = None
            return cursor
        except Exception as e:
            logger.warning(f"MongoDB list users error: {e}")

    for email, u in _MEMORY_USERS.items():
        u_logs = [l for l in _QUERY_LOGS if l.get("user_email") == email]
        users.append({
            "email": u["email"],
            "full_name": u.get("full_name", ""),
            "role": u.get("role", "user"),
            "total_queries": len(u_logs),
            "total_tokens": sum(l.get("total_tokens", 0) for l in u_logs),
            "last_active": u_logs[0].get("timestamp") if u_logs else None
        })
    return users


def log_query_activity(
    question: str,
    status: str,
    user_email: str = "anonymous",
    latency_ms: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0
):
    """Logs user query execution activity to MongoDB or memory store."""
    from datetime import datetime
    computed_total = total_tokens or (prompt_tokens + completion_tokens)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_email": user_email,
        "question": question,
        "status": status,
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": computed_total
    }
    logs_col = get_query_logs_collection()
    if logs_col is not None:
        try:
            logs_col.insert_one(entry.copy())
        except Exception as e:
            logger.warning(f"MongoDB query log insert error: {e}")
            _QUERY_LOGS.insert(0, entry)
    else:
        _QUERY_LOGS.insert(0, entry)
        if len(_QUERY_LOGS) > 100:
            _QUERY_LOGS.pop()


def get_all_query_logs(limit: int = 50) -> list:
    """Returns list of recent user query activity logs."""
    logs_col = get_query_logs_collection()
    if logs_col is not None:
        try:
            cursor = logs_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.warning(f"MongoDB get logs error: {e}")
    return _QUERY_LOGS[:limit]


def get_user_analytics_by_email(user_email: str) -> dict:
    """Returns comprehensive activity and token analytics for a specific user."""
    # Find user profile
    users_col = get_users_collection()
    user_info = None
    if users_col is not None:
        try:
            user_info = users_col.find_one({"email": user_email}, {"_id": 0, "password_hash": 0})
        except Exception as e:
            logger.warning(f"MongoDB find user error: {e}")

    if not user_info and user_email in _MEMORY_USERS:
        u = _MEMORY_USERS[user_email]
        user_info = {
            "email": u["email"],
            "full_name": u.get("full_name", ""),
            "role": u.get("role", "user")
        }

    if not user_info:
        user_info = {
            "email": user_email,
            "full_name": user_email.split('@')[0].replace('.', ' ').title() if '@' in user_email else user_email,
            "role": "user"
        }

    # Fetch logs for this user
    logs_col = get_query_logs_collection()
    user_logs = []
    if logs_col is not None:
        try:
            cursor = logs_col.find({"user_email": user_email}, {"_id": 0}).sort("timestamp", -1)
            user_logs = list(cursor)
        except Exception as e:
            logger.warning(f"MongoDB get user logs error: {e}")
            user_logs = [l for l in _QUERY_LOGS if l.get("user_email") == user_email]
    else:
        user_logs = [l for l in _QUERY_LOGS if l.get("user_email") == user_email]

    total_queries = len(user_logs)
    prompt_tokens = sum(l.get("prompt_tokens", 0) for l in user_logs)
    completion_tokens = sum(l.get("completion_tokens", 0) for l in user_logs)
    total_tokens = sum(l.get("total_tokens", 0) for l in user_logs)
    
    latencies = [l.get("latency_ms", 0.0) for l in user_logs if l.get("latency_ms")]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    successful_queries = sum(1 for l in user_logs if l.get("status") == "SUPPORTED")
    failed_queries = total_queries - successful_queries
    last_active = user_logs[0].get("timestamp") if user_logs else None

    return {
        "user": user_info,
        "metrics": {
            "total_queries": total_queries,
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "avg_latency_ms": avg_latency,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "last_active": last_active
        },
        "logs": user_logs
    }

