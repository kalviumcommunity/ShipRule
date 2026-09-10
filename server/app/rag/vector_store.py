"""
ShipRule CDLP - Vector Database Storage Layer (ChromaDB)
=========================================================
Provides persistent local vector storage, collection management, vector dimension
validation, metadata normalization, single/batch insertion, and deterministic readback
for the Customs Duty & Documentation Lookup Platform (CDLP).
"""

import os
import sys
from typing import Dict, List, Any, Optional, Union
import chromadb
from chromadb.api import ClientAPI
from dotenv import load_dotenv

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()


# ==============================================================================
# CONFIGURATION HELPERS
# ==============================================================================

def get_vector_db_config() -> Dict[str, str]:
    """
    Reads vector database configuration from environment variables with safe defaults.
    """
    load_dotenv()
    db_path = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    collection_name = os.getenv("VECTOR_COLLECTION_NAME", "shiprule_documents")
    distance_metric = os.getenv("VECTOR_DISTANCE_METRIC", "cosine")
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    return {
        "db_path": db_path,
        "collection_name": collection_name,
        "distance_metric": distance_metric,
        "embed_model": embed_model
    }


# ==============================================================================
# 1. CLIENT & CONNECTION MANAGEMENT
# ==============================================================================

def get_vector_db_client(db_path: Optional[str] = None) -> ClientAPI:
    """
    Initializes and returns a persistent ChromaDB client pointing to the configured path.
    Verifies that the database is reachable via heartbeat.
    """
    config = get_vector_db_config()
    target_path = db_path if db_path is not None else config["db_path"]

    # If relative path, resolve relative to project root
    if not os.path.isabs(target_path):
        resolved_path = os.path.normpath(os.path.join(project_root, target_path))
    else:
        resolved_path = target_path

    os.makedirs(resolved_path, exist_ok=True)

    try:
        client = chromadb.PersistentClient(path=resolved_path)
        # Verify reachability
        client.heartbeat()
        return client
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to persistent ChromaDB at '{resolved_path}': {e}"
        ) from e


def get_or_create_collection(
    client: Optional[ClientAPI] = None,
    collection_name: Optional[str] = None,
    distance_metric: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Retrieves or creates a ChromaDB collection with the specified distance metric.
    """
    config = get_vector_db_config()
    col_name = collection_name or config["collection_name"]
    metric = distance_metric or config["distance_metric"]

    if client is None:
        client = get_vector_db_client()

    collection_metadata = {"hnsw:space": metric}
    if metadata:
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                collection_metadata[k] = v

    try:
        collection = client.get_or_create_collection(
            name=col_name,
            metadata=collection_metadata
        )
        return collection
    except Exception as e:
        raise RuntimeError(
            f"Failed to get or create ChromaDB collection '{col_name}': {e}"
        ) from e


# ==============================================================================
# 2. DIMENSION DETECTION & VALIDATION
# ==============================================================================

def detect_embedding_dimension(
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> int:
    """
    Dynamically detects the embedding vector dimension produced by the active embedding pipeline.
    Avoids blind hardcoding of dimensions.
    """
    from app.rag.embeddings import create_embedding_client, generate_embedding

    config = get_vector_db_config()
    model_name = model or config["embed_model"]

    if client is None:
        client = create_embedding_client()

    probe_text = "ShipRule vector dimension probe"
    vector = generate_embedding(client=client, text=probe_text, model=model_name)
    dimension = len(vector)

    if dimension == 0:
        raise ValueError("Detected embedding vector dimension is zero.")

    return dimension


def validate_vector_dimension(vector: List[float], expected_dimension: int) -> bool:
    """
    Validates that a vector matches the expected dimension.
    Raises ValueError with a clear explanation if the dimension does not match.
    """
    if not isinstance(vector, (list, tuple)) and not hasattr(vector, "__iter__"):
        raise TypeError(f"Expected embedding vector to be a list/sequence of floats, got {type(vector).__name__}")

    actual_dim = len(vector)
    if actual_dim != expected_dimension:
        raise ValueError(
            f"Vector dimension mismatch: expected {expected_dimension}, got {actual_dim}."
        )

    return True


# ==============================================================================
# 3. METADATA NORMALIZATION & COMPATIBILITY
# ==============================================================================

def format_chroma_metadata(
    metadata_dict: Optional[Dict[str, Any]] = None,
    chunk_item: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[str, int, float, bool]]:
    """
    Constructs a ChromaDB-compliant metadata dictionary.
    Guarantees that all values are scalar types (str, int, float, bool),
    mapping None/missing values to safe non-empty defaults without inventing fake document information.
    """
    source_data: Dict[str, Any] = {}
    if chunk_item:
        source_data.update(chunk_item)
    if metadata_dict:
        source_data.update(metadata_dict)

    # Standardized metadata extraction
    source = str(source_data.get("source", "unknown"))
    chunk_index = int(source_data.get("chunk_index", 0))
    chunk_id = str(source_data.get("chunk_id", ""))
    document_type = str(source_data.get("document_type", source_data.get("doc_type", "txt")))
    embedding_model = str(source_data.get("embedding_model", ""))
    vector_dimension = int(source_data.get("vector_dimension", 0))

    # Handle section and page safely
    raw_section = source_data.get("section")
    section = str(raw_section) if raw_section is not None else ""

    raw_page = source_data.get("page")
    if raw_page is not None:
        try:
            page = int(raw_page)
        except (ValueError, TypeError):
            page = 0
    else:
        page = 0

    chroma_meta: Dict[str, Union[str, int, float, bool]] = {
        "source": source,
        "chunk_index": chunk_index,
        "section": section,
        "page": page,
        "chunk_id": chunk_id,
        "document_type": document_type,
        "embedding_model": embedding_model,
        "vector_dimension": vector_dimension
    }

    # Include optional fields if present and valid
    for key, val in source_data.items():
        if key in ("embedding", "chunk_text", "document_text", "document", "id", "embedding_id"):
            continue
        if key not in chroma_meta and isinstance(val, (str, int, float, bool)):
            chroma_meta[key] = val

    return chroma_meta


# ==============================================================================
# 4. RECORD INSERTION & UPSERT
# ==============================================================================

def insert_embedding_record(
    collection: Any,
    record: Dict[str, Any],
    expected_dimension: Optional[int] = None,
    upsert: bool = True
) -> str:
    """
    Inserts or upserts a single embedded chunk record into ChromaDB.
    Validates vector dimension before insertion.
    """
    # 1. Resolve Record ID
    record_id = record.get("embedding_id") or record.get("chunk_id") or record.get("id")
    if not record_id or not str(record_id).strip():
        raise ValueError("Record is missing a valid 'embedding_id', 'chunk_id', or 'id'.")
    record_id = str(record_id).strip()

    # 2. Resolve Document Text
    document_text = record.get("chunk_text") or record.get("document_text") or record.get("document") or ""
    if not document_text or not str(document_text).strip():
        raise ValueError(f"Record '{record_id}' is missing valid chunk/document text.")
    document_text = str(document_text)

    # 3. Resolve Embedding Vector
    vector = record.get("embedding") or record.get("vector")
    if vector is None:
        raise ValueError(f"Record '{record_id}' is missing an embedding vector.")

    # Convert to list of floats if needed
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    vector = [float(x) for x in vector]

    # 4. Validate Vector Dimension
    if expected_dimension is not None:
        validate_vector_dimension(vector, expected_dimension)

    # 5. Build Chroma-compatible Metadata
    raw_meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else record
    chroma_metadata = format_chroma_metadata(metadata_dict=raw_meta, chunk_item=record)
    if "vector_dimension" not in chroma_metadata or chroma_metadata["vector_dimension"] == 0:
        chroma_metadata["vector_dimension"] = len(vector)

    # 6. Insert / Upsert into ChromaDB
    try:
        if upsert and hasattr(collection, "upsert"):
            collection.upsert(
                ids=[record_id],
                documents=[document_text],
                embeddings=[vector],
                metadatas=[chroma_metadata]
            )
        else:
            collection.add(
                ids=[record_id],
                documents=[document_text],
                embeddings=[vector],
                metadatas=[chroma_metadata]
            )
        return record_id
    except Exception as e:
        raise RuntimeError(f"Failed to insert record '{record_id}' into collection: {e}") from e


def insert_embedding_records(
    collection: Any,
    records: List[Dict[str, Any]],
    expected_dimension: Optional[int] = None,
    upsert: bool = True
) -> List[str]:
    """
    Inserts or upserts multiple embedded chunk records into ChromaDB in batch.
    """
    inserted_ids: List[str] = []
    for record in records:
        inserted_id = insert_embedding_record(
            collection=collection,
            record=record,
            expected_dimension=expected_dimension,
            upsert=upsert
        )
        inserted_ids.append(inserted_id)
    return inserted_ids


# ==============================================================================
# 5. RECORD RETRIEVAL & READBACK
# ==============================================================================

def get_embedding_record(
    collection: Any,
    record_id: str
) -> Optional[Dict[str, Any]]:
    """
    Reads back a stored record by ID from ChromaDB including its embedding, metadata, and text.
    Returns None if record not found.
    """
    try:
        res = collection.get(
            ids=[record_id],
            include=["embeddings", "metadatas", "documents"]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to query ChromaDB for record '{record_id}': {e}") from e

    if not res or not res.get("ids") or len(res["ids"]) == 0:
        return None

    r_id = res["ids"][0]
    doc = res["documents"][0] if res.get("documents") else ""
    meta = res["metadatas"][0] if res.get("metadatas") else {}

    raw_embeddings = res.get("embeddings")
    if raw_embeddings is not None and len(raw_embeddings) > 0:
        emb_item = raw_embeddings[0]
        if hasattr(emb_item, "tolist"):
            vector = emb_item.tolist()
        else:
            vector = [float(x) for x in emb_item]
    else:
        vector = []

    return {
        "id": r_id,
        "embedding_id": r_id,
        "document_text": doc,
        "chunk_text": doc,
        "embedding": vector,
        "vector_dimension": len(vector),
        "metadata": meta
    }


# ==============================================================================
# 6. HEALTH CHECK
# ==============================================================================

def run_vector_db_health_check(
    db_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verifies vector database reachability, connects to collection, and prints formatted status.
    """
    config = get_vector_db_config()
    target_path = db_path or config["db_path"]
    target_collection = collection_name or config["collection_name"]

    try:
        client = get_vector_db_client(db_path=target_path)
        collection = get_or_create_collection(
            client=client,
            collection_name=target_collection
        )
        count = collection.count()

        status_result = {
            "status": "CONNECTED",
            "db_path": target_path,
            "collection": target_collection,
            "record_count": count
        }

        if verbose:
            print("[Vector DB]")
            print("Status: CONNECTED")
            print(f"Path: {target_path}")
            print(f"Collection: {target_collection}")

        return status_result

    except Exception as e:
        status_result = {
            "status": "ERROR",
            "db_path": target_path,
            "collection": target_collection,
            "error": str(e)
        }
        if verbose:
            print("[Vector DB]")
            print("Status: ERROR")
            print(f"Path: {target_path}")
            print(f"Error: {e}")
        return status_result


if __name__ == "__main__":
    run_vector_db_health_check()
