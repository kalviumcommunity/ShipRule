"""
ShipRule CDLP - Vector Database End-to-End Verification Test
============================================================
Connects to persistent ChromaDB, verifies collection reachability,
generates a real embedding using the existing embedding pipeline,
validates vector dimensions, inserts a test record, and reads it back
from ChromaDB to confirm complete round-trip integrity.
"""

import os
import sys
import json

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
from src.embeddings import create_embedding_client, generate_embedding
from src.vector_store import (
    get_vector_db_config,
    get_vector_db_client,
    get_or_create_collection,
    detect_embedding_dimension,
    validate_vector_dimension,
    insert_embedding_record,
    get_embedding_record
)


def run_vector_db_test() -> bool:
    """
    Executes the end-to-end vector database readback test.
    """
    load_dotenv()
    config = get_vector_db_config()
    db_path = config["db_path"]
    collection_name = config["collection_name"]
    model_name = config["embed_model"]

    # 1. Connect to Vector DB
    client = get_vector_db_client(db_path=db_path)
    collection = get_or_create_collection(
        client=client,
        collection_name=collection_name
    )

    # 2. Generate Real Test Embedding via Existing Pipeline
    embed_client = create_embedding_client()
    detected_dim = detect_embedding_dimension(client=embed_client, model=model_name)

    test_id = "vector_test_record_001"
    test_text = (
        "Customs declaration documents must include commercial invoices, packing lists, "
        "and certificates of origin for cross-border freight compliance."
    )
    test_source = "shipping_rules.txt"

    # Generate real vector
    vector = generate_embedding(
        client=embed_client,
        text=test_text,
        model=model_name
    )
    vector_len = len(vector)

    # Validate dimension
    validate_vector_dimension(vector, detected_dim)
    dim_check = "PASS" if vector_len == detected_dim else "FAIL"

    # 3. Construct Record Schema
    test_record = {
        "embedding_id": test_id,
        "chunk_id": "shipping_rules_p01_test",
        "source": test_source,
        "document_type": "txt",
        "strategy": "paragraph",
        "chunk_index": 1,
        "character_count": len(test_text),
        "chunk_text": test_text,
        "embedding_model": model_name,
        "vector_dimension": vector_len,
        "embedding": vector,
        "section": "Documentation Requirements",
        "page": 1
    }

    # 4. Insert Test Record (Idempotent upsert)
    inserted_id = insert_embedding_record(
        collection=collection,
        record=test_record,
        expected_dimension=detected_dim,
        upsert=True
    )
    insert_check = "PASS" if inserted_id == test_id else "FAIL"

    # 5. Read Back from ChromaDB
    readback = get_embedding_record(collection=collection, record_id=test_id)
    if readback is None:
        readback_check = "FAIL"
        readback_status = "FAILED"
        rb_id = "N/A"
        rb_len = 0
        rb_text = "N/A"
        rb_meta = {}
    else:
        readback_status = "SUCCESS"
        rb_id = readback["id"]
        rb_len = readback["vector_dimension"]
        rb_text = readback["document_text"]
        rb_meta = readback["metadata"]
        
        # Verify readback integrity
        if (
            rb_id == test_id
            and rb_len == vector_len
            and rb_text == test_text
            and rb_meta.get("source") == test_source
        ):
            readback_check = "PASS"
        else:
            readback_check = "FAIL"

    # 6. Format and Print Test Output
    metadata_repr = json.dumps(test_record.get("metadata", {
        "source": test_source,
        "chunk_index": 1,
        "section": "Documentation Requirements",
        "page": 1,
        "chunk_id": "shipping_rules_p01_test",
        "document_type": "txt",
        "embedding_model": model_name,
        "vector_dimension": vector_len
    }))

    readback_meta_repr = json.dumps(rb_meta)

    print("========================================")
    print("VECTOR DATABASE TEST")
    print(f"Database status : CONNECTED")
    print(f"Collection      : {collection_name}")
    print("Inserted record:")
    print(f"  ID            : {test_id}")
    print(f"  Vector length : {vector_len}")
    print(f"  Text          : {test_text}")
    print(f"  Metadata      : {metadata_repr}")
    print("Readback:")
    print(f"  Status        : {readback_status}")
    print(f"  ID            : {rb_id}")
    print(f"  Vector length : {rb_len}")
    print(f"  Text          : {rb_text}")
    print(f"  Metadata      : {readback_meta_repr}")
    print(f"Dimension check : {dim_check}")
    print(f"Insert check    : {insert_check}")
    print(f"Readback check  : {readback_check}")
    print("========================================")

    return (
        dim_check == "PASS"
        and insert_check == "PASS"
        and readback_check == "PASS"
    )


if __name__ == "__main__":
    success = run_vector_db_test()
    if not success:
        sys.exit(1)
