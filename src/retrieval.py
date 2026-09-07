"""
ShipRule CDLP - Similarity Search & Top-K Retrieval Module
===========================================================
Implements the retrieval pipeline (retrieve(query, k=3)) using the exact same
embedding model used for document embeddings and the existing indexed vector collection.
Returns ordered top-k most similar chunks with rank, similarity score, chunk text,
source, chunk index, and all metadata.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple

# Ensure project root directory is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# UTF-8 stdout configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from src.embeddings import (
    create_embedding_client,
    generate_query_embedding,
    cosine_similarity,
    rank_chunks_by_similarity
)
from src.indexing import VectorCollection, to_vector_record


# ==============================================================================
# 1. VECTOR COLLECTION LOADER
# ==============================================================================

def load_indexed_vector_collection(
    chunks_file: Optional[str] = None
) -> VectorCollection:
    """
    Loads pre-indexed document chunk embeddings from file into a VectorCollection.
    Reuses existing embeddings without re-indexing or changing the embedding model.

    Args:
        chunks_file: Optional path to JSON file with embedded chunks.
                     Defaults to outputs/embedded_chunks.json.

    Returns:
        Populated VectorCollection instance.
    """
    if not chunks_file:
        chunks_file = os.path.join(project_root, "outputs", "embedded_chunks.json")

    # Fallback to processed chunks if embedded_chunks.json is missing
    if not os.path.exists(chunks_file):
        fallback = os.path.join(project_root, "outputs", "processed_chunks.json")
        if os.path.exists(fallback):
            chunks_file = fallback

    if not os.path.exists(chunks_file):
        raise FileNotFoundError(
            f"Indexed vector collection source file not found at '{chunks_file}'."
        )

    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Invalid vector store file format in '{chunks_file}': expected list.")

    collection = VectorCollection(name="cdlp_indexed_collection")

    vector_records = []
    for idx, item in enumerate(data, start=1):
        # Extract fields from item
        rec_id = item.get("id") or item.get("chunk_id") or item.get("embedding_id") or f"chunk_{idx}"
        vec = item.get("embedding") if "embedding" in item else item.get("vector", [])
        text = item.get("chunk_text") or item.get("text") or ""
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}

        source = item.get("source") or meta.get("source", "unknown")
        chunk_idx = item.get("chunk_index") or meta.get("chunk_index", idx)

        merged_meta = {
            "source": str(source),
            "chunk_index": chunk_idx,
            "section": item.get("section") or meta.get("section"),
            "country": item.get("country") or meta.get("country"),
            "hs_code": item.get("hs_code") or meta.get("hs_code"),
            "document_type": item.get("document_type") or meta.get("document_type"),
            "strategy": item.get("strategy") or meta.get("strategy"),
            "page": str(item.get("page") or meta.get("page", "1")),
            "embedding_id": item.get("embedding_id") or item.get("id") or f"emb_{idx:03d}"
        }

        rec = {
            "id": rec_id,
            "vector": list(vec) if vec else [],
            "text": text,
            "metadata": merged_meta
        }
        vector_records.append(rec)

    collection.upsert(vector_records)
    return collection


# ==============================================================================
# 2. CORE RETRIEVAL PIPELINE FUNCTION
# ==============================================================================

def retrieve(
    query: str,
    k: int = 3,
    collection: Optional[VectorCollection] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Executes top-k similarity retrieval for a given natural language query.

    Steps:
    1. Validates query string and k parameter.
    2. Generates query embedding vector using exact same embedding model.
    3. Searches existing vector collection using cosine similarity.
    4. Returns ranked list of results with rank, score, text, source, chunk_index, and metadata.

    Args:
        query: Non-empty search query string.
        k: Positive integer specifying number of top results to return.
        collection: Optional VectorCollection instance (loaded automatically if None).
        client: Optional embedding client instance.
        model: Optional embedding model name string.

    Returns:
        List of result dicts sorted from most similar to least similar.
    """
    # 1. Input Query Validation
    if query is None or not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # 2. Input K Parameter Validation
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("Parameter 'k' must be a positive integer greater than 0.")

    # 3. Vector Collection Resolution & Validation
    if collection is None:
        collection = load_indexed_vector_collection()

    total_chunks = collection.count()
    if total_chunks == 0:
        raise ValueError("Vector collection is empty. No indexed document chunks available for retrieval.")

    # Cap k to total indexed chunks if k > total_chunks
    effective_k = min(k, total_chunks)

    # 4. Query Embedding Generation (using exact same model)
    load_dotenv()
    selected_model = model or os.getenv("EMBED_MODEL", "text-embedding-3-small")

    try:
        query_vector = generate_query_embedding(query.strip(), client=client, model=selected_model)
    except Exception as e:
        raise RuntimeError(f"Embedding API error while generating query vector: {e}")

    if not query_vector:
        raise RuntimeError("Failed to generate vector embedding for the input query.")

    # 5. Vector Similarity Search
    try:
        raw_matches = collection.query(query_vector, top_k=effective_k)
    except Exception as e:
        raise RuntimeError(f"Vector database search error: {e}")

    # 6. Format and Structure Results
    structured_results: List[Dict[str, Any]] = []

    for rank_idx, match in enumerate(raw_matches, start=1):
        meta = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}

        # Handle missing or fallback metadata fields
        source = meta.get("source") or match.get("source", "unknown")
        chunk_idx = meta.get("chunk_index") if meta.get("chunk_index") is not None else match.get("chunk_index", rank_idx)
        text = match.get("text") or meta.get("chunk_text") or "[Missing chunk text]"
        score = match.get("score", 0.0)

        structured_results.append({
            "rank": rank_idx,
            "similarity_score": round(float(score), 4),
            "chunk_text": str(text),
            "source": str(source),
            "chunk_index": chunk_idx,
            "metadata": meta,
            "embedding_model": selected_model,
            "document_id": match.get("id", f"chunk_{rank_idx}")
        })

    return structured_results


# ==============================================================================
# 3. RETRIEVAL OUTPUT FORMATTER
# ==============================================================================

def format_retrieval_output(
    query: str,
    results: List[Dict[str, Any]],
    model_name: str = "text-embedding-3-small",
    k: int = 3
) -> str:
    """
    Formats top-k retrieval results into a clean, human-readable ASCII text report.

    Args:
        query: The user query string.
        results: List of structured result dicts returned by retrieve().
        model_name: Embedding model used.
        k: Configured top-k value.

    Returns:
        Formatted output string.
    """
    lines = [
        "========================================",
        "Top-K Retrieval",
        "===============",
        "",
        f"Query: {query}",
        f"Embedding Model: {model_name}",
        f"Top-K: {k}",
        ""
    ]

    if not results:
        lines.append("No relevant chunks retrieved.")
        return "\n".join(lines)

    for item in results:
        lines.append(f"--- Rank {item['rank']} ---")
        lines.append(f"Similarity Score: {item['similarity_score']:.4f}")
        lines.append(f"Source: {item['source']}")
        lines.append(f"Chunk Index: {item['chunk_index']}")
        lines.append(f"Text: {item['chunk_text']}")
        lines.append("")

    return "\n".join(lines).strip()


# ==============================================================================
# 4. TOP-K DEMONSTRATION & TEST RUNNER
# ==============================================================================

def run_retrieval_demonstration() -> Dict[str, Any]:
    """
    Runs top-k retrieval demonstrations for specified queries with k=1, k=3, and k=5,
    prints output reports, and saves output artifacts to disk.
    """
    load_dotenv()
    model_name = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    # Load vector collection
    collection = load_indexed_vector_collection()

    test_queries = [
        "How can a learner reset their password?",
        "What are the shipping rules in India?",
        "What documents are required?",
        "How many moons does Jupiter have?"  # Unrelated query with low relevance
    ]

    all_demo_results: Dict[str, Any] = {
        "embedding_model": model_name,
        "total_indexed_chunks": collection.count(),
        "query_runs": []
    }

    report_text_blocks = []

    print("========================================================================")
    print("      SHIPRULE CDLP - SIMILARITY SEARCH & TOP-K RETRIEVAL DEMO          ")
    print("========================================================================")
    print(f"Embedding Model: {model_name}")
    print(f"Indexed Collection Size: {collection.count()} chunks\n")

    for q in test_queries:
        query_run_data = {
            "query": q,
            "k_runs": {}
        }

        for k in [1, 3, 5]:
            results = retrieve(q, k=k, collection=collection, model=model_name)
            query_run_data["k_runs"][f"k_{k}"] = results

            formatted = format_retrieval_output(q, results, model_name=model_name, k=k)
            report_text_blocks.append(formatted)
            report_text_blocks.append("\n" + "=" * 50 + "\n")

            print(formatted)
            print("-" * 50 + "\n")

        all_demo_results["query_runs"].append(query_run_data)

    # Save artifacts to outputs/
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "retrieval_demo_output.json")
    text_path = os.path.join(output_dir, "retrieval_output.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_demo_results, f, indent=2, ensure_ascii=False)

    full_text_report = "\n".join(report_text_blocks)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full_text_report)

    print(f"[SUCCESS] Retrieval artifacts saved successfully:")
    print(f"  JSON: {json_path}")
    print(f"  TXT : {text_path}")
    print("========================================================================\n")

    return all_demo_results


if __name__ == "__main__":
    run_retrieval_demonstration()
