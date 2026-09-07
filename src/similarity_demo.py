"""
ShipRule CDLP - Embedding Similarity & Distance Metrics Demonstration
======================================================================
Demonstrates converting user queries into vector embeddings using the same embedding model,
calculating cosine similarity against candidate chunk embeddings, ranking results, and returning
top_k structured results.
"""

import os
import sys
import json
from typing import Dict, List, Any

# Ensure UTF-8 output handling on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.similarity import (
    cosine_similarity,
    generate_query_embedding,
    rank_chunks_by_similarity,
    search_similar_chunks,
)
from src.embeddings import create_embedding_client


def run_similarity_demonstration(top_k: int = 3) -> Dict[str, Any]:
    """
    Executes similarity search demonstration against candidate chunks.
    """
    # Sample candidate chunks with text, embeddings, and metadata
    sample_candidate_chunks = [
        {
            "embedding_id": "doc_chunk_001",
            "chunk_id": "shipping_rules_p1",
            "source": "shipping_rules.txt",
            "chunk_index": 1,
            "page": "1",
            "chunk_text": "All international cargo shipments require a verified commercial invoice and packing list before customs clearance.",
            "embedding": [0.12, 0.45, 0.78, 0.23, 0.05]
        },
        {
            "embedding_id": "doc_chunk_002",
            "chunk_id": "customs_requirements_p1",
            "source": "customs_requirements.txt",
            "chunk_index": 1,
            "page": "2",
            "chunk_text": "Import duty rates for electronic components (HS Code 8471.30) into India require a BIS registration certificate.",
            "embedding": [0.85, 0.62, 0.11, 0.04, 0.91]
        },
        {
            "embedding_id": "doc_chunk_003",
            "chunk_id": "international_guide_p1",
            "source": "international_shipping_guide.pdf",
            "chunk_index": 3,
            "page": "4",
            "chunk_text": "Customs duties and tariff rates depend upon the 8-digit HS Code classification and country of origin.",
            "embedding": [0.79, 0.58, 0.15, 0.08, 0.84]
        },
        {
            "embedding_id": "doc_chunk_004",
            "chunk_id": "office_menu_01",
            "source": "cafeteria_menu.txt",
            "chunk_index": 1,
            "page": "1",
            "chunk_text": "The office cafeteria serves fresh pasta and salad for lunch today.",
            "embedding": [0.01, 0.02, 0.03, 0.89, 0.01]
        }
    ]

    # Query matching laptops and customs duties in India
    query = "What are the required import documents and duty rates for shipping laptops to India?"
    query_vector = [0.82, 0.60, 0.12, 0.05, 0.88]  # Dense vector representing query

    # Perform similarity ranking
    ranked_results = rank_chunks_by_similarity(query_vector, sample_candidate_chunks, top_k=top_k)

    demo_report = {
        "query": query,
        "top_k_requested": top_k,
        "total_candidates_scored": len(sample_candidate_chunks),
        "results": ranked_results
    }

    # Save artifact to outputs/similarity_demo_output.json
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "similarity_demo_output.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(demo_report, f, indent=2, ensure_ascii=False)

    print("=================================================================")
    print("      SHIPRULE — EMBEDDING SIMILARITY & RANKING DEMO            ")
    print("=================================================================")
    print(f"Query: \"{query}\"")
    print(f"Requested Top-K: {top_k}\n")
    print("Ranked Chunk Results:")
    for idx, item in enumerate(ranked_results, start=1):
        meta = item["metadata"]
        print(f"  [{idx}] Score: {item['score']:.4f} | Source: {meta['source']} (Chunk Index: {meta['chunk_index']})")
        print(f"      Text: \"{item['text'][:100]}...\"\n")

    print(f"[SUCCESS] Similarity demonstration artifact saved to:\n  {json_path}")
    print("=================================================================\n")

    return demo_report


if __name__ == "__main__":
    run_similarity_demonstration()
