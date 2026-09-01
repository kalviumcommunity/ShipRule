"""
KDU 3.25: Embeddings Fundamentals & Vector Representation
===========================================================
Demonstrates how text is transformed into dense vector embeddings that represent semantic
meaning, reports vector dimensions, calculates cosine similarity between text pairs, and
explains why embeddings enable semantic search in RAG pipelines.
"""

import json
import os
import sys
import numpy as np
from typing import List, Dict, Any

# Ensure project root directory is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import chromadb.utils.embedding_functions as ef


def get_embedding_function():
    """
    Returns an embedding function for text vectorization.
    Uses ChromaDB's default ONNX embedding function (all-MiniLM-L6-v2, 384 dimensions)
    providing reliable offline local embedding generation.
    """
    return ef.DefaultEmbeddingFunction()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate dense numerical embedding vectors for a list of input texts.

    Args:
        texts: List of input strings to embed.

    Returns:
        List of float vectors representing semantic meaning in vector space.
    """
    embedding_fn = get_embedding_function()
    raw_embeddings = embedding_fn(texts)
    # Convert numpy arrays to standard python float lists if necessary
    embeddings = [
        [float(x) for x in vec] for vec in raw_embeddings
    ]
    return embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate the cosine similarity between two vector embeddings.

    Cosine similarity measures the cosine of the angle between two multi-dimensional vectors:
        cosine_similarity(a, b) = (a · b) / (||a|| * ||b||)

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Float score between -1.0 and 1.0 (where 1.0 indicates identical direction/meaning).
    """
    vec_a = np.array(a, dtype=np.float64)
    vec_b = np.array(b, dtype=np.float64)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def explain_vector_representation() -> Dict[str, str]:
    """
    Returns concise plain-term explanations of embedding vectors, vector dimensions,
    and why embeddings enable semantic search in RAG applications.
    """
    return {
        "what_is_embedding": (
            "An embedding vector is a numerical representation of semantic meaning in a high-dimensional space. "
            "Rather than using static database IDs or keyword counts (one-hot encodings), the embedding model converts "
            "text concepts into numeric coordinates so that phrases with similar underlying intent land close to each "
            "other in vector space."
        ),
        "what_is_dimension": (
            "The vector dimension represents the total number of numeric coordinates describing each text sample. "
            "For example, a 384-dimensional or 1536-dimensional vector contains 384 or 1536 continuous floats. "
            "Individual coordinates are not analyzed in isolation; the holistic pattern across all dimensions captures "
            "the nuances of semantics and contextual meaning."
        ),
        "why_semantic_search": (
            "In a RAG (Retrieval-Augmented Generation) pipeline, chunked documents are embedded and indexed into a "
            "vector database. When a user asks a query (e.g., 'How do I reset my password?'), the query is embedded into "
            "the same vector space. This enables semantic search: retrieval performs a nearest-neighbor vector similarity search "
            "(such as cosine similarity) to retrieve chunks with matching meaning (e.g., 'Steps to recover access to my login') "
            "even when exact keywords differ."
        )
    }


def run_embedding_demonstration() -> Dict[str, Any]:
    """
    Executes tasks 1-4 of KDU 3.25:
      - Task 1: Generate embeddings for sample texts (similar pair + CDLP PRD text + unrelated text)
      - Task 2: Report vector dimension & confirm equal lengths across all texts
      - Task 3: Compare similarity of similar vs. dissimilar text pairs using Cosine Similarity
      - Task 4: Explain what vector embeddings represent in plain terms
    """
    # Sample texts: Pair with similar intent, CDLP PRD customs text, plus an unrelated topic
    sample_texts = [
        "How do I reset my account password?",                                      # Text 0 (Target query)
        "Steps to recover access to my login",                                      # Text 1 (Similar intent)
        "What is the duty rate and required import documents for laptops in India?",# Text 2 (CDLP PRD Query)
        "The cafeteria menu has pasta today"                                         # Text 3 (Unrelated topic)
    ]

    # Task 1: Generate Embeddings
    embeddings = embed_texts(sample_texts)

    # Task 2: Report Vector Dimension & Verify Uniform Length
    dimensions = [len(vec) for vec in embeddings]
    vector_dimension = dimensions[0]
    all_dimensions_equal = all(d == vector_dimension for d in dimensions)

    # Task 3: Calculate Cosine Similarities
    sim_similar_pair = cosine_similarity(embeddings[0], embeddings[1])
    sim_dissimilar_pair = cosine_similarity(embeddings[0], embeddings[3])
    sim_cdlp_pair = cosine_similarity(embeddings[0], embeddings[2])

    ranking_passed = sim_similar_pair > sim_dissimilar_pair

    # Task 4: Explanations
    explanations = explain_vector_representation()

    results = {
        "sample_texts": [
            {"index": 0, "label": "Query (Password Reset)", "text": sample_texts[0]},
            {"index": 1, "label": "Similar Intent (Login Recovery)", "text": sample_texts[1]},
            {"index": 2, "label": "CDLP PRD Query (Customs Duties & Docs)", "text": sample_texts[2]},
            {"index": 3, "label": "Unrelated Topic (Cafeteria Menu)", "text": sample_texts[3]}
        ],
        "vector_dimension": vector_dimension,
        "all_dimensions_equal": all_dimensions_equal,
        "sample_vector_preview": {
            "text_0_first_8_values": embeddings[0][:8]
        },
        "similarity_comparison": {
            "similar_pair": {
                "text_a": sample_texts[0],
                "text_b": sample_texts[1],
                "cosine_similarity": round(sim_similar_pair, 4)
            },
            "dissimilar_pair": {
                "text_a": sample_texts[0],
                "text_b": sample_texts[3],
                "cosine_similarity": round(sim_dissimilar_pair, 4)
            },
            "cdlp_domain_pair": {
                "text_a": sample_texts[0],
                "text_b": sample_texts[2],
                "cosine_similarity": round(sim_cdlp_pair, 4)
            },
            "ranking_test_passed": ranking_passed
        },
        "explanations": explanations
    }

    return results


def export_demonstration_outputs():
    """
    Exports execution results to outputs/embedding_demo_output.json and outputs/embedding_demo_output.txt.
    """
    results = run_embedding_demonstration()

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Write JSON output
    json_path = os.path.join(output_dir, "embedding_demo_output.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Build human-readable text output string
    lines = []
    lines.append("======================================================================")
    lines.append("     KDU 3.25 EMBEDDINGS FUNDAMENTALS & VECTOR REPRESENTATION DEMO")
    lines.append("======================================================================\n")

    lines.append("--- TASK 1: GENERATED SAMPLE TEXT EMBEDDINGS ---")
    for item in results["sample_texts"]:
        lines.append(f"[{item['index']}] {item['label']}: \"{item['text']}\"")
    lines.append("")

    lines.append("--- TASK 2: VECTOR DIMENSION REPORT ---")
    lines.append(f"Vector Dimension: {results['vector_dimension']}")
    lines.append(f"Uniform Vector Length Confirmed: {'YES' if results['all_dimensions_equal'] else 'NO'}")
    lines.append(f"First 8 values of Vector #0: {results['sample_vector_preview']['text_0_first_8_values']}\n")

    lines.append("--- TASK 3: COSINE SIMILARITY COMPARISON ---")
    sim_info = results["similarity_comparison"]
    lines.append("1. Similar Pair:")
    lines.append(f"   Text A: \"{sim_info['similar_pair']['text_a']}\"")
    lines.append(f"   Text B: \"{sim_info['similar_pair']['text_b']}\"")
    lines.append(f"   Cosine Similarity Score: {sim_info['similar_pair']['cosine_similarity']:.4f}\n")

    lines.append("2. Dissimilar Pair:")
    lines.append(f"   Text A: \"{sim_info['dissimilar_pair']['text_a']}\"")
    lines.append(f"   Text B: \"{sim_info['dissimilar_pair']['text_b']}\"")
    lines.append(f"   Cosine Similarity Score: {sim_info['dissimilar_pair']['cosine_similarity']:.4f}\n")

    lines.append(f"Score Ranking Verification (Similar > Dissimilar): {'PASSED' if sim_info['ranking_test_passed'] else 'FAILED'}\n")

    lines.append("--- TASK 4: PLAIN-TERM EXPLANATION NOTES ---")
    exp = results["explanations"]
    lines.append(f"What is an Embedding Vector?\n  {exp['what_is_embedding']}\n")
    lines.append(f"What does Vector Dimension represent?\n  {exp['what_is_dimension']}\n")
    lines.append(f"Why does this enable Semantic Search in RAG?\n  {exp['why_semantic_search']}\n")

    output_text = "\n".join(lines)

    # Print directly to terminal console
    print(output_text)

    # Write Text output file
    txt_path = os.path.join(output_dir, "embedding_demo_output.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\n[Embedding Demo] Results exported to:\n  - {json_path}\n  - {txt_path}")
    return results


if __name__ == "__main__":
    export_demonstration_outputs()
