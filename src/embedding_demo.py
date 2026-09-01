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


# Standard Logistics Rules & Regulations Corpus aligned with CDLP PRD
LOGISTICS_PRD_CORPUS = [
    {
        "id": "REG-IN-8471",
        "title": "Customs Record - Laptop Computers (India)",
        "text": (
            "Customs Record for Laptop Computers in India: Destination Country: India. Commodity: Laptop Computers / "
            "Portable Automatic Data Processing Machines. HS Code: 8471.30. Duty Rate: 7.5% Basic Customs Duty + 10% "
            "Social Welfare Surcharge (SWS). Required Documents: Commercial Invoice, Bill of Lading, BIS Registration "
            "Certificate, DGFT Import License. Restricted Status: Restricted (Import License Required). Source Agency: "
            "Directorate General of Foreign Trade (DGFT) & CBIC, India. Source URL: https://www.cbic.gov.in. "
            "Last Confirmed Date: 2026-02-10."
        ),
        "country": "India",
        "hs_code": "8471.30"
    },
    {
        "id": "REG-IT-8703",
        "title": "Customs Record - Motor Vehicles (Italy)",
        "text": (
            "Customs Record for Cars / Motor Vehicles in Italy: Destination Country: Italy. Commodity: Cars / Motor Vehicles "
            "principally designed for transport of persons. HS Code: 8703. Duty Rate: 10.0% Basic Customs Duty + 22% "
            "Value Added Tax (VAT). Required Documents: Commercial Invoice, Certificate of Origin, Bill of Lading, EU "
            "Type-Approval Certificate, EUR.1 Movement Certificate. Restricted Status: Unrestricted (subject to EU emissions "
            "standards). Source Agency: Agenzia delle Dogane e dei Monopoli (ADM), Italy / European Union TARIC. "
            "Source URL: https://ec.europa.eu/taxation_customs/dds2/taric. Last Confirmed Date: 2026-01-15."
        ),
        "country": "Italy",
        "hs_code": "8703"
    },
    {
        "id": "REG-DE-8541",
        "title": "Customs Record - Solar Photovoltaic Modules (Germany)",
        "text": (
            "Customs Record for Solar Panels / Photovoltaic Modules in Germany: Destination Country: Germany. Commodity: "
            "Photosensitive semiconductor devices / Solar PV Modules. HS Code: 8541.43. Duty Rate: 0.0% Basic Customs Duty + "
            "19% VAT (0% VAT for residential PV installations). Required Documents: Commercial Invoice, Packing List, CE "
            "Declaration of Conformity, Bill of Lading. Restricted Status: Unrestricted. Source Agency: Bundeszollverwaltung, "
            "Germany / EU Customs. Source URL: https://www.zoll.de. Last Confirmed Date: 2026-01-20."
        ),
        "country": "Germany",
        "hs_code": "8541.43"
    },
    {
        "id": "REG-BR-9018",
        "title": "Customs Record - Medical Diagnostic Equipment (Brazil)",
        "text": (
            "Customs Record for Medical Diagnostic Instruments in Brazil: Destination Country: Brazil. Commodity: Electromedical "
            "instruments and appliances for medical diagnosis. HS Code: 9018.90. Duty Rate: 14.0% Import Duty (II) + 1.65% "
            "PIS + 7.6% COFINS. Required Documents: Commercial Invoice, Air Waybill, ANVISA Sanitary Registration Certificate, "
            "Certificate of Free Sale. Restricted Status: Restricted (ANVISA Sanitary Approval mandatory prior to customs clearance). "
            "Source Agency: Receita Federal & ANVISA, Brazil. Source URL: https://www.gov.br/receitafederal. "
            "Last Confirmed Date: 2026-02-01."
        ),
        "country": "Brazil",
        "hs_code": "9018.90"
    },
    {
        "id": "MISC-OFFICE-01",
        "title": "Unrelated Document - Cafeteria Menu",
        "text": "The corporate cafeteria lunch menu for today includes penne pasta arrabbiata, garlic bread, and fresh fruit salad.",
        "country": "N/A",
        "hs_code": "N/A"
    }
]


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


def demo_logistics_query_matching() -> Dict[str, Any]:
    """
    Demonstrates embedding-based matching of user logistics queries against PRD logistics rules records.
    """
    user_query = "What are the required import documents and tariff rates for bringing laptops into India under CDLP rules?"
    query_vec = embed_texts([user_query])[0]

    corpus_texts = [item["text"] for item in LOGISTICS_PRD_CORPUS]
    corpus_vecs = embed_texts(corpus_texts)

    scored_records = []
    for item, vec in zip(LOGISTICS_PRD_CORPUS, corpus_vecs):
        score = cosine_similarity(query_vec, vec)
        scored_records.append({
            "id": item["id"],
            "title": item["title"],
            "country": item["country"],
            "hs_code": item["hs_code"],
            "cosine_similarity": round(score, 4),
            "text_snippet": item["text"][:120] + "..."
        })

    # Sort descending by cosine similarity score
    scored_records.sort(key=lambda x: x["cosine_similarity"], reverse=True)

    top_match = scored_records[0]
    passed = (top_match["id"] == "REG-IN-8471") and (top_match["cosine_similarity"] > 0.6)

    return {
        "user_query": user_query,
        "ranked_results": scored_records,
        "top_match_id": top_match["id"],
        "top_match_title": top_match["title"],
        "top_match_score": top_match["cosine_similarity"],
        "test_passed": passed
    }


def demo_paraphrase_robustness() -> Dict[str, Any]:
    """
    Demonstrates how semantic embeddings score high similarity for paraphrased logistics queries
    even when different vocabulary is used.
    """
    phrase1 = "What paperwork is mandatory for clearing customs when importing laptop computers to India?"
    phrase2 = "Required import documentation and customs clearance requirements for Indian laptop shipments."
    unrelated = "How to install a graphics card driver on Windows 11?"

    vecs = embed_texts([phrase1, phrase2, unrelated])

    sim_paraphrase = cosine_similarity(vecs[0], vecs[1])
    sim_unrelated = cosine_similarity(vecs[0], vecs[2])

    passed = sim_paraphrase > (sim_unrelated + 0.3)

    return {
        "phrase_a": phrase1,
        "phrase_b_paraphrase": phrase2,
        "unrelated_phrase": unrelated,
        "paraphrase_similarity": round(sim_paraphrase, 4),
        "unrelated_similarity": round(sim_unrelated, 4),
        "test_passed": passed
    }


def run_embedding_demonstration() -> Dict[str, Any]:
    """
    Executes all tasks of KDU 3.25:
      - Task 1: Generate embeddings for sample texts (similar pair + CDLP PRD text + unrelated text)
      - Task 2: Report vector dimension & confirm equal lengths across all texts
      - Task 3: Compare similarity of similar vs. dissimilar text pairs using Cosine Similarity
      - Task 4: Explain what vector embeddings represent in plain terms
      - Task 5: Logistics Query Matching against PRD Rules Corpus
      - Task 6: Semantic Paraphrase Robustness Demo
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

    # Additional Demos (Task 5 & 6)
    logistics_match_demo = demo_logistics_query_matching()
    paraphrase_demo = demo_paraphrase_robustness()

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
        "explanations": explanations,
        "logistics_rules_matching_demo": logistics_match_demo,
        "paraphrase_robustness_demo": paraphrase_demo
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

    lines.append("--- TASK 5: LOGISTICS RULES & PRD MATCHING DEMO ---")
    log_demo = results["logistics_rules_matching_demo"]
    lines.append(f"User Query: \"{log_demo['user_query']}\"\n")
    lines.append("Corpus Similarity Ranking:")
    for rank, item in enumerate(log_demo["ranked_results"], start=1):
        lines.append(f"  Rank #{rank}: [{item['id']}] {item['title']} (Score: {item['cosine_similarity']:.4f})")
    lines.append(f"\nTop Match: [{log_demo['top_match_id']}] {log_demo['top_match_title']} (Score: {log_demo['top_match_score']:.4f})")
    lines.append(f"Logistics Retrieval Test: {'PASSED' if log_demo['test_passed'] else 'FAILED'}\n")

    lines.append("--- TASK 6: PARAPHRASE ROBUSTNESS DEMO ---")
    para_demo = results["paraphrase_robustness_demo"]
    lines.append(f"Phrase A: \"{para_demo['phrase_a']}\"")
    lines.append(f"Phrase B (Paraphrase): \"{para_demo['phrase_b_paraphrase']}\"")
    lines.append(f"Paraphrase Similarity Score: {para_demo['paraphrase_similarity']:.4f}")
    lines.append(f"Unrelated Topic Similarity Score: {para_demo['unrelated_similarity']:.4f}")
    lines.append(f"Paraphrase Robustness Test: {'PASSED' if para_demo['test_passed'] else 'FAILED'}\n")

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
