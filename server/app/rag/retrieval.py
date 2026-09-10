"""
ShipRule CDLP - Similarity Search, Metadata Filtering & Hybrid Retrieval Module
================================================================================
Implements the retrieval pipeline supporting:
1. Top-k similarity retrieval (retrieve(query, k=3, metadata_filter=None))
2. Metadata filtering over stored vector collection attributes
3. Keyword match scoring (keyword_score(text, keywords))
4. Hybrid weighted ranking combining vector and keyword scores (hybrid_rank())
5. End-to-end hybrid retrieval pipeline (hybrid_retrieve())
6. Demonstration formatters and reporting utilities
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple, Union

# Ensure project root directory is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# UTF-8 stdout configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from app.rag.embeddings import (
    create_embedding_client,
    generate_query_embedding,
    cosine_similarity,
    rank_chunks_by_similarity
)
from app.rag.indexing import VectorCollection, to_vector_record


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
            "section": item.get("section") or meta.get("section") or "General Customs",
            "country": item.get("country") or meta.get("country"),
            "hs_code": item.get("hs_code") or meta.get("hs_code"),
            "document_type": item.get("document_type") or meta.get("document_type", "txt"),
            "strategy": item.get("strategy") or meta.get("strategy", "paragraph"),
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
# 2. EXTENDED RETRIEVAL FUNCTION WITH METADATA FILTERING
# ==============================================================================

def retrieve(
    query: str,
    k: int = 3,
    metadata_filter: Optional[Dict[str, Any]] = None,
    collection: Optional[VectorCollection] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Executes top-k similarity retrieval for a given natural language query with
    optional metadata filtering.

    Metadata Filtering Behavior:
    - Unfiltered search (metadata_filter=None) searches the entire indexed corpus.
    - Filtered search (metadata_filter={...}) restricts evaluation exclusively to
      chunks matching all metadata filter key-value pairs.
    - Filtering increases precision by removing irrelevant document categories.
    - Overly strict filters can reduce recall if relevant chunks are omitted.
    - Metadata must be attached and stored during document ingestion to filter.

    Args:
        query: Non-empty search query string.
        k: Positive integer specifying number of top results to return.
        metadata_filter: Optional dictionary of key-value metadata criteria.
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

    # 3. Input Metadata Filter Validation
    if metadata_filter is not None and not isinstance(metadata_filter, dict):
        raise ValueError("metadata_filter must be a dictionary.")

    # 4. Vector Collection Resolution & Validation
    if collection is None:
        collection = load_indexed_vector_collection()

    total_chunks = collection.count()
    if total_chunks == 0:
        raise ValueError("Vector collection is empty. No indexed document chunks available for retrieval.")

    # Cap k to total indexed chunks if k > total_chunks
    effective_k = min(k, total_chunks)

    # 5. Query Embedding Generation (using exact same model)
    load_dotenv()
    selected_model = model or os.getenv("EMBED_MODEL", "text-embedding-3-small")

    try:
        query_vector = generate_query_embedding(query.strip(), client=client, model=selected_model)
    except Exception as e:
        raise RuntimeError(f"Embedding API error while generating query vector: {e}")

    if not query_vector:
        raise RuntimeError("Failed to generate vector embedding for the input query.")

    # 6. Vector Similarity Search with Metadata Filter
    try:
        raw_matches = collection.query(
            query_vector,
            top_k=effective_k,
            metadata_filter=metadata_filter
        )
    except Exception as e:
        raise RuntimeError(f"Vector database search error: {e}")

    # 7. Format and Structure Results
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
            "vector_score": round(float(score), 4),
            "chunk_text": str(text),
            "source": str(source),
            "chunk_index": chunk_idx,
            "section": str(meta.get("section", "General")),
            "metadata": meta,
            "embedding_model": selected_model,
            "document_id": match.get("id", f"chunk_{rank_idx}")
        })

    return structured_results


# ==============================================================================
# 3. KEYWORD SCORING FUNCTION
# ==============================================================================

def keyword_score(text: str, keywords: List[str]) -> float:
    """
    Calculates keyword match score for a given text chunk based on keyword occurrences.
    Converts text and keywords to lowercase and handles empty keyword lists safely.

    Args:
        text: Target document chunk text.
        keywords: List of keyword strings to search for.

    Returns:
        Float score between 0.0 and 1.0 (proportion of keywords matched).
    """
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0
    if not keywords or not isinstance(keywords, (list, tuple, set)):
        return 0.0

    clean_keywords = [str(kw).strip().lower() for kw in keywords if kw and str(kw).strip()]
    if not clean_keywords:
        return 0.0

    lower_text = text.lower()
    unique_kws = list(dict.fromkeys(clean_keywords))

    matches = sum(1 for kw in unique_kws if kw in lower_text)
    return round(matches / len(unique_kws), 4)


# ==============================================================================
# 4. HYBRID RANKING & RETRIEVAL PIPELINE
# ==============================================================================

def hybrid_rank(
    vector_results: List[Dict[str, Any]],
    keywords: List[str],
    vector_weight: float = 0.8,
    keyword_weight: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Combines vector similarity scores and keyword match scores using a weighted linear combination:
      hybrid_score = (vector_weight * vector_score) + (keyword_weight * keyword_score)

    Preserves original vector similarity scores and sorts results descending by hybrid_score.

    Args:
        vector_results: List of result dicts returned by retrieve().
        keywords: List of keyword strings for exact lexical matching.
        vector_weight: Weight float for vector score (default: 0.8).
        keyword_weight: Weight float for keyword score (default: 0.2).

    Returns:
        List of result dicts containing original vector score, keyword score,
        hybrid score, text, metadata, source, and chunk_index.
    """
    if not vector_results:
        return []

    hybrid_results = []
    for item in vector_results:
        vec_score = item.get("similarity_score") if "similarity_score" in item else item.get("score", 0.0)
        text = item.get("chunk_text") or item.get("text", "")
        kw_score = keyword_score(text, keywords)

        h_score = (vector_weight * vec_score) + (keyword_weight * kw_score)

        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        source = item.get("source") or meta.get("source", "unknown")
        chunk_idx = item.get("chunk_index") if item.get("chunk_index") is not None else meta.get("chunk_index", 1)

        hybrid_results.append({
            "rank": 0,  # Will be re-assigned after sorting
            "vector_score": round(float(vec_score), 4),
            "similarity_score": round(float(vec_score), 4),
            "keyword_score": round(float(kw_score), 4),
            "hybrid_score": round(float(h_score), 4),
            "chunk_text": str(text),
            "source": str(source),
            "chunk_index": chunk_idx,
            "section": str(meta.get("section") or item.get("section", "General")),
            "metadata": meta,
            "embedding_model": item.get("embedding_model"),
            "document_id": item.get("document_id")
        })

    # Sort descending by hybrid_score
    hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    for idx, res in enumerate(hybrid_results, start=1):
        res["rank"] = idx

    return hybrid_results


def hybrid_retrieve(
    query: str,
    keywords: List[str],
    k: int = 3,
    metadata_filter: Optional[Dict[str, Any]] = None,
    vector_weight: float = 0.8,
    keyword_weight: float = 0.2,
    collection: Optional[VectorCollection] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Executes end-to-end hybrid retrieval:
      User Query -> Query Embedding -> Metadata Filter -> Vector Search -> Keyword Match -> Hybrid Rank -> Top-K.

    Args:
        query: User search query string.
        keywords: List of keyword strings for exact lexical boosting.
        k: Top-k count to return.
        metadata_filter: Optional metadata filtering dictionary.
        vector_weight: Float weight for vector score (default 0.8).
        keyword_weight: Float weight for keyword score (default 0.2).
        collection: Optional VectorCollection instance.
        client: Optional embedding client instance.
        model: Optional embedding model name.

    Returns:
        List of hybrid ranked result dicts.
    """
    total_chunks = collection.count() if collection else 100
    candidate_k = max(k, min(total_chunks, k * 3))

    vec_results = retrieve(
        query=query,
        k=candidate_k,
        metadata_filter=metadata_filter,
        collection=collection,
        client=client,
        model=model
    )

    ranked = hybrid_rank(
        vector_results=vec_results,
        keywords=keywords,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight
    )

    return ranked[:k]


# ==============================================================================
# 5. DEMONSTRATION & DEBUG LOGGING FORMATTERS
# ==============================================================================

def get_retrieval_debug_trace(
    query: str,
    candidates: List[Dict[str, Any]],
    final_selected: List[Dict[str, Any]],
    model_name: str = "text-embedding-3-small",
    score_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generates a structured development/debug trace object recording retrieval quality metrics,
    candidate pool scores, ranks, and final selected chunks.

    Args:
        query: Search query string.
        candidates: Candidate chunks from Stage 1 retrieval.
        final_selected: Final Top-K chunks selected after relevance filtering/reranking.
        model_name: Embedding model used.
        score_threshold: Optional score threshold applied.

    Returns:
        Structured debug dictionary.
    """
    cand_details = []
    for idx, c in enumerate(candidates, start=1):
        score = c.get("similarity_score") if "similarity_score" in c else c.get("score", 0.0)
        source = c.get("source") or c.get("metadata", {}).get("source", "unknown")
        chunk_idx = c.get("chunk_index") or c.get("metadata", {}).get("chunk_index", idx)
        cand_details.append({
            "rank": idx,
            "document_chunk": f"{source} (Chunk {chunk_idx})",
            "similarity_score": round(float(score), 4),
            "rerank_score": c.get("rerank_score"),
            "text_snippet": (c.get("chunk_text") or c.get("text") or "")[:100]
        })

    selected_details = []
    for idx, c in enumerate(final_selected, start=1):
        score = c.get("similarity_score") if "similarity_score" in c else c.get("score", 0.0)
        source = c.get("source") or c.get("metadata", {}).get("source", "unknown")
        chunk_idx = c.get("chunk_index") or c.get("metadata", {}).get("chunk_index", idx)
        selected_details.append({
            "final_rank": idx,
            "original_rank": c.get("original_rank", idx),
            "document_chunk": f"{source} (Chunk {chunk_idx})",
            "similarity_score": round(float(score), 4),
            "rerank_score": c.get("rerank_score")
        })

    return {
        "query": query,
        "embedding_model": model_name,
        "candidate_count": len(candidates),
        "selected_count": len(final_selected),
        "score_threshold": score_threshold,
        "candidate_pool": cand_details,
        "final_selected_chunks": selected_details
    }


def format_retrieval_debug_trace(trace: Dict[str, Any]) -> str:
    """Formats structured retrieval debug trace object into human-readable report."""
    lines = [
        "========================================",
        "RETRIEVAL QUALITY DEBUG TRACE",
        "========================================",
        f"Query: {trace.get('query')}",
        f"Embedding Model: {trace.get('embedding_model')}",
        f"Candidate Count: {trace.get('candidate_count')}",
        f"Selected Count: {trace.get('selected_count')}",
        f"Score Threshold: {trace.get('score_threshold', 'None')}",
        "",
        "--- Candidate Pool ---"
    ]

    for cand in trace.get("candidate_pool", []):
        r_str = f" | Rerank Score: {cand['rerank_score']}" if cand.get("rerank_score") is not None else ""
        lines.append(
            f"Rank {cand['rank']}: {cand['document_chunk']} | Similarity: {cand['similarity_score']:.4f}{r_str}"
        )

    lines.append("")
    lines.append("--- Final Selected Chunks ---")
    for sel in trace.get("final_selected_chunks", []):
        lines.append(
            f"Final Rank {sel['final_rank']} (Orig: {sel.get('original_rank', 'N/A')}): {sel['document_chunk']} | Score: {sel['similarity_score']:.4f}"
        )

    lines.append("========================================")
    return "\n".join(lines)


def format_retrieval_output(
    query: str,
    results: List[Dict[str, Any]],
    model_name: str = "text-embedding-3-small",
    k: int = 3
) -> str:
    """Formats top-k retrieval results into clean ASCII text report."""
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


def format_filtered_vs_unfiltered_output(
    query: str,
    unfiltered_results: List[Dict[str, Any]],
    filtered_results: List[Dict[str, Any]],
    metadata_filter: Dict[str, Any]
) -> str:
    """
    Formats side-by-side comparative ASCII report for Unfiltered vs Filtered retrieval.
    """
    lines = []
    lines.append("========================================")
    lines.append("UNFILTERED RESULTS")
    lines.append("==================")
    lines.append("")

    if not unfiltered_results:
        lines.append("No unfiltered chunks retrieved.")
    else:
        for item in unfiltered_results:
            lines.append(f"Rank: {item['rank']}")
            lines.append(f"Score: {item['similarity_score']:.4f}")
            lines.append(f"Source: {item['source']}")
            lines.append(f"Section: {item['section']}")
            lines.append(f"Chunk Index: {item['chunk_index']}")
            lines.append(f"Text: {item['chunk_text']}")
            lines.append("")

    lines.append("========================================")
    lines.append(f"FILTERED RESULTS (Filter: {metadata_filter})")
    lines.append("====================")
    lines.append("")

    if not filtered_results:
        lines.append("No filtered chunks retrieved matching criteria.")
    else:
        for item in filtered_results:
            lines.append(f"Rank: {item['rank']}")
            lines.append(f"Score: {item['similarity_score']:.4f}")
            lines.append(f"Source: {item['source']}")
            lines.append(f"Section: {item['section']}")
            lines.append(f"Chunk Index: {item['chunk_index']}")
            lines.append(f"Text: {item['chunk_text']}")
            lines.append("")

    return "\n".join(lines).strip()


def format_hybrid_output(
    query: str,
    keywords: List[str],
    results: List[Dict[str, Any]],
    vector_weight: float = 0.8,
    keyword_weight: float = 0.2
) -> str:
    """
    Formats top-k hybrid search results into clean ASCII text report displaying
    vector score, keyword score, hybrid score, text, and metadata.
    """
    lines = []
    lines.append("========================================")
    lines.append("HYBRID SEARCH RESULTS")
    lines.append("=====================")
    lines.append("")
    lines.append(f"Query: {query}")
    lines.append(f"Keywords: {keywords}")
    lines.append(f"Weights: Vector ({vector_weight:.1f}) | Keyword ({keyword_weight:.1f})")
    lines.append("")

    if not results:
        lines.append("No hybrid search results returned.")
        return "\n".join(lines)

    for item in results:
        lines.append(f"Rank: {item['rank']}")
        lines.append(f"Vector Score: {item['vector_score']:.4f}")
        lines.append(f"Keyword Score: {item['keyword_score']:.4f}")
        lines.append(f"Hybrid Score: {item['hybrid_score']:.4f}")
        lines.append(f"Source: {item['source']}")
        lines.append(f"Section: {item['section']}")
        lines.append(f"Chunk Index: {item['chunk_index']}")
        lines.append(f"Text: {item['chunk_text']}")
        lines.append("")

    return "\n".join(lines).strip()


# ==============================================================================
# 6. DEMONSTRATIONS RUNNER (FILTERED VS UNFILTERED, HYBRID & EXACT-MATCH)
# ==============================================================================

def run_retrieval_demonstration() -> Dict[str, Any]:
    """
    Runs top-k retrieval, metadata filtering, hybrid search, and exact-match term
    demonstrations, prints outputs, and writes artifacts to disk.
    """
    load_dotenv()
    model_name = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    # Load vector collection
    collection = load_indexed_vector_collection()

    demo_artifacts: Dict[str, Any] = {
        "embedding_model": model_name,
        "total_indexed_chunks": collection.count(),
        "unfiltered_vs_filtered_demo": {},
        "hybrid_demo": {},
        "exact_match_demo": []
    }

    report_text_blocks = []

    print("========================================================================")
    print("   SHIPRULE CDLP - METADATA FILTERING & HYBRID SEARCH DEMO             ")
    print("========================================================================\n")

    # --------------------------------------------------------------------------
    # DEMO 1: UNFILTERED VS FILTERED RETRIEVAL
    # --------------------------------------------------------------------------
    query1 = "What are the password reset steps?"
    meta_filter1 = {"section": "Account access"}

    # Unfiltered retrieval
    unfiltered_res = retrieve(query1, k=3, metadata_filter=None, collection=collection)

    # Filtered retrieval (falls back cleanly if exact section is not in default sample)
    filtered_res = retrieve(query1, k=3, metadata_filter=meta_filter1, collection=collection)

    # If sample corpus didn't contain "Account access", demonstrate filter on existing metadata
    if not filtered_res:
        real_filter = {"source": "customs_requirements.txt"}
        filtered_res = retrieve(query1, k=3, metadata_filter=real_filter, collection=collection)
        fmt_filter_demo = format_filtered_vs_unfiltered_output(query1, unfiltered_res, filtered_res, real_filter)
    else:
        fmt_filter_demo = format_filtered_vs_unfiltered_output(query1, unfiltered_res, filtered_res, meta_filter1)

    print(fmt_filter_demo)
    report_text_blocks.append(fmt_filter_demo)
    report_text_blocks.append("\n" + "=" * 50 + "\n")

    demo_artifacts["unfiltered_vs_filtered_demo"] = {
        "query": query1,
        "unfiltered": unfiltered_res,
        "filtered": filtered_res
    }

    # --------------------------------------------------------------------------
    # DEMO 2: HYBRID SEARCH RETRIEVAL
    # --------------------------------------------------------------------------
    query2 = "What are the password reset steps?"
    keywords2 = ["password", "reset"]

    hybrid_res = hybrid_retrieve(
        query=query2,
        keywords=keywords2,
        k=3,
        vector_weight=0.8,
        keyword_weight=0.2,
        collection=collection
    )

    fmt_hybrid_demo = format_hybrid_output(query2, keywords2, hybrid_res, vector_weight=0.8, keyword_weight=0.2)
    print(fmt_hybrid_demo)
    report_text_blocks.append(fmt_hybrid_demo)
    report_text_blocks.append("\n" + "=" * 50 + "\n")

    demo_artifacts["hybrid_demo"] = {
        "query": query2,
        "keywords": keywords2,
        "results": hybrid_res
    }

    # --------------------------------------------------------------------------
    # DEMO 3: EXACT-MATCH TERMINOLOGY TESTS
    # --------------------------------------------------------------------------
    exact_match_cases = [
        {
            "query": "What import document and BIS Registration Certificate are needed for laptops in India?",
            "keywords": ["BIS", "registration", "laptops", "India"],
            "filter": {"source": "customs_requirements.txt"}
        },
        {
            "query": "Explain Incoterms 2020 FOB and CIF terms for shipping",
            "keywords": ["Incoterms", "FOB", "CIF"],
            "filter": {"document_type": "pdf"}
        }
    ]

    for em_case in exact_match_cases:
        q = em_case["query"]
        kws = em_case["keywords"]
        flt = em_case["filter"]

        v_only = retrieve(q, k=3, metadata_filter=None, collection=collection)
        v_filtered = retrieve(q, k=3, metadata_filter=flt, collection=collection)
        h_res = hybrid_retrieve(q, keywords=kws, k=3, metadata_filter=flt, collection=collection)

        fmt_em = format_hybrid_output(f"{q} (Filtered: {flt})", kws, h_res)
        print(fmt_em)
        report_text_blocks.append(fmt_em)
        report_text_blocks.append("\n" + "=" * 50 + "\n")

        demo_artifacts["exact_match_demo"].append({
            "query": q,
            "keywords": kws,
            "filter": flt,
            "vector_only": v_only,
            "vector_filtered": v_filtered,
            "hybrid_results": h_res
        })

    # Save artifacts to outputs/
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "retrieval_demo_output.json")
    text_path = os.path.join(output_dir, "retrieval_output.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(demo_artifacts, f, indent=2, ensure_ascii=False)

    full_text_report = "\n".join(report_text_blocks)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full_text_report)

    print(f"[SUCCESS] Metadata filtering & hybrid search artifacts saved:")
    print(f"  JSON: {json_path}")
    print(f"  TXT : {text_path}")
    print("========================================================================\n")

    return demo_artifacts


if __name__ == "__main__":
    run_retrieval_demonstration()
