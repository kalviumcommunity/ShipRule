"""
ShipRule CDLP - Two-Stage Retrieval & Chunk Re-Ranking Module
=============================================================
Implements a two-stage retrieval pipeline:
1. Stage 1: Vector search retrieves a larger candidate pool (candidate_k=10).
2. Stage 2: Re-ranker scores candidate relevance to the query (rerank_score()).
3. Final Stage: Returns top final_k chunks sorted by rerank_score descending.

Preserves original vector similarity scores, tracks rank movement (Original Rank -> New Rank),
supports metadata filtering, and provides modular abstractions for future cross-encoder models.
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Union

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# UTF-8 stdout configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.rag.indexing import VectorCollection
from app.rag.retrieval import retrieve, load_indexed_vector_collection


# ==============================================================================
# 1. RE-RANKER SCORING ABSTRACTION
# ==============================================================================

def rerank_score(
    query: str,
    chunk_text: str,
    vector_score: float = 0.0
) -> float:
    """
    Computes a relevance score for a single chunk text given a user query.
    Combines semantic vector similarity with query-chunk term density and n-gram overlap.

    Note: This modular function abstracts the relevance scoring logic so it can later
    be replaced with a neural cross-encoder model (e.g. MS-MARCO MiniLM) or reranker API.

    Args:
        query: User search query string.
        chunk_text: Target document chunk text.
        vector_score: Vector similarity score from Stage 1 retrieval.

    Returns:
        Float relevance score (higher is more relevant).
    """
    if not query or not isinstance(query, str) or not query.strip():
        return 0.0
    if not chunk_text or not isinstance(chunk_text, str) or not chunk_text.strip():
        return 0.0

    clean_query = query.strip().lower()
    clean_text = chunk_text.strip().lower()

    # Extract clean query terms (length > 2 to ignore trivial stopwords)
    query_terms = [t for t in clean_query.split() if len(t) > 2]
    if not query_terms:
        query_terms = clean_query.split()

    unique_terms = list(dict.fromkeys(query_terms))
    matches = sum(1 for t in unique_terms if t in clean_text)
    term_coverage = matches / len(unique_terms) if unique_terms else 0.0

    # Calculate phrase or contiguous 2-gram overlap bonus
    phrase_bonus = 0.0
    if len(query_terms) >= 2:
        bigrams = [f"{query_terms[i]} {query_terms[i+1]}" for i in range(len(query_terms)-1)]
        bigram_matches = sum(1 for bg in bigrams if bg in clean_text)
        phrase_bonus = (bigram_matches / len(bigrams)) if bigrams else 0.0

    # Combine vector score (weight 0.75), term coverage (weight 0.18), and phrase bonus (weight 0.07)
    # Vector similarity is the primary semantic driver so paraphrased queries without exact word matches maintain high scores
    vec_component = max(0.0, float(vector_score))
    combined_score = (0.75 * vec_component) + (0.18 * term_coverage) + (0.07 * phrase_bonus)

    # Scale score to 0-10 for clear readability
    final_score = round(combined_score * 10.0, 2)
    return final_score


# ==============================================================================
# 2. RE-RANKER FUNCTION
# ==============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    final_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Re-ranks a list of candidate chunks for relevance to the user query and returns top final_k.

    Args:
        query: Non-empty search query string.
        candidates: List of candidate chunk dicts returned by Stage 1 vector retrieval.
        final_k: Positive integer specifying number of top chunks to return.

    Returns:
        List of re-ranked result dicts containing original vector score, rerank_score,
        original_rank, text, source, chunk_index, and metadata.
    """
    if query is None or not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string for re-ranking.")

    if isinstance(final_k, bool) or not isinstance(final_k, int) or final_k <= 0:
        raise ValueError("Parameter 'final_k' must be a positive integer greater than 0.")

    if not candidates or not isinstance(candidates, list):
        return []

    reranked_candidates = []
    for orig_idx, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            continue

        text = item.get("chunk_text") or item.get("text", "")
        vec_score = item.get("similarity_score") if "similarity_score" in item else item.get("score", 0.0)

        # Calculate re-rank score
        r_score = rerank_score(query=query, chunk_text=text, vector_score=vec_score)

        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        source = item.get("source") or meta.get("source", "unknown")
        chunk_idx = item.get("chunk_index") if item.get("chunk_index") is not None else meta.get("chunk_index", orig_idx)

        rec = {
            "rank": 0,  # Will be assigned after sorting
            "original_rank": orig_idx,
            "vector_score": round(float(vec_score), 4),
            "similarity_score": round(float(vec_score), 4),
            "rerank_score": r_score,
            "chunk_text": str(text),
            "source": str(source),
            "chunk_index": chunk_idx,
            "section": str(meta.get("section") or item.get("section", "General")),
            "metadata": meta,
            "embedding_model": item.get("embedding_model"),
            "document_id": item.get("document_id") or item.get("id")
        }
        reranked_candidates.append(rec)

    # Sort descending by rerank_score
    reranked_candidates.sort(key=lambda x: (x["rerank_score"], x["vector_score"]), reverse=True)

    # Slice top final_k and assign final rank
    top_final = reranked_candidates[:final_k]
    for final_idx, res in enumerate(top_final, start=1):
        res["rank"] = final_idx

    return top_final


# ==============================================================================
# 3. TWO-STAGE RETRIEVAL PIPELINE FUNCTION
# ==============================================================================

def retrieve_and_rerank(
    query: str,
    candidate_k: int = 10,
    final_k: int = 3,
    metadata_filter: Optional[Dict[str, Any]] = None,
    collection: Optional[VectorCollection] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes two-stage retrieval:
      User Query -> Stage 1: Vector Search (candidate_k) -> Stage 2: Re-ranking -> Final Top-K (final_k).

    Args:
        query: User query string.
        candidate_k: Number of candidates to retrieve in Stage 1 (must be > final_k).
        final_k: Final top-k context size to return (must be < candidate_k).
        metadata_filter: Optional metadata filtering criteria applied during Stage 1.
        collection: Optional VectorCollection instance.
        client: Optional embedding client instance.
        model: Optional embedding model name.

    Returns:
        Tuple of (stage1_candidates, final_reranked_top_k, timing_metrics_dict)
    """
    # 1. Parameter Validations
    if query is None or not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k <= 0:
        raise ValueError("Parameter 'candidate_k' must be a positive integer greater than 0.")

    if isinstance(final_k, bool) or not isinstance(final_k, int) or final_k <= 0:
        raise ValueError("Parameter 'final_k' must be a positive integer greater than 0.")

    if candidate_k <= final_k:
        raise ValueError(
            f"Invalid parameters: candidate_k ({candidate_k}) must be strictly greater than final_k ({final_k})."
        )

    if collection is None:
        collection = load_indexed_vector_collection()

    # 2. Stage 1: Vector Retrieval (candidate_k)
    t0 = time.perf_counter()
    candidates = retrieve(
        query=query,
        k=candidate_k,
        metadata_filter=metadata_filter,
        collection=collection,
        client=client,
        model=model
    )
    t1 = time.perf_counter()
    vector_retrieval_ms = round((t1 - t0) * 1000, 2)

    # 3. Stage 2: Re-Ranking (final_k)
    t2 = time.perf_counter()
    reranked_top_k = rerank(query=query, candidates=candidates, final_k=final_k)
    t3 = time.perf_counter()
    reranking_ms = round((t3 - t2) * 1000, 2)
    total_ms = round((t3 - t0) * 1000, 2)

    metrics = {
        "candidate_k": candidate_k,
        "final_k": final_k,
        "retrieved_candidates_count": len(candidates),
        "final_top_k_count": len(reranked_top_k),
        "vector_retrieval_time_ms": vector_retrieval_ms,
        "reranking_time_ms": reranking_ms,
        "total_pipeline_time_ms": total_ms
    }

    return candidates, reranked_top_k, metrics


# ==============================================================================
# 4. OUTPUT COMPARISON FORMATTERS
# ==============================================================================

def format_before_after_comparison(
    query: str,
    before_candidates: List[Dict[str, Any]],
    after_candidates: List[Dict[str, Any]]
) -> str:
    """Formats side-by-side comparative ASCII report for BEFORE vs AFTER re-ranking."""
    lines = []
    lines.append("========================================")
    lines.append("BEFORE RE-RANKING (Top-3 Initial Vector Candidates)")
    lines.append("========================================")
    lines.append(f"Query: {query}\n")

    top3_before = before_candidates[:3]
    if not top3_before:
        lines.append("No initial candidates retrieved.")
    else:
        for idx, item in enumerate(top3_before, start=1):
            score = item.get("similarity_score") if "similarity_score" in item else item.get("score", 0.0)
            lines.append(f"Rank: {idx}")
            lines.append(f"Vector Score: {score:.4f}")
            lines.append(f"Source: {item.get('source', 'unknown')}")
            lines.append(f"Section: {item.get('section', 'General')}")
            lines.append(f"Chunk Index: {item.get('chunk_index', 1)}")
            lines.append(f"Text: {item.get('chunk_text', item.get('text', ''))[:140]}...")
            lines.append("")

    lines.append("========================================")
    lines.append("AFTER RE-RANKING (Top-3 Final Re-Ranked Context)")
    lines.append("========================================")
    lines.append(f"Query: {query}\n")

    top3_after = after_candidates[:3]
    if not top3_after:
        lines.append("No re-ranked candidates returned.")
    else:
        for item in top3_after:
            lines.append(f"Rank: {item['rank']}")
            lines.append(f"Vector Score: {item['vector_score']:.4f}")
            lines.append(f"Re-Rank Score: {item['rerank_score']:.1f}")
            lines.append(f"Source: {item['source']}")
            lines.append(f"Section: {item.get('section', 'General')}")
            lines.append(f"Chunk Index: {item['chunk_index']}")
            lines.append(f"Text: {item['chunk_text'][:140]}...")
            lines.append("")

    return "\n".join(lines).strip()


def format_rank_movement_report(
    after_candidates: List[Dict[str, Any]]
) -> str:
    """Formats rank movement summary displaying Original Rank -> New Rank for re-ranked chunks."""
    lines = []
    lines.append("========================================")
    lines.append("CANDIDATE RANK MOVEMENT TRACKING")
    lines.append("========================================")
    lines.append(f"{'Original Rank':<16} -> {'New Rank':<10} {'Source':<32} {'Re-Rank Score':<12}")
    lines.append("-" * 75)

    for item in after_candidates:
        orig_r = item.get("original_rank", "N/A")
        new_r = item.get("rank", "N/A")
        source = item.get("source", "unknown")
        score = item.get("rerank_score", 0.0)
        lines.append(f"Rank {orig_r:<11} -> Rank {new_r:<5} {source:<32} {score:<12.1f}")

    lines.append("=" * 75)
    return "\n".join(lines).strip()
