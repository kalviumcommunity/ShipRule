"""
ShipRule CDLP - Retrieval Quality Guardrail Module
==================================================
Evaluates candidate retrieval quality before answer generation to prevent hallucinations
and unsupported LLM completions on weak or missing context.

Key Capabilities:
1. Detects empty retrieval (Case A: no retrieved chunks -> 'no_results').
2. Evaluates chunk relevance against distance and similarity thresholds (Case B: 'no_relevant_context').
3. Enforces minimum relevant chunk count constraints (Case C: 'too_few_relevant_chunks').
4. Intercepts weak retrieval before LLM invocation to return deterministic safe refusals.
5. Emits structured guardrail evaluation payloads and standardized ASCII audit reports.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Default Guardrail Configuration Thresholds
DEFAULT_MAX_DISTANCE_THRESHOLD = 1.35
DEFAULT_MIN_RELEVANT_CHUNKS = 1

SAFE_REFUSAL_MESSAGE = (
    "I could not find enough relevant information in the available knowledge base to answer this question. "
    "I don't know based on the available documents. "
    "The provided context is insufficient to answer this question."
)

POLITE_INSUFFICIENT_CONTEXT_MESSAGE = (
    "The provided context is insufficient to answer this question."
)




def get_guardrail_config() -> Tuple[float, int]:
    """
    Loads guardrail configuration thresholds from environment variables with safe defaults.

    Returns:
        Tuple of (max_distance_threshold, min_relevant_chunks).
    """
    load_dotenv()
    try:
        max_dist = float(os.getenv("MAX_DISTANCE_THRESHOLD", str(DEFAULT_MAX_DISTANCE_THRESHOLD)))
    except (ValueError, TypeError):
        max_dist = DEFAULT_MAX_DISTANCE_THRESHOLD

    try:
        min_chunks = int(os.getenv("MIN_RELEVANT_CHUNKS", str(DEFAULT_MIN_RELEVANT_CHUNKS)))
    except (ValueError, TypeError):
        min_chunks = DEFAULT_MIN_RELEVANT_CHUNKS

    return max_dist, min_chunks


def is_chunk_relevant(
    chunk: Dict[str, Any],
    max_distance_threshold: float = DEFAULT_MAX_DISTANCE_THRESHOLD
) -> bool:
    """
    Determines if an individual chunk meets relevance criteria based on existing scoring metadata.

    Supports:
    - ChromaDB distance (L2 or Cosine distance: lower distance is better, distance <= threshold).
    - Re-ranker score (0-10 scale: higher is better, rerank_score >= 0.5).
    - Cosine similarity / vector_score (higher is better, score >= 0.0 or distance <= threshold).

    Args:
        chunk: Chunk dictionary containing scoring information.
        max_distance_threshold: Maximum acceptable distance threshold (default: 1.35).

    Returns:
        Boolean indicating whether the chunk is relevant.
    """
    if not isinstance(chunk, dict):
        return False

    # 1. Check direct distance metric (if explicitly provided by ChromaDB / retrieval)
    # If distance is explicitly present: lower is closer. distance > threshold is IRRELEVANT.
    if "distance" in chunk and chunk["distance"] is not None:
        try:
            dist = float(chunk["distance"])
            return dist <= max_distance_threshold
        except (ValueError, TypeError):
            pass

    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    if "distance" in meta and meta["distance"] is not None:
        try:
            dist = float(meta["distance"])
            return dist <= max_distance_threshold
        except (ValueError, TypeError):
            pass

    # 2. Check rerank_score (0-10 scale where higher is better)
    rerank_sc = chunk.get("rerank_score")
    if rerank_sc is not None:
        try:
            r_val = float(rerank_sc)
            if r_val >= 0.5:
                return True
        except (ValueError, TypeError):
            pass

    # 3. Check similarity_score / vector_score / score
    score = chunk.get("similarity_score")
    if score is None:
        score = chunk.get("vector_score")
    if score is None and "score" in chunk:
        score = chunk.get("score")

    if score is not None:
        try:
            val = float(score)
            # If value represents distance (> 1.0)
            if val > 1.0:
                return val <= max_distance_threshold

            # For similarity score in [-1.0, 1.0]:
            # Cosine distance = 1.0 - val. Must satisfy (1.0 - val) <= max_distance_threshold.
            # For threshold 1.35, val >= -0.35 is accepted; val < -0.35 is rejected.
            equiv_dist = 1.0 - val
            return equiv_dist <= max_distance_threshold
        except (ValueError, TypeError):
            pass

    # 4. Default: If text exists and no distance/score violation, consider relevant
    text = chunk.get("chunk_text") or chunk.get("text") or meta.get("chunk_text")
    if text and str(text).strip():
        return True

    return False


def check_retrieval_quality(
    retrieved_chunks: Optional[List[Dict[str, Any]]],
    max_distance_threshold: Optional[float] = None,
    min_relevant_chunks: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluates the quality and relevance of retrieved candidate chunks.

    Classifies retrieval into:
    - 'no_results': When 0 chunks were retrieved (retrieved_chunks is empty).
    - 'no_relevant_context': When chunks were retrieved, but none passed the relevance threshold.
    - 'too_few_relevant_chunks': When some chunks passed, but fewer than min_relevant_chunks.
    - 'strong_retrieval': When at least min_relevant_chunks passed the threshold.

    Args:
        retrieved_chunks: List of candidate chunk dicts from vector retrieval.
        max_distance_threshold: Max allowed distance threshold (defaults to env / 1.35).
        min_relevant_chunks: Minimum number of required relevant chunks (defaults to env / 1).

    Returns:
        Structured evaluation dictionary:
        {
            "is_sufficient": bool,
            "reason": str,
            "retrieved_count": int,
            "relevant_count": int,
            "threshold": float,
            "min_relevant_chunks": int,
            "decision": str ("ALLOW" | "REFUSE"),
            "relevant_chunks": List[Dict[str, Any]],
            "report": str
        }
    """
    env_dist, env_min_chunks = get_guardrail_config()
    threshold = max_distance_threshold if max_distance_threshold is not None else env_dist
    min_chunks = min_relevant_chunks if min_relevant_chunks is not None else env_min_chunks

    if not retrieved_chunks or not isinstance(retrieved_chunks, list):
        report = format_guardrail_report(
            is_sufficient=False,
            retrieved_count=0,
            relevant_count=0,
            threshold=threshold,
            min_relevant_chunks=min_chunks,
            reason="no_results",
            decision="REFUSE"
        )
        return {
            "is_sufficient": False,
            "reason": "no_results",
            "retrieved_count": 0,
            "relevant_count": 0,
            "threshold": threshold,
            "min_relevant_chunks": min_chunks,
            "decision": "REFUSE",
            "relevant_chunks": [],
            "report": report
        }

    retrieved_count = len(retrieved_chunks)
    relevant_chunks = [c for c in retrieved_chunks if is_chunk_relevant(c, max_distance_threshold=threshold)]
    relevant_count = len(relevant_chunks)

    if relevant_count == 0:
        reason = "no_relevant_context"
        is_sufficient = False
        decision = "REFUSE"
    elif relevant_count < min_chunks:
        reason = "too_few_relevant_chunks"
        is_sufficient = False
        decision = "REFUSE"
    else:
        reason = "strong_retrieval"
        is_sufficient = True
        decision = "ALLOW"

    report = format_guardrail_report(
        is_sufficient=is_sufficient,
        retrieved_count=retrieved_count,
        relevant_count=relevant_count,
        threshold=threshold,
        min_relevant_chunks=min_chunks,
        reason=reason,
        decision=decision
    )

    return {
        "is_sufficient": is_sufficient,
        "reason": reason,
        "retrieved_count": retrieved_count,
        "relevant_count": relevant_count,
        "threshold": threshold,
        "min_relevant_chunks": min_chunks,
        "decision": decision,
        "relevant_chunks": relevant_chunks,
        "report": report
    }


def format_guardrail_report(
    is_sufficient: bool,
    retrieved_count: int,
    relevant_count: int,
    threshold: float,
    min_relevant_chunks: int,
    reason: str,
    decision: str
) -> str:
    """Formats standardized ASCII retrieval guardrail report string."""
    quality_label = "SUFFICIENT" if is_sufficient else "INSUFFICIENT"
    lines = [
        "[Retrieval Guardrail]",
        f"Retrieval Quality    : {quality_label}",
        f"Retrieved Chunks     : {retrieved_count}",
        f"Relevant Chunks      : {relevant_count}",
        f"Relevance Threshold  : {threshold}",
        f"Min Required Chunks  : {min_relevant_chunks}",
        f"Reason Code          : {reason}",
        f"Guardrail Decision   : {decision}"
    ]
    return "\n".join(lines)


def get_safe_refusal_response(
    guardrail_result: Dict[str, Any],
    query: str = "",
    custom_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs a deterministic, safe refusal payload when retrieval quality is insufficient.
    Guarantees no LLM call, no hallucinated content, and no fabricated citations.
    """
    message = custom_message or SAFE_REFUSAL_MESSAGE
    return {
        "answer": message,
        "sources": [],
        "retrieved_chunks": [],
        "retrieved_count": guardrail_result.get("retrieved_count", 0),
        "relevant_count": guardrail_result.get("relevant_count", 0),
        "guardrail_result": guardrail_result,
        "llm_called": False,
        "citation_registry": {},
        "citation_mapping": {},
        "citation_verification": {
            "is_valid": True,
            "verification_status": "PASS",
            "grounding_status": "NO RETRIEVAL CONTEXT",
            "used_citations": [],
            "available_citations": [],
            "unsupported_citations": [],
            "verified_details": [],
            "report": "[Source Verification]\nCitations           : NONE\nVerification Status : PASS\nGrounding Status    : NO RETRIEVAL CONTEXT"
        }
    }
