"""
ShipRule CDLP - Retrieval Relevance Tuning & Evaluation Module
===============================================================
Provides an evaluation framework for testing and tuning retrieval quality across
multiple configurations (Baseline Top-K, Metadata Filtered, Score Threshold, Hybrid Search).
Computes empirical metrics: Hit Rate, Hits/Misses, MRR (Mean Reciprocal Rank), Recall@K,
and Average Rank of expected sources.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# UTF-8 stdout configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.indexing import VectorCollection
from src.retrieval import (
    retrieve,
    hybrid_retrieve,
    load_indexed_vector_collection
)

# ==============================================================================
# 1. EVALUATION DATASET (BUILT FROM ACTUAL INDEXED CORPUS DOCUMENTS)
# ==============================================================================

EVALUATION_DATASET: List[Dict[str, Any]] = [
    {
        "query": "What are the mandatory statutory registrations for importing IT electronics and telecommunications equipment?",
        "expected_source": "customs_requirements.txt",
        "keywords": ["statutory", "registrations", "electronics", "telecommunications"],
        "topic": "Statutory Registrations & BIS CRS"
    },
    {
        "query": "What standard Incoterms 2020 rules define FOB, CIF, and DDP shipping obligations?",
        "expected_source": "international_shipping_guide.pdf",
        "keywords": ["Incoterms", "FOB", "CIF", "DDP"],
        "topic": "Incoterms 2020 Division of Risk"
    },
    {
        "query": "What shipping documents are required for international customs clearance?",
        "expected_source": "shipping_rules.txt",
        "keywords": ["commercial", "invoice", "packing", "list"],
        "topic": "Mandatory Shipping Documentation"
    },
    {
        "query": "How are preferential tariff rates granted under Free Trade Agreements?",
        "expected_source": "customs_requirements.txt",
        "keywords": ["preferential", "tariff", "Certificate of Origin", "FTA"],
        "topic": "Preferential Tariffs & FTAs"
    },
    {
        "query": "How are Basic Customs Duty and HS Code classifications determined?",
        "expected_source": "customs_requirements.txt",
        "keywords": ["Basic Customs Duty", "HS", "code", "commodity"],
        "topic": "HS Code Tariff Assessment"
    }
]

# ==============================================================================
# 2. EVALUATION CONFIGURATIONS
# ==============================================================================

EVALUATION_CONFIGURATIONS: List[Dict[str, Any]] = [
    {
        "name": "baseline_k3",
        "label": "Baseline Top-K=3 (Unfiltered)",
        "k": 3,
        "filter": None,
        "min_score": 0.0,
        "use_hybrid": False
    },
    {
        "name": "baseline_k5",
        "label": "Baseline Top-K=5 (Unfiltered)",
        "k": 5,
        "filter": None,
        "min_score": 0.0,
        "use_hybrid": False
    },
    {
        "name": "filtered_k3",
        "label": "Metadata Filtered Top-K=3 (Source Filter)",
        "k": 3,
        "filter": {"source": "customs_requirements.txt"},
        "min_score": 0.0,
        "use_hybrid": False
    },
    {
        "name": "threshold_k5",
        "label": "Score Threshold Top-K=5 (Min Score 0.02)",
        "k": 5,
        "filter": None,
        "min_score": 0.02,
        "use_hybrid": False
    },
    {
        "name": "hybrid_k3",
        "label": "Hybrid Search Top-K=3 (Vector + Keyword)",
        "k": 3,
        "filter": None,
        "min_score": 0.0,
        "use_hybrid": True
    }
]

MANUAL_RELEVANCE_LEVELS = [
    "Excellent",
    "Relevant",
    "Partially Relevant",
    "Irrelevant",
    "Duplicate"
]


# ==============================================================================
# 3. CORE EVALUATION FUNCTION
# ==============================================================================

def evaluate_retrieval(
    setting: Dict[str, Any],
    test_dataset: Optional[List[Dict[str, Any]]] = None,
    collection: Optional[VectorCollection] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates a single retrieval configuration across all test queries in dataset.

    Args:
        setting: Configuration dict containing 'name', 'k', 'filter', 'min_score', 'use_hybrid'.
        test_dataset: Optional custom evaluation dataset (defaults to EVALUATION_DATASET).
        collection: Optional VectorCollection instance.
        client: Optional embedding client instance.
        model: Optional embedding model name.

    Returns:
        Structured evaluation dict with metrics (Hits, Misses, Hit Rate %, MRR, Recall@K, Avg Rank).
    """
    if not setting or not isinstance(setting, dict) or "k" not in setting:
        raise ValueError("Evaluation setting must be a valid configuration dictionary with 'k' key.")

    dataset = test_dataset if test_dataset is not None else EVALUATION_DATASET
    if not dataset or not isinstance(dataset, list):
        raise ValueError("Evaluation dataset must be a non-empty list of test query dicts.")

    if collection is None:
        collection = load_indexed_vector_collection()

    k = setting.get("k", 3)
    meta_filter = setting.get("filter")
    min_score = setting.get("min_score", 0.0)
    use_hybrid = setting.get("use_hybrid", False)
    use_reranking = setting.get("use_reranking", False)
    candidate_k = setting.get("candidate_k", 10)

    query_eval_results = []
    total_queries = len(dataset)
    hits = 0
    misses = 0
    reciprocal_ranks = []
    found_ranks = []

    for test_case in dataset:
        query = test_case.get("query", "")
        expected_source = test_case.get("expected_source", "")
        keywords = test_case.get("keywords", [])

        # 1. Perform Retrieval
        if use_reranking:
            from src.reranker import retrieve_and_rerank
            _, raw_results, _ = retrieve_and_rerank(
                query=query,
                candidate_k=candidate_k,
                final_k=k,
                metadata_filter=meta_filter,
                collection=collection,
                client=client,
                model=model
            )
        elif use_hybrid:
            raw_results = hybrid_retrieve(
                query=query,
                keywords=keywords,
                k=k,
                metadata_filter=meta_filter,
                collection=collection,
                client=client,
                model=model
            )
        else:
            raw_results = retrieve(
                query=query,
                k=k,
                metadata_filter=meta_filter,
                collection=collection,
                client=client,
                model=model
            )

        # 2. Apply Score Threshold Filter
        if use_reranking:
            score_key = "rerank_score"
        elif use_hybrid:
            score_key = "hybrid_score"
        else:
            score_key = "similarity_score"

        threshold_filtered_results = [
            res for res in raw_results
            if res.get(score_key, res.get("similarity_score", 0.0)) >= min_score
        ]

        # Re-assign rank after threshold filtering
        for idx, res in enumerate(threshold_filtered_results, start=1):
            res["eval_rank"] = idx

        returned_sources = [res.get("source", "unknown") for res in threshold_filtered_results]

        # 3. Check Hit/Miss and Rank of Expected Source
        hit = False
        expected_rank = None
        for res in threshold_filtered_results:
            if res.get("source") == expected_source:
                hit = True
                expected_rank = res.get("eval_rank")
                break

        if hit and expected_rank is not None:
            hits += 1
            found_ranks.append(expected_rank)
            reciprocal_ranks.append(1.0 / expected_rank)
        else:
            misses += 1
            reciprocal_ranks.append(0.0)

        # Build manual inspection judgements for retrieved chunks
        inspection_chunks = []
        for res in threshold_filtered_results:
            chunk_score = res.get(score_key, res.get("similarity_score", 0.0))
            is_expected = (res.get("source") == expected_source)

            # Rule-based initial manual judgement hint
            if is_expected and res.get("eval_rank") == 1 and chunk_score >= 0.5:
                judgement_hint = "Excellent"
            elif is_expected:
                judgement_hint = "Relevant"
            elif chunk_score >= 0.2:
                judgement_hint = "Partially Relevant"
            else:
                judgement_hint = "Irrelevant"

            inspection_chunks.append({
                "rank": res.get("eval_rank"),
                "score": chunk_score,
                "source": res.get("source"),
                "section": res.get("section", "General"),
                "chunk_index": res.get("chunk_index"),
                "text": res.get("chunk_text"),
                "judgement_hint": judgement_hint
            })

        query_eval_results.append({
            "query": query,
            "topic": test_case.get("topic", "General"),
            "expected_source": expected_source,
            "returned_sources": returned_sources,
            "hit": hit,
            "expected_rank": expected_rank,
            "retrieved_count": len(threshold_filtered_results),
            "inspection_chunks": inspection_chunks
        })

    # 4. Calculate Aggregate Metrics
    hit_rate_ratio = hits / total_queries if total_queries > 0 else 0.0
    hit_rate_pct = round(hit_rate_ratio * 100, 2)
    mrr = round(sum(reciprocal_ranks) / total_queries, 4) if total_queries > 0 else 0.0
    avg_rank = round(sum(found_ranks) / len(found_ranks), 2) if found_ranks else None
    recall_at_k = round(hits / total_queries, 4) if total_queries > 0 else 0.0

    return {
        "configuration": setting.get("name", "custom"),
        "label": setting.get("label", setting.get("name")),
        "k": k,
        "filter": meta_filter,
        "min_score": min_score,
        "use_hybrid": use_hybrid,
        "total_queries": total_queries,
        "hits": hits,
        "misses": misses,
        "hit_rate_pct": hit_rate_pct,
        "mrr": mrr,
        "recall_at_k": recall_at_k,
        "avg_expected_rank": avg_rank,
        "query_evaluations": query_eval_results
    }


# ==============================================================================
# 4. COMPARISON & RECOMMENDATION ENGINE
# ==============================================================================

def compare_configurations(
    configurations: Optional[List[Dict[str, Any]]] = None,
    test_dataset: Optional[List[Dict[str, Any]]] = None,
    collection: Optional[VectorCollection] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Evaluates all retrieval configurations, compiles comparative metrics summary table,
    and automatically selects best configuration based on empirical data.

    Selection Criteria Priority:
    1. Maximum Hit Rate (%)
    2. Lowest Average Rank of Expected Source
    3. Highest MRR (Mean Reciprocal Rank)
    4. Minimal Noise (Lower k preferred when hit rate is equal)

    Returns:
        Tuple of (all_eval_results, summary_table_dict, best_recommendation_dict)
    """
    configs = configurations if configurations is not None else EVALUATION_CONFIGURATIONS
    if collection is None:
        collection = load_indexed_vector_collection()

    eval_results = []
    for cfg in configs:
        res = evaluate_retrieval(cfg, test_dataset=test_dataset, collection=collection)
        eval_results.append(res)

    # Sort configurations to select best
    def recommendation_sort_key(res):
        hr = res["hit_rate_pct"]
        mrr = res["mrr"]
        # Inverse of avg rank (higher is better; handle None as 99)
        avg_r = res["avg_expected_rank"]
        rank_score = (1.0 / avg_r) if (avg_r is not None and avg_r > 0) else 0.0
        # Prefer smaller k when hit rates and MRR match to reduce noise
        k_penalty = 1.0 / res["k"]
        return (hr, mrr, rank_score, k_penalty)

    sorted_by_best = sorted(eval_results, key=recommendation_sort_key, reverse=True)
    best_config = sorted_by_best[0]

    # Build comparative summary table rows
    summary_rows = []
    for res in eval_results:
        summary_rows.append({
            "configuration": res["configuration"],
            "hits": res["hits"],
            "total": res["total_queries"],
            "hit_rate": f"{res['hit_rate_pct']:.0f}%",
            "mrr": f"{res['mrr']:.4f}",
            "avg_rank": f"{res['avg_expected_rank']:.2f}" if res["avg_expected_rank"] is not None else "N/A"
        })

    best_recommendation = {
        "recommended_configuration": best_config["configuration"],
        "recommended_label": best_config["label"],
        "best_k": best_config["k"],
        "best_filter": best_config["filter"],
        "best_min_score": best_config["min_score"],
        "use_hybrid": best_config["use_hybrid"],
        "hit_rate_pct": best_config["hit_rate_pct"],
        "mrr": best_config["mrr"],
        "avg_expected_rank": best_config["avg_expected_rank"],
        "reasons": [
            f"Achieved highest empirical hit rate ({best_config['hit_rate_pct']:.0f}% across test queries).",
            f"Maintained optimal Mean Reciprocal Rank (MRR = {best_config['mrr']:.4f}).",
            f"Delivered lowest average position for target sources (Avg Rank = {best_config['avg_expected_rank'] or 'N/A'}).",
            f"Balanced context precision and recall without introducing unnecessary context noise."
        ],
        "trade_offs": [
            f"Top-K={best_config['k']} caps retrieved chunks to prevent prompt bloat in generation phase.",
            f"Metadata filter {best_config['filter']} restricts corpus scope to increase precision."
        ]
    }

    return eval_results, {"rows": summary_rows}, best_recommendation


# ==============================================================================
# 5. MANUAL INSPECTION VIEW FORMATTER
# ==============================================================================

def format_manual_inspection_view(eval_result: Dict[str, Any]) -> str:
    """
    Formats query-by-query manual relevance inspection view displaying
    query, expected source, rank, score, source, chunk index, text preview,
    and manual relevance judgements.
    """
    lines = []
    lines.append("========================================================")
    lines.append(f"MANUAL RELEVANCE INSPECTION: {eval_result.get('label', eval_result.get('configuration'))}")
    lines.append("========================================================\n")

    for q_eval in eval_result.get("query_evaluations", []):
        lines.append(f"Query: {q_eval['query']}")
        lines.append(f"Topic: {q_eval['topic']}")
        lines.append(f"Expected Source: {q_eval['expected_source']}")
        lines.append(f"Status: {'[HIT]' if q_eval['hit'] else '[MISS]'}")
        lines.append("")

        if not q_eval.get("inspection_chunks"):
            lines.append("  No chunks retrieved matching criteria.")
            lines.append("")
        else:
            for chunk in q_eval["inspection_chunks"]:
                lines.append(f"Rank {chunk['rank']}")
                lines.append(f"Score: {chunk['score']:.4f}")
                lines.append(f"Source: {chunk['source']}")
                lines.append(f"Section: {chunk['section']}")
                lines.append(f"Chunk Index: {chunk['chunk_index']}")
                lines.append(f"Text: {chunk['text'][:120]}...")
                lines.append(f"Judgement: [{chunk['judgement_hint']}]")
                lines.append("")

        lines.append("-" * 50 + "\n")

    return "\n".join(lines).strip()


# ==============================================================================
# 6. GROUND-TRUTH LABELLED QUERY DATASET (PHASE 3.36)
# ==============================================================================

LABELLED_QUERIES: List[Dict[str, Any]] = [
    {
        "query": "What mandatory statutory registrations and compliance licenses are required for importing IT hardware and telecommunications equipment?",
        "relevant_chunk_ids": [
            "customs_requirements_paragraph_003",
            "customs_requirements.txt:3"
        ],
        "expected_sources": ["customs_requirements.txt"],
        "keywords": ["statutory", "registrations", "BIS", "hardware", "telecommunications"],
        "topic": "Statutory Registrations & BIS CRS"
    },
    {
        "query": "Which Incoterms 2020 rules define cost, risk, insurance, FOB, and CIF obligations for international trade?",
        "relevant_chunk_ids": [
            "international_shipping_guide_paragraph_001",
            "international_shipping_guide.pdf:1"
        ],
        "expected_sources": ["international_shipping_guide.pdf"],
        "keywords": ["Incoterms", "FOB", "CIF", "DDP", "risk"],
        "topic": "Incoterms 2020 Rules"
    },
    {
        "query": "What shipping documentation, commercial invoice, and packing list requirements apply to international customs clearance?",
        "relevant_chunk_ids": [
            "shipping_rules_paragraph_001",
            "shipping_rules.txt:1"
        ],
        "expected_sources": ["shipping_rules.txt"],
        "keywords": ["commercial", "invoice", "packing", "list", "customs"],
        "topic": "Mandatory Shipping Documentation"
    },
    {
        "query": "How are Basic Customs Duty, GST, and 8-digit HS Code tariff classifications determined for imported commodities?",
        "relevant_chunk_ids": [
            "customs_requirements_paragraph_001",
            "customs_requirements.txt:1"
        ],
        "expected_sources": ["customs_requirements.txt"],
        "keywords": ["Basic Customs Duty", "HS", "code", "tariff"],
        "topic": "HS Code & Basic Customs Duty"
    },
    {
        "query": "How can importers claim preferential tariff rates under Free Trade Agreements using a Certificate of Origin?",
        "relevant_chunk_ids": [
            "customs_requirements_paragraph_002",
            "customs_requirements.txt:2"
        ],
        "expected_sources": ["customs_requirements.txt"],
        "keywords": ["preferential", "tariff", "Certificate of Origin", "FTA"],
        "topic": "Preferential Tariffs & FTAs"
    }
]


# ==============================================================================
# 7. QUERY & DATASET RECALL/PRECISION EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_query(
    item: Dict[str, Any],
    k: int = 5,
    strategy: str = "vector",
    metadata_filter: Optional[Dict[str, Any]] = None,
    collection: Optional[VectorCollection] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates retrieval for a single labelled query item against ground-truth relevant chunk IDs.
    Calculates Recall@K, Precision@K, Reciprocal Rank (RR), hits, missing chunks, and preserves retrieved chunk details.

    Args:
        item: Dict containing 'query', 'relevant_chunk_ids', 'expected_sources', 'keywords'.
        k: Positive integer specifying Top-K context size.
        strategy: Retrieval strategy string ('vector', 'hybrid', 'reranked').
        metadata_filter: Optional metadata filtering criteria.
        collection: Optional VectorCollection instance.
        client: Optional embedding client.
        model: Optional embedding model name.

    Returns:
        Structured result dict containing recall, precision, RR, hits, retrieved_ids, missing_ids.
    """
    if not item or not isinstance(item, dict) or "query" not in item:
        raise ValueError("Item must be a valid labelled query dictionary containing 'query'.")

    query = item["query"]
    rel_ids = item.get("relevant_chunk_ids") or item.get("relevant_chunk_ids_set") or []
    rel_id_set = set(rel_ids) if isinstance(rel_ids, (list, tuple, set)) else set()
    exp_sources = set(item.get("expected_sources", []))

    if collection is None:
        collection = load_indexed_vector_collection()

    total_chunks = collection.count()
    effective_k = min(k, total_chunks) if total_chunks > 0 else k

    # Execute specified retrieval strategy
    if strategy == "reranked":
        from src.reranker import retrieve_and_rerank
        candidate_k = max(effective_k + 1, min(total_chunks, effective_k * 3))
        if candidate_k <= effective_k:
            raw_results = retrieve(query, k=effective_k, metadata_filter=metadata_filter, collection=collection, client=client, model=model)
        else:
            _, raw_results, _ = retrieve_and_rerank(
                query=query,
                candidate_k=candidate_k,
                final_k=effective_k,
                metadata_filter=metadata_filter,
                collection=collection,
                client=client,
                model=model
            )
    elif strategy == "hybrid":
        keywords = item.get("keywords", query.split())
        raw_results = hybrid_retrieve(
            query=query,
            keywords=keywords,
            k=effective_k,
            metadata_filter=metadata_filter,
            collection=collection,
            client=client,
            model=model
        )
    else:
        raw_results = retrieve(
            query=query,
            k=effective_k,
            metadata_filter=metadata_filter,
            collection=collection,
            client=client,
            model=model
        )

    retrieved_chunk_records = []
    retrieved_ids = []
    hits = []
    first_hit_rank = None

    for rank_idx, match in enumerate(raw_results, start=1):
        doc_id = match.get("document_id") or match.get("id") or f"chunk_{rank_idx}"
        source = match.get("source", "unknown")
        chunk_idx = match.get("chunk_index", 1)
        source_key = f"{source}:{chunk_idx}"
        text = match.get("chunk_text") or match.get("text", "")
        score = match.get("rerank_score", match.get("hybrid_score", match.get("similarity_score", 0.0)))

        retrieved_ids.append(doc_id)

        # Check if match is hit against ground-truth rel_id_set or expected_sources
        is_hit = False
        if doc_id in rel_id_set or source_key in rel_id_set or source in exp_sources:
            is_hit = True
            hits.append(doc_id)
            if first_hit_rank is None:
                first_hit_rank = rank_idx

        retrieved_chunk_records.append({
            "rank": rank_idx,
            "document_id": doc_id,
            "source_key": source_key,
            "source": source,
            "chunk_index": chunk_idx,
            "score": score,
            "text": text,
            "is_hit": is_hit
        })

    unique_hits = list(dict.fromkeys(hits))
    total_relevant = len(exp_sources) if exp_sources else max(1, len(rel_id_set) // 2)
    if total_relevant == 0:
        total_relevant = 1

    recall = round(len(unique_hits) / total_relevant, 4)
    recall = min(1.0, recall)

    retrieved_count = len(retrieved_ids)
    precision = round(len(unique_hits) / retrieved_count, 4) if retrieved_count > 0 else 0.0
    reciprocal_rank = round(1.0 / first_hit_rank, 4) if first_hit_rank is not None else 0.0

    missing_ids = [rid for rid in rel_ids if rid not in hits and not any(rid.startswith(s) for s in [r["source"] for r in retrieved_chunk_records if r["is_hit"]])]

    return {
        "query": query,
        "topic": item.get("topic", "General"),
        "k": k,
        "strategy": strategy,
        "retrieved_ids": retrieved_ids,
        "relevant_chunk_ids": list(rel_id_set),
        "hits": unique_hits,
        "missing_chunk_ids": missing_ids,
        "hit": len(unique_hits) > 0,
        "first_hit_rank": first_hit_rank,
        "recall": recall,
        "precision": precision,
        "reciprocal_rank": reciprocal_rank,
        "retrieved_chunks": retrieved_chunk_records
    }


def evaluate_dataset(
    labelled_queries: Optional[List[Dict[str, Any]]] = None,
    k: int = 5,
    strategy: str = "vector",
    metadata_filter: Optional[Dict[str, Any]] = None,
    collection: Optional[VectorCollection] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates retrieval performance across all queries in a labelled query dataset.
    """
    dataset = labelled_queries if labelled_queries is not None else LABELLED_QUERIES
    if not dataset or not isinstance(dataset, list):
        raise ValueError("labelled_queries must be a non-empty list of query dicts.")

    if collection is None:
        collection = load_indexed_vector_collection()

    query_evals = []
    total_queries = len(dataset)
    recalls = []
    precisions = []
    rrs = []
    successful = 0
    failed = 0

    for item in dataset:
        res = evaluate_query(
            item=item,
            k=k,
            strategy=strategy,
            metadata_filter=metadata_filter,
            collection=collection,
            client=client,
            model=model
        )
        query_evals.append(res)
        recalls.append(res["recall"])
        precisions.append(res["precision"])
        rrs.append(res["reciprocal_rank"])

        if res["recall"] >= 1.0 or res["hit"]:
            successful += 1
        else:
            failed += 1

    avg_recall = round(sum(recalls) / total_queries, 4) if total_queries > 0 else 0.0
    avg_precision = round(sum(precisions) / total_queries, 4) if total_queries > 0 else 0.0
    mrr = round(sum(rrs) / total_queries, 4) if total_queries > 0 else 0.0

    return {
        "strategy": strategy,
        "k": k,
        "total_queries": total_queries,
        "successful_queries": successful,
        "failed_queries": failed,
        "average_recall": avg_recall,
        "average_precision": avg_precision,
        "mrr": mrr,
        "recall_pct": round(avg_recall * 100, 2),
        "precision_pct": round(avg_precision * 100, 2),
        "query_evaluations": query_evals
    }


def evaluate_multi_k(
    labelled_queries: Optional[List[Dict[str, Any]]] = None,
    k_values: Optional[List[int]] = None,
    strategy: str = "vector",
    collection: Optional[VectorCollection] = None
) -> Dict[str, Any]:
    """
    Evaluates Recall@K, Precision@K, and MRR@K across multiple K values (e.g. K=1, 3, 5, 10).
    """
    ks = k_values if k_values is not None else [1, 3, 5, 10]
    dataset = labelled_queries if labelled_queries is not None else LABELLED_QUERIES
    if collection is None:
        collection = load_indexed_vector_collection()

    k_results = {}
    for k in ks:
        eval_res = evaluate_dataset(dataset, k=k, strategy=strategy, collection=collection)
        k_results[k] = {
            "k": k,
            "recall": eval_res["average_recall"],
            "recall_pct": eval_res["recall_pct"],
            "precision": eval_res["average_precision"],
            "precision_pct": eval_res["precision_pct"],
            "mrr": eval_res["mrr"],
            "successful_queries": eval_res["successful_queries"],
            "failed_queries": eval_res["failed_queries"]
        }

    return {
        "strategy": strategy,
        "k_values": ks,
        "metrics_by_k": k_results
    }


# ==============================================================================
# 8. FAILURE INSPECTION & ROOT CAUSE DIAGNOSIS
# ==============================================================================

def inspect_retrieval_failures(dataset_eval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts detailed failure records for any query where Recall@K < 1.0 or hit is False.
    """
    failures = []
    for q_eval in dataset_eval_result.get("query_evaluations", []):
        if q_eval.get("recall", 0.0) < 1.0 or not q_eval.get("hit", False):
            failures.append({
                "query": q_eval["query"],
                "topic": q_eval.get("topic", "General"),
                "recall": q_eval["recall"],
                "precision": q_eval["precision"],
                "expected_relevant_chunks": q_eval["relevant_chunk_ids"],
                "retrieved_chunk_ids": q_eval["retrieved_ids"],
                "missing_relevant_chunks": q_eval.get("missing_chunk_ids", []),
                "retrieved_scores": [c["score"] for c in q_eval.get("retrieved_chunks", [])],
                "retrieved_sources": [c["source"] for c in q_eval.get("retrieved_chunks", [])],
                "retrieved_texts": [c["text"][:100] for c in q_eval.get("retrieved_chunks", [])]
            })
    return failures


def diagnose_failure_causes(failure_record: Dict[str, Any]) -> List[str]:
    """
    Analyzes a retrieval failure record and identifies empirical root causes.
    """
    causes = []
    scores = failure_record.get("retrieved_scores", [])

    if not failure_record.get("retrieved_chunk_ids"):
        causes.append("Empty retrieval result (no chunks matched criteria or metadata filter was overly restrictive).")
    elif scores and max(scores) < 0.2:
        causes.append("Weak semantic similarity (query wording mismatch or embedding distance was too high).")
    elif len(failure_record.get("missing_relevant_chunks", [])) > 0:
        causes.append("Relevant information split across chunk boundaries or K was too small to capture all relevant chunks.")

    if not causes:
        causes.append("Keyword mismatch or missing domain-specific terminology in document chunks.")

    return causes


def compare_retrieval_strategies(
    labelled_queries: Optional[List[Dict[str, Any]]] = None,
    k_values: Optional[List[int]] = None,
    collection: Optional[VectorCollection] = None
) -> Dict[str, Any]:
    """
    Executes multi-K evaluation across Vector Search, Hybrid Search, and Re-Ranked Retrieval.
    """
    dataset = labelled_queries if labelled_queries is not None else LABELLED_QUERIES
    ks = k_values if k_values is not None else [1, 3, 5, 10]
    if collection is None:
        collection = load_indexed_vector_collection()

    strategies = ["vector", "hybrid", "reranked"]
    comparison = {}

    for strat in strategies:
        comparison[strat] = evaluate_multi_k(dataset, k_values=ks, strategy=strat, collection=collection)

    return comparison
