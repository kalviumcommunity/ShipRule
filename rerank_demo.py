"""
ShipRule CDLP - Chunk Re-Ranking for Precision CLI Demonstration
================================================================
Demonstrates two-stage retrieval pipeline:
1. Stage 1: Vector Retrieval (candidate_k=10)
2. Stage 2: Relevance Re-Ranking (rerank())
3. Stage 3: Top-3 Final Context Selection

Displays BEFORE / AFTER rankings, explicit Original Rank -> New Rank movements,
candidate size benchmarking (k=5, 10, 20), and metrics comparison (Standard vs Re-ranked).
Saves outputs to outputs/rerank_demo_output.json and outputs/rerank_output.txt.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# UTF-8 stdout configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from src.retrieval import load_indexed_vector_collection
from src.reranker import (
    retrieve_and_rerank,
    format_before_after_comparison,
    format_rank_movement_report,
)
from src.evaluation import evaluate_retrieval, EVALUATION_DATASET


def run_reranking_demonstration() -> Dict[str, Any]:
    """
    Runs the end-to-end two-stage retrieval and re-ranking demonstration,
    candidate size benchmarking, evaluation comparison, and saves output artifacts.
    """
    load_dotenv()
    collection = load_indexed_vector_collection()

    demo_artifacts: Dict[str, Any] = {
        "two_stage_demo": {},
        "candidate_benchmarking": [],
        "evaluation_comparison": {}
    }

    report_text_blocks = []

    print("========================================================================")
    print("      SHIPRULE CDLP - CHUNK RE-RANKING FOR PRECISION DEMO              ")
    print("========================================================================\n")

    # --------------------------------------------------------------------------
    # DEMO 1: TWO-STAGE RETRIEVAL & RE-RANKING ON COMPLEX QUERY
    # --------------------------------------------------------------------------
    sample_query = "What mandatory statutory registrations and import documents are required for IT hardware in India?"
    candidate_k = 10
    final_k = 3

    print(f"Executing Two-Stage Retrieval...")
    print(f"  Query      : '{sample_query}'")
    print(f"  Candidate K: {candidate_k}")
    print(f"  Final K    : {final_k}\n")

    candidates, reranked_top_k, timing_metrics = retrieve_and_rerank(
        query=sample_query,
        candidate_k=candidate_k,
        final_k=final_k,
        collection=collection
    )

    # 1. BEFORE vs AFTER Comparison Report
    fmt_before_after = format_before_after_comparison(
        query=sample_query,
        before_candidates=candidates,
        after_candidates=reranked_top_k
    )
    print(fmt_before_after)
    report_text_blocks.append(fmt_before_after)
    report_text_blocks.append("\n" + "=" * 50 + "\n")

    # 2. Rank Movement Report
    fmt_rank_movement = format_rank_movement_report(after_candidates=reranked_top_k)
    print("\n" + fmt_rank_movement)
    report_text_blocks.append(fmt_rank_movement)
    report_text_blocks.append("\n" + "=" * 50 + "\n")

    print("\nPipeline Timing Metrics:")
    print(f"  Vector Retrieval (Stage 1) : {timing_metrics['vector_retrieval_time_ms']} ms")
    print(f"  Re-Ranking (Stage 2)       : {timing_metrics['reranking_time_ms']} ms")
    print(f"  Total Pipeline Latency     : {timing_metrics['total_pipeline_time_ms']} ms\n")

    demo_artifacts["two_stage_demo"] = {
        "query": sample_query,
        "candidate_k": candidate_k,
        "final_k": final_k,
        "timing_metrics": timing_metrics,
        "before_candidates": candidates,
        "after_candidates": reranked_top_k
    }

    # --------------------------------------------------------------------------
    # DEMO 2: CANDIDATE SIZE BENCHMARKING (k=5, k=10, k=20)
    # --------------------------------------------------------------------------
    print("========================================================================")
    print("          CANDIDATE SIZE BENCHMARKING (k=5, k=10, k=20)                 ")
    print("========================================================================")

    candidate_sizes = [5, 10, 20]
    benchmarking_results = []

    print(f"{'Candidate K':<14} {'Final K':<10} {'Hit Rate':<12} {'MRR':<10} {'Retrieval Latency':<20} {'Re-Rank Latency':<18}")
    print("-" * 84)

    for ck in candidate_sizes:
        # Note: If corpus is smaller than ck, retrieve will cap to total indexed chunks cleanly
        t_ret_sum = 0.0
        t_rr_sum = 0.0

        eval_setting = {
            "name": f"rerank_ck{ck}_fk3",
            "label": f"Two-Stage Re-Ranking (ck={ck}, fk=3)",
            "k": 3,
            "candidate_k": ck,
            "use_reranking": True,
            "filter": None,
            "min_score": 0.0
        }

        t_start = time.perf_counter()
        eval_res = evaluate_retrieval(eval_setting, collection=collection)
        t_end = time.perf_counter()

        # Compute average latencies across dataset queries
        for q_item in EVALUATION_DATASET:
            _, _, metrics = retrieve_and_rerank(
                query=q_item["query"],
                candidate_k=ck,
                final_k=3,
                collection=collection
            )
            t_ret_sum += metrics["vector_retrieval_time_ms"]
            t_rr_sum += metrics["reranking_time_ms"]

        num_queries = len(EVALUATION_DATASET)
        avg_ret_ms = round(t_ret_sum / num_queries, 2)
        avg_rr_ms = round(t_rr_sum / num_queries, 2)

        bench_record = {
            "candidate_k": ck,
            "final_k": 3,
            "hit_rate_pct": eval_res["hit_rate_pct"],
            "mrr": eval_res["mrr"],
            "recall_at_k": eval_res["recall_at_k"],
            "avg_vector_retrieval_ms": avg_ret_ms,
            "avg_reranking_ms": avg_rr_ms,
            "avg_total_ms": round(avg_ret_ms + avg_rr_ms, 2)
        }
        benchmarking_results.append(bench_record)

        print(f"{ck:<14} {3:<10} {eval_res['hit_rate_pct']:<12.1f}% {eval_res['mrr']:<10.4f} {avg_ret_ms:<20.2f} ms {avg_rr_ms:<18.2f} ms")

    print("=" * 84 + "\n")
    demo_artifacts["candidate_benchmarking"] = benchmarking_results

    # --------------------------------------------------------------------------
    # DEMO 3: EVALUATION COMPARISON (Standard Top-K vs Re-ranked Top-K)
    # --------------------------------------------------------------------------
    print("========================================================================")
    print("     EVALUATION COMPARISON: STANDARD TOP-K vs RE-RANKED TOP-K          ")
    print("========================================================================")

    std_baseline_cfg = {
        "name": "baseline_k3",
        "label": "Standard Top-K=3 (Vector Only)",
        "k": 3,
        "filter": None,
        "min_score": 0.0,
        "use_hybrid": False,
        "use_reranking": False
    }

    reranked_cfg = {
        "name": "reranked_ck10_fk3",
        "label": "Two-Stage Re-Ranked Top-K=3 (ck=10, fk=3)",
        "k": 3,
        "candidate_k": 10,
        "filter": None,
        "min_score": 0.0,
        "use_hybrid": False,
        "use_reranking": True
    }

    std_eval = evaluate_retrieval(std_baseline_cfg, collection=collection)
    rerank_eval = evaluate_retrieval(reranked_cfg, collection=collection)

    print(f"{'Pipeline Strategy':<38} {'Hits':<8} {'Hit Rate':<12} {'MRR':<10} {'Avg Rank':<10}")
    print("-" * 78)
    print(f"{std_baseline_cfg['label']:<38} {std_eval['hits']}/{std_eval['total_queries']:<5} {std_eval['hit_rate_pct']:<12.1f}% {std_eval['mrr']:<10.4f} {std_eval['avg_expected_rank'] or 'N/A':<10}")
    print(f"{reranked_cfg['label']:<38} {rerank_eval['hits']}/{rerank_eval['total_queries']:<5} {rerank_eval['hit_rate_pct']:<12.1f}% {rerank_eval['mrr']:<10.4f} {rerank_eval['avg_expected_rank'] or 'N/A':<10}")
    print("=" * 78 + "\n")

    demo_artifacts["evaluation_comparison"] = {
        "standard_top_k": std_eval,
        "reranked_top_k": rerank_eval
    }

    # --------------------------------------------------------------------------
    # SAVE OUTPUT ARTIFACTS
    # --------------------------------------------------------------------------
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "rerank_demo_output.json")
    text_path = os.path.join(output_dir, "rerank_output.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(demo_artifacts, f, indent=2, ensure_ascii=False)

    full_text_report = "\n".join(report_text_blocks)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full_text_report)

    print(f"[SUCCESS] Re-ranking demonstration artifacts saved:")
    print(f"  JSON Artifact: {json_path}")
    print(f"  TXT Artifact : {text_path}")
    print("========================================================================\n")

    return demo_artifacts


if __name__ == "__main__":
    run_reranking_demonstration()
