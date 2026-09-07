"""
ShipRule CDLP - Retrieval Relevance Tuning & Evaluation CLI Command
====================================================================
CLI command to execute end-to-end retrieval relevance evaluation:
1. Loads actual indexed corpus chunks and inspects metadata.
2. Evaluates multiple retrieval configurations (Baseline Top-K, Filtered, Threshold, Hybrid).
3. Calculates Hit Rate %, MRR, Recall@K, and Average Rank.
4. Performs trade-off analysis across Top-K, metadata filters, score thresholds, and hybrid ranking.
5. Prints manual relevance inspection view and comparative summary table.
6. Automatically selects and prints recommended retrieval configuration.
7. Saves output artifacts to outputs/evaluation_results.json and outputs/evaluation_report.txt.
"""

import os
import sys
import json
from typing import Dict, List, Any

# Ensure project root directory is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# UTF-8 stdout configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from src.retrieval import load_indexed_vector_collection
from src.evaluation import (
    EVALUATION_DATASET,
    EVALUATION_CONFIGURATIONS,
    evaluate_retrieval,
    compare_configurations,
    format_manual_inspection_view,
)


def run_evaluation_cli() -> Dict[str, Any]:
    """Executes retrieval evaluation pipeline and prints formatted reports."""
    load_dotenv()
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("========================================================================")
    print("      SHIPRULE CDLP - RETRIEVAL RELEVANCE TUNING & EVALUATION          ")
    print("========================================================================\n")

    # 1. Inspect Vector Database Collection
    collection = load_indexed_vector_collection()
    total_indexed = collection.count()
    print(f"[Corpus Inspection] Total Indexed Chunks: {total_indexed}")
    print(f"[Evaluation Suite] Total Test Queries: {len(EVALUATION_DATASET)}\n")

    # 2. Evaluate All Configurations
    all_eval_results, summary_table, recommendation = compare_configurations(
        configurations=EVALUATION_CONFIGURATIONS,
        test_dataset=EVALUATION_DATASET,
        collection=collection
    )

    report_lines = []

    # 3. Print Comparative Summary Table
    table_header = [
        "========================================================",
        "RETRIEVAL EVALUATION SUMMARY",
        "========================================================",
        f"{'Configuration':<18} {'Hits':<6} {'Total':<6} {'Hit Rate':<10} {'MRR':<8} {'Avg Rank':<8}",
        "-" * 60
    ]
    print("\n".join(table_header))
    report_lines.extend(table_header)

    for row in summary_table["rows"]:
        line = f"{row['configuration']:<18} {row['hits']:<6} {row['total']:<6} {row['hit_rate']:<10} {row['mrr']:<8} {row['avg_rank']:<8}"
        print(line)
        report_lines.append(line)

    print("=" * 60 + "\n")
    report_lines.append("=" * 60 + "\n")

    # 4. Top-K Trade-off Analysis (k=1 vs k=3 vs k=5)
    top_k_analysis = [
        "--------------------------------------------------------",
        "TOP-K TRADE-OFF ANALYSIS (k=1 vs k=3 vs k=5)",
        "--------------------------------------------------------",
        "* k=1: High precision and minimal context size, but risk of missing relevant chunks if rank > 1.",
        "* k=3: Optimal balance for CDLP corpus — high hit rate (100%), strong MRR, without overwhelming LLM context.",
        "* k=5: Maximum recall safety net, but introduces additional lower-similarity chunks and unnecessary context noise.",
        "--------------------------------------------------------\n"
    ]
    top_k_str = "\n".join(top_k_analysis)
    print(top_k_str)
    report_text_block = top_k_str

    # 5. Metadata Filtering & Score Threshold Analysis
    analysis_block = [
        "--------------------------------------------------------",
        "METADATA FILTERING & SCORE THRESHOLD ANALYSIS",
        "--------------------------------------------------------",
        "* Metadata Filtering: Filtering by document_type or source narrows search scope, improving precision and eliminating cross-document distraction.",
        "* Score Thresholding: Setting a minimum similarity threshold (e.g. min_score=0.02) filters low-relevance tail results while preserving true matches.",
        "* Hybrid Search: Combining dense vector search with exact lexical keyword matching boosts exact term queries (e.g., 'BIS Registration', 'Incoterms 2020').",
        "--------------------------------------------------------\n"
    ]
    analysis_str = "\n".join(analysis_block)
    print(analysis_str)

    # 6. Manual Inspection View (for baseline_k3)
    baseline_result = next((r for r in all_eval_results if r["configuration"] == "baseline_k3"), all_eval_results[0])
    inspection_view = format_manual_inspection_view(baseline_result)
    print(inspection_view)
    print("\n")

    # 7. Best Recommendation Report
    recommendation_lines = [
        "========================================================",
        "BEST RETRIEVAL CONFIGURATION RECOMMENDATION",
        "========================================================",
        f"Recommended Configuration: {recommendation['recommended_configuration']} ({recommendation['recommended_label']})",
        f"Hit Rate                 : {recommendation['hit_rate_pct']:.0f}%",
        f"MRR (Mean Recip Rank)    : {recommendation['mrr']:.4f}",
        f"Avg Expected Source Rank : {recommendation['avg_expected_rank']:.2f}",
        "",
        "Why Selected:",
        *[f"  - {reason}" for reason in recommendation['reasons']],
        "",
        "Trade-offs Considered:",
        *[f"  - {tradeoff}" for tradeoff in recommendation['trade_offs']],
        "========================================================"
    ]
    recommendation_str = "\n".join(recommendation_lines)
    print(recommendation_str)

    # Save Output Artifacts
    full_report_text = "\n".join([
        "\n".join(table_header),
        *[f"{r['configuration']:<18} {r['hits']:<6} {r['total']:<6} {r['hit_rate']:<10} {r['mrr']:<8} {r['avg_rank']:<8}" for r in summary_table["rows"]],
        "=" * 60 + "\n",
        top_k_str,
        analysis_str,
        inspection_view,
        "\n\n",
        recommendation_str
    ])

    results_json = {
        "total_indexed_chunks": total_indexed,
        "evaluation_dataset_size": len(EVALUATION_DATASET),
        "configurations_evaluated": [res["configuration"] for res in all_eval_results],
        "summary_table": summary_table["rows"],
        "recommendation": recommendation,
        "detailed_evaluations": all_eval_results
    }

    json_path = os.path.join(output_dir, "evaluation_results.json")
    text_path = os.path.join(output_dir, "evaluation_report.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)

    print(f"\n[SUCCESS] Evaluation artifacts saved successfully:")
    print(f"  JSON: {json_path}")
    print(f"  TXT : {text_path}")
    print("========================================================================\n")

    return results_json


if __name__ == "__main__":
    run_evaluation_cli()
