"""
ShipRule CDLP - RAG Evaluation & Answer Quality Scoring Runner
===============================================================
Executes the end-to-end RAG system evaluation across all test cases.
Outputs structured JSON and human-readable evaluation summary report.
"""

import os
import sys
import json

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# UTF-8 stdout configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from evaluation.rag_evaluation import (
    TEST_SET,
    evaluate_rag_quality,
    format_evaluation_summary
)


def main():
    print("[INFO] Running RAG System Evaluation & Answer Quality Scoring...")
    print(f"[INFO] Evaluating {len(TEST_SET)} test cases...")

    summary_data = evaluate_rag_quality(TEST_SET)
    summary_report = format_evaluation_summary(summary_data)

    print("\n" + summary_report)

    # Ensure outputs directory exists
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Save JSON results
    json_path = os.path.join(outputs_dir, "rag_evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n[OUTPUT] Evaluation results saved to JSON: {json_path}")

    # Save Summary TXT
    txt_path = os.path.join(outputs_dir, "rag_evaluation_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_report)
    print(f"[OUTPUT] Evaluation summary report saved to TXT: {txt_path}")


if __name__ == "__main__":
    main()
