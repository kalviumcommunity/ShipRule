"""
ShipRule CDLP - End-to-End RAG Pipeline CLI Demonstration Script
=================================================================
Runs end-to-end RAG pipeline demonstrations:
1. Standard documentation query ("What shipping documents are required for international customs clearance?")
2. Metadata filtered query ("What mandatory statutory registrations are required for IT hardware?")
3. Grounding & Hallucination test for unanswerable question ("What is the speed of light in vacuum?")

Prints clean ASCII formatted output displaying retrieved context, citations, generated answer,
sources, step latencies, and saves output artifacts to outputs/rag_pipeline_output.json and outputs/rag_output.txt.
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
from src.rag_pipeline import answer_query


def run_rag_pipeline_demonstration() -> Dict[str, Any]:
    """
    Executes end-to-end RAG pipeline demonstration across multiple queries,
    verifies grounding & citations, formats outputs, and writes artifacts to disk.
    """
    load_dotenv()
    collection = load_indexed_vector_collection()

    demo_artifacts: Dict[str, Any] = {
        "pipeline_demonstration": [],
        "grounding_test": {}
    }

    report_text_blocks = []

    print("========================================================================")
    print("                SHIPRULE CDLP - END-TO-END RAG PIPELINE                ")
    print("========================================================================\n")

    test_cases = [
        {
            "label": "Standard Customs Documentation Query",
            "query": "What shipping documents are required for international customs clearance?",
            "candidate_k": 10,
            "final_k": 3,
            "filter": None
        },
        {
            "label": "Statutory Registrations Query (Filtered)",
            "query": "What mandatory statutory registrations and compliance licenses are required for importing IT hardware in India?",
            "candidate_k": 10,
            "final_k": 3,
            "filter": {"source": "customs_requirements.txt"}
        },
        {
            "label": "Unanswerable / Out-of-Domain Query (Hallucination Test)",
            "query": "What is the speed of light in vacuum and how is it measured?",
            "candidate_k": 10,
            "final_k": 3,
            "filter": None
        }
    ]

    for idx, case in enumerate(test_cases, start=1):
        q = case["query"]
        lbl = case["label"]
        flt = case["filter"]

        print(f"--- Query {idx}: [{lbl}] ---")
        print(f"Query: '{q}'\n")

        res = answer_query(
            query=q,
            candidate_k=case["candidate_k"],
            final_k=case["final_k"],
            metadata_filter=flt,
            use_reranking=True,
            debug=True,
            collection=collection
        )

        lines = []
        lines.append("========================================")
        lines.append(f"RAG PIPELINE DEMONSTRATION #{idx}")
        lines.append("========================================")
        lines.append(f"Query: {q}")
        if flt:
            lines.append(f"Metadata Filter: {flt}")
        lines.append("")

        lines.append("----------------------------------------")
        lines.append("RETRIEVED CONTEXT")
        lines.append("----------------------------------------")

        if not res.get("retrieved_chunks"):
            lines.append("No relevant chunks retrieved.")
        else:
            for chunk_idx, chunk in enumerate(res["retrieved_chunks"], start=1):
                src = chunk.get("source", "unknown")
                c_idx = chunk.get("chunk_index", 1)
                r_score = chunk.get("rerank_score", chunk.get("similarity_score", 0.0))
                txt = chunk.get("chunk_text") or chunk.get("text", "")
                lines.append(f"[{chunk_idx}] Source: {src} | Chunk Index: {c_idx}")
                lines.append(f"    Re-Rank Score: {r_score:.2f} | Vector Score: {chunk.get('vector_score', 0.0):.4f}")
                lines.append(f"    Text: {txt[:140]}...")
                lines.append("")

        lines.append("----------------------------------------")
        lines.append("GENERATED ANSWER")
        lines.append("----------------------------------------")
        lines.append(res["answer"])
        lines.append("")

        lines.append("----------------------------------------")
        lines.append("SOURCES")
        lines.append("----------------------------------------")
        if not res.get("sources"):
            lines.append("None (Safe Fallback / Zero Hallucination Response)")
        else:
            for s_idx, src_name in enumerate(res["sources"], start=1):
                lines.append(f"[{s_idx}] {src_name}")
        lines.append("")

        lines.append("----------------------------------------")
        lines.append("PIPELINE TIMING METRICS")
        lines.append("----------------------------------------")
        tm = res.get("timing", {})
        lines.append(f"Embedding:        {tm.get('embedding_ms', 0.0):>7.2f} ms")
        lines.append(f"Retrieval:        {tm.get('retrieval_ms', 0.0):>7.2f} ms")
        lines.append(f"Re-ranking:       {tm.get('reranking_ms', 0.0):>7.2f} ms")
        lines.append(f"Context Assembly: {tm.get('context_assembly_ms', 0.0):>7.2f} ms")
        lines.append(f"Generation:       {tm.get('generation_ms', 0.0):>7.2f} ms")
        lines.append(f"Total Pipeline:   {tm.get('total_pipeline_time_ms', 0.0):>7.2f} ms")
        lines.append("========================================\n")

        block_text = "\n".join(lines)
        print(block_text)
        report_text_blocks.append(block_text)

        demo_record = {
            "label": lbl,
            "query": q,
            "filter": flt,
            "answer": res["answer"],
            "sources": res["sources"],
            "timing": res["timing"],
            "retrieved_chunks_count": len(res.get("retrieved_chunks", []))
        }

        if idx == 3:
            demo_artifacts["grounding_test"] = demo_record
        else:
            demo_artifacts["pipeline_demonstration"].append(demo_record)

    # --------------------------------------------------------------------------
    # SAVE OUTPUT ARTIFACTS
    # --------------------------------------------------------------------------
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "rag_pipeline_output.json")
    text_path = os.path.join(output_dir, "rag_output.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(demo_artifacts, f, indent=2, ensure_ascii=False)

    full_report_text = "\n".join(report_text_blocks)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)

    print(f"[SUCCESS] End-to-End RAG Pipeline demonstration artifacts saved:")
    print(f"  JSON Artifact: {json_path}")
    print(f"  TXT Artifact : {text_path}")
    print("========================================================================\n")

    return demo_artifacts


if __name__ == "__main__":
    run_rag_pipeline_demonstration()
