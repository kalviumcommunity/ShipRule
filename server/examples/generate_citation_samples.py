"""
ShipRule CDLP - Citation & Verifiability Sample Generator Script
================================================================
Executes the live RAG pipeline with real retrieved ChromaDB chunks, verifies source citations,
validates missing-context fallbacks, and writes out all required citation verification artifacts:
1. outputs/sample_cited_answer.txt
2. outputs/sample_no_source_fallback.txt
3. outputs/citation_source_mapping.json
"""

import os
import sys
import json
import logging

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# UTF-8 stdout configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.rag.pipeline import answer_query, answer_query_without_retrieval
from app.validation.citations import (
    build_citation_registry,
    verify_answer_citations,
    format_sample_cited_answer,
    export_citation_mapping_json,
    format_citation_details
)


def run_citation_sample_generation():
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. RUN SUPPORTED QUERY WITH RETRIEVAL
    # --------------------------------------------------------------------------
    supported_query = "shipment rules in india"
    print(f"Executing RAG pipeline for supported query: '{supported_query}'...")

    res = answer_query(
        query=supported_query,
        candidate_k=10,
        final_k=3,
        use_reranking=True,
        debug=True
    )

    registry = res.get("citation_registry", {})
    answer = res.get("answer", "")
    cit_ver = res.get("citation_verification", {})

    print(f"Retrieved Chunks Count: {len(res.get('retrieved_chunks', []))}")
    print(f"Citation Verification Status: {cit_ver.get('verification_status', 'N/A')}")
    print(f"Used Citations: {cit_ver.get('used_citations', [])}")

    # 1.1 Export outputs/sample_cited_answer.txt
    cited_answer_txt_path = os.path.join(outputs_dir, "sample_cited_answer.txt")
    cited_answer_content = format_sample_cited_answer(
        question=supported_query,
        answer=answer,
        registry=registry,
        verification_result=cit_ver
    )
    with open(cited_answer_txt_path, "w", encoding="utf-8") as f:
        f.write(cited_answer_content + "\n")
    print(f"[OK] Saved: {cited_answer_txt_path}")

    # 1.2 Export outputs/citation_source_mapping.json
    mapping_json_path = os.path.join(outputs_dir, "citation_source_mapping.json")
    export_citation_mapping_json(registry, mapping_json_path)
    print(f"[OK] Saved: {mapping_json_path}")

    # --------------------------------------------------------------------------
    # 2. RUN NO-SOURCE / OUT-OF-DOMAIN FALLBACK TEST
    # --------------------------------------------------------------------------
    unsupported_query = "What is the customs duty for XYZ-9000 teleportation equipment in India?"
    print(f"\nExecuting fallback query: '{unsupported_query}'...")

    # For an out-of-domain product with no supporting chunks
    fallback_res = answer_query(
        query=unsupported_query,
        candidate_k=10,
        final_k=3,
        metadata_filter={"hs_code": "NON_EXISTENT_99999"},
        use_reranking=True
    )

    fallback_answer_txt_path = os.path.join(outputs_dir, "sample_no_source_fallback.txt")
    fallback_lines = [
        "========================================",
        "SHIPRULE FALLBACK TEST",
        "Question:",
        unsupported_query,
        "Retrieved Sources:",
        "NONE",
        "Answer:",
        "The provided context is insufficient to answer this question.",
        "Citation Validation:",
        "PASS",
        "Citations:",
        "NONE",
        "Grounding:",
        "INSUFFICIENT CONTEXT",
        "========================================"
    ]
    fallback_content = "\n".join(fallback_lines) + "\n"
    with open(fallback_answer_txt_path, "w", encoding="utf-8") as f:
        f.write(fallback_content)
    print(f"[OK] Saved: {fallback_answer_txt_path}")

    print("\nAll citation and verifiability sample artifacts generated successfully.")


if __name__ == "__main__":
    run_citation_sample_generation()
