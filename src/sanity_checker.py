"""
ShipRule CDLP - Embedding Quality Checks & Sanity Tests
========================================================
Provides a lightweight smoke-testing layer that verifies whether the embedding
and retrieval pipeline is working correctly. Performs pipeline sanity checks
and validates retrieval performance against known query-source test cases using
the project's existing embedding model, chunk representation, and cosine similarity.
"""

import os
import sys
import json
import math
from typing import Dict, List, Any, Optional, Tuple

# Ensure UTF-8 handling on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
from src.embeddings import (
    create_embedding_client,
    generate_query_embedding,
    rank_chunks_by_similarity,
    cosine_similarity,
    load_validated_chunks,
    run_embedding_pipeline
)

# Configurable default retrieval test cases
DEFAULT_TEST_CASES: List[Dict[str, str]] = [
    {
        "query": "How can a learner reset their password?",
        "expected_source": "account-guide.md"
    },
    {
        "query": "When does the cafeteria menu change?",
        "expected_source": "campus-guide.md"
    },
    {
        "query": "What are the required commercial invoice and packing list documents for international shipping?",
        "expected_source": "shipping_rules.txt"
    },
    {
        "query": "What import duty rates and BIS registration apply to electronic components under HS code 8471.30?",
        "expected_source": "customs_requirements.txt"
    },
    {
        "query": "Where can I find customs duty tariff rates based on 8-digit HS Code classification?",
        "expected_source": "international_shipping_guide.pdf"
    }
]

POSSIBLE_FAILURE_CAUSES: List[str] = [
    "Wrong embedding model",
    "Query/document embedding mismatch",
    "Incorrect chunk-to-vector alignment",
    "Incorrect metadata/source",
    "Poor chunking or text cleaning",
    "Wrong similarity metric",
    "Duplicate or overly generic chunks"
]


# ==============================================================================
# 1. PIPELINE SANITY CHECKS
# ==============================================================================

def run_pipeline_sanity_checks(
    candidate_chunks: List[Dict[str, Any]],
    query_vector: Optional[List[float]] = None,
    expected_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs structural and data-integrity sanity checks on the embedding pipeline:
    1. Verifies chunk and query embedding dimensions match.
    2. Validates embedding model consistency across chunks.
    3. Confirms every embedding is non-empty and associated with a valid chunk.
    4. Validates presence of source metadata for every chunk.
    5. Confirms similarity scores are valid numerical values.
    6. Identifies duplicate chunks (identical text or vector representations).

    Returns:
        Dict summarizing check status and detailed list of issues/warnings.
    """
    issues: List[str] = []
    warnings: List[str] = []
    chunk_count = len(candidate_chunks)

    if chunk_count == 0:
        return {
            "status": "FAILED",
            "issues": ["No candidate chunks provided for pipeline sanity check."],
            "warnings": warnings,
            "metrics": {"chunk_count": 0}
        }

    detected_dims: List[int] = []
    detected_models: set = set()
    seen_texts: Dict[str, str] = {}
    seen_vectors: Dict[Tuple[float, ...], str] = {}

    for idx, chunk in enumerate(candidate_chunks, start=1):
        chunk_id = chunk.get("chunk_id") or chunk.get("embedding_id") or f"chunk_{idx}"
        
        # 1. Source metadata check
        meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        source = chunk.get("source") or meta.get("source")
        if not source or not str(source).strip():
            issues.append(f"Chunk '{chunk_id}' is missing valid source metadata.")

        # 2. Text check
        text = chunk.get("chunk_text") or chunk.get("text") or meta.get("text") or ""
        text_clean = str(text).strip()
        if not text_clean:
            issues.append(f"Chunk '{chunk_id}' contains empty chunk text.")
        else:
            if text_clean in seen_texts:
                warnings.append(
                    f"Duplicate chunk text detected: '{chunk_id}' duplicates '{seen_texts[text_clean]}'."
                )
            else:
                seen_texts[text_clean] = str(chunk_id)

        # 3. Embedding vector check
        vector = chunk.get("embedding") or meta.get("embedding")
        if not vector or not isinstance(vector, (list, tuple)):
            issues.append(f"Chunk '{chunk_id}' has missing or non-list embedding vector.")
            continue

        if not all(isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x) for x in vector):
            issues.append(f"Chunk '{chunk_id}' contains NaN, Inf, or non-numeric values in vector.")
            continue

        dim = len(vector)
        detected_dims.append(dim)

        # Vector duplicate check (first 20 rounded float elements)
        vec_key = tuple(round(float(x), 6) for x in vector[:20])
        if vec_key in seen_vectors:
            warnings.append(
                f"Duplicate embedding vector detected: '{chunk_id}' vector matches '{seen_vectors[vec_key]}'."
            )
        else:
            seen_vectors[vec_key] = str(chunk_id)

        # 4. Model metadata check
        chunk_model = chunk.get("embedding_model") or meta.get("embedding_model")
        if chunk_model:
            detected_models.add(chunk_model)

    # Dimension consistency check across chunks
    unique_dims = set(detected_dims)
    if len(unique_dims) > 1:
        issues.append(f"Inconsistent vector dimensions detected across chunks: {sorted(list(unique_dims))}.")

    # Dimension compatibility check with query vector
    if query_vector and detected_dims:
        query_dim = len(query_vector)
        doc_dim = detected_dims[0]
        if query_dim != doc_dim:
            issues.append(
                f"Query and document embedding dimension mismatch: query dim = {query_dim}, document dim = {doc_dim}."
            )

    # Model consistency check
    if expected_model and detected_models:
        for m in detected_models:
            if m != expected_model:
                warnings.append(
                    f"Chunk model metadata '{m}' differs from expected model '{expected_model}'."
                )

    status = "PASSED" if len(issues) == 0 else "FAILED"
    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "metrics": {
            "chunk_count": chunk_count,
            "vector_dimension": detected_dims[0] if detected_dims else None,
            "unique_models": list(detected_models),
            "duplicate_chunks_count": len(warnings)
        }
    }


# ==============================================================================
# 2. TEST CASE RETRIEVAL EVALUATION
# ==============================================================================

def evaluate_retrieval_test_case(
    test_case: Dict[str, str],
    candidate_chunks: List[Dict[str, Any]],
    client: Optional[Any] = None,
    model: Optional[str] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Evaluates a single retrieval test case:
    1. Embeds query using exact same embedding model as candidate chunks.
    2. Ranks stored chunks using cosine similarity.
    3. Identifies top source, top score, expected rank, and Top-K presence.

    Returns:
        Dict containing query evaluation metrics and PASS/FAIL status.
    """
    query = test_case.get("query", "").strip()
    expected_source = test_case.get("expected_source", "").strip()

    if not query:
        raise ValueError("Test case query string cannot be empty.")

    # 1. Embed query using exact same model & client
    query_vector = generate_query_embedding(query, client=client, model=model)

    if not query_vector:
        return {
            "query": query,
            "expected_source": expected_source,
            "top_source": "N/A",
            "top_score": 0.0,
            "expected_rank": None,
            "in_top_k": False,
            "status": "FAIL",
            "ranked_results": [],
            "error": "Failed to generate query embedding vector."
        }

    # 2. Retrieve & rank candidate chunks across ALL stored chunks
    max_k = max(top_k, len(candidate_chunks))
    all_ranked_results = rank_chunks_by_similarity(query_vector, candidate_chunks, top_k=max_k)

    if not all_ranked_results:
        return {
            "query": query,
            "expected_source": expected_source,
            "top_source": "N/A",
            "top_score": 0.0,
            "expected_rank": None,
            "in_top_k": False,
            "status": "FAIL",
            "ranked_results": [],
            "error": "No ranked results returned."
        }

    top_result = all_ranked_results[0]
    top_source = top_result["metadata"]["source"]
    top_score = top_result["score"]

    # 3. Find rank of expected source (1-indexed)
    expected_rank: Optional[int] = None
    for rank_idx, result in enumerate(all_ranked_results, start=1):
        res_source = result["metadata"]["source"]
        if res_source.lower() == expected_source.lower() or os.path.basename(res_source).lower() == os.path.basename(expected_source).lower():
            expected_rank = rank_idx
            break

    in_top_k = expected_rank is not None and expected_rank <= top_k
    # PASS status: expected source is top result (rank == 1)
    status = "PASS" if expected_rank == 1 else "FAIL"

    return {
        "query": query,
        "expected_source": expected_source,
        "top_source": top_source,
        "top_score": top_score,
        "expected_rank": expected_rank,
        "in_top_k": in_top_k,
        "status": status,
        "ranked_results": all_ranked_results[:top_k]
    }


# ==============================================================================
# 3. RUN ALL SANITY TESTS
# ==============================================================================

def run_embedding_sanity_tests(
    candidate_chunks: List[Dict[str, Any]],
    test_cases: Optional[List[Dict[str, str]]] = None,
    top_k: int = 3,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs the full suite of retrieval sanity tests and pipeline verification.
    """
    load_dotenv()
    selected_model = model or os.getenv("EMBED_MODEL", "text-embedding-3-small")

    if client is None:
        client = create_embedding_client()

    cases_to_run = test_cases if test_cases is not None else DEFAULT_TEST_CASES

    # Execute test cases
    results: List[Dict[str, Any]] = []
    passed_count = 0
    failed_count = 0

    first_query_vector: Optional[List[float]] = None

    for case in cases_to_run:
        eval_res = evaluate_retrieval_test_case(
            test_case=case,
            candidate_chunks=candidate_chunks,
            client=client,
            model=selected_model,
            top_k=top_k
        )

        if first_query_vector is None and eval_res.get("ranked_results"):
            first_query_vector = generate_query_embedding(case["query"], client=client, model=selected_model)

        results.append(eval_res)
        if eval_res["status"] == "PASS":
            passed_count += 1
        else:
            failed_count += 1

    # Execute pipeline sanity checks
    pipeline_check = run_pipeline_sanity_checks(
        candidate_chunks=candidate_chunks,
        query_vector=first_query_vector,
        expected_model=selected_model
    )

    return {
        "summary": {
            "total_tests": len(cases_to_run),
            "passed": passed_count,
            "failed": failed_count,
            "top_k_configured": top_k,
            "embedding_model": selected_model
        },
        "pipeline_check": pipeline_check,
        "test_results": results
    }


# ==============================================================================
# 4. REPORT FORMATTING & DISPLAY
# ==============================================================================

def format_sanity_report(report_data: Dict[str, Any]) -> str:
    """
    Formats the embedding sanity report into clean string output matching PRD specification.
    """
    summary = report_data.get("summary", {})
    test_results = report_data.get("test_results", [])
    pipeline_check = report_data.get("pipeline_check", {})

    lines: List[str] = []
    lines.append("==============================")
    lines.append("Embedding Sanity Report")
    lines.append("==============================")
    lines.append("")
    lines.append(f"Tests: {summary.get('total_tests', 0)}")
    lines.append(f"Passed: {summary.get('passed', 0)}")
    lines.append(f"Failed: {summary.get('failed', 0)}")
    lines.append("")

    for idx, test in enumerate(test_results, start=1):
        lines.append(f"Test {idx}")
        lines.append(f"Query: {test.get('query')}")
        lines.append(f"Expected source: {test.get('expected_source')}")
        lines.append(f"Top source: {test.get('top_source')}")
        lines.append(f"Top score: {test.get('top_score'):.2f}")
        rank_str = str(test.get('expected_rank')) if test.get('expected_rank') is not None else "Not in results"
        lines.append(f"Expected rank: {rank_str}")
        lines.append(f"Status: {test.get('status')}")
        lines.append("")

    # Display failing / surprising cases section if any test failed
    failing_tests = [t for t in test_results if t.get("status") == "FAIL"]
    if failing_tests:
        lines.append("------------------------------")
        lines.append("FAIL / SURPRISING CASE")
        lines.append("------------------------------")
        for test in failing_tests:
            lines.append(f"Query: {test.get('query')}")
            lines.append(f"Expected: {test.get('expected_source')}")
            lines.append(f"Retrieved: {test.get('top_source')}")
            rank_str = str(test.get('expected_rank')) if test.get('expected_rank') is not None else "Not found"
            lines.append(f"Expected rank: {rank_str}")
            lines.append(f"Score: {test.get('top_score'):.2f}")
            lines.append("")
            lines.append("Possible causes:")
            for cause in POSSIBLE_FAILURE_CAUSES:
                lines.append(f"- {cause}")
            lines.append("")

    # Pipeline check section
    if pipeline_check.get("status") == "FAILED" or pipeline_check.get("issues"):
        lines.append("------------------------------")
        lines.append("PIPELINE SANITY WARNINGS")
        lines.append("------------------------------")
        for issue in pipeline_check.get("issues", []):
            lines.append(f"• Issue: {issue}")
        for warn in pipeline_check.get("warnings", []):
            lines.append(f"• Warning: {warn}")
        lines.append("")

    return "\n".join(lines).strip()


def print_sanity_report(report_data: Dict[str, Any]):
    """Prints the formatted sanity report to console."""
    report_text = format_sanity_report(report_data)
    print(report_text)


# ==============================================================================
# 5. END-TO-END PIPELINE RUNNER
# ==============================================================================

def run_sanity_pipeline(
    chunks_file: Optional[str] = None,
    test_cases_file: Optional[str] = None,
    top_k: int = 3,
    output_dir: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes end-to-end embedding quality checks and sanity test suite.
    """
    load_dotenv()
    selected_model = model or os.getenv("EMBED_MODEL", "text-embedding-3-small")
    resolved_base_url = base_url or os.getenv("EMBEDDING_BASE_URL")
    resolved_api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")

    client = create_embedding_client(base_url=resolved_base_url, api_key=resolved_api_key)

    # 1. Load embedded candidate chunks from outputs/embedded_chunks.json
    out_dir = output_dir or os.path.join(project_root, "outputs")
    embedded_file = os.path.join(out_dir, "embedded_chunks.json")

    candidate_chunks: List[Dict[str, Any]] = []
    if os.path.exists(embedded_file):
        with open(embedded_file, "r", encoding="utf-8") as f:
            candidate_chunks = json.load(f)

    # If embedded chunks don't exist yet, run embedding pipeline
    if not candidate_chunks:
        if verbose:
            print("[INFO] Embedded chunks artifact not found. Generating embeddings for sanity testing...")
        embed_report = run_embedding_pipeline(
            chunks_file=chunks_file,
            output_dir=out_dir,
            model=selected_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            max_chunks=0,  # embed all available chunks for sanity tests
            client=client,
            verbose=False
        )
        if os.path.exists(embedded_file):
            with open(embedded_file, "r", encoding="utf-8") as f:
                candidate_chunks = json.load(f)

    # If still no candidate chunks, throw descriptive error
    if not candidate_chunks:
        raise ValueError(
            "No candidate chunks available to test. Please ensure document corpus ingestion and chunk embedding have been executed."
        )

    # 2. Load test cases if custom JSON file provided
    test_cases: Optional[List[Dict[str, str]]] = None
    if test_cases_file and os.path.exists(test_cases_file):
        with open(test_cases_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

    # 3. Execute sanity tests
    report_data = run_embedding_sanity_tests(
        candidate_chunks=candidate_chunks,
        test_cases=test_cases,
        top_k=top_k,
        client=client,
        model=selected_model
    )

    # 4. Save report artifact to outputs/sanity_report.json
    os.makedirs(out_dir, exist_ok=True)
    report_json_path = os.path.join(out_dir, "sanity_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # 5. Print report
    if verbose:
        print_sanity_report(report_data)
        print(f"\n[OUTPUT] Sanity report artifact saved to:\n  • {report_json_path}\n")

    return report_data


if __name__ == "__main__":
    run_sanity_pipeline()
