"""
Script to generate sample augmented prompt from live indexed vector retrieval.
Outputs:
- outputs/sample_augmented_prompt.txt
- outputs/sample_augmented_prompt.json
"""

import os
import sys
import json

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
from app.rag.retrieval import load_indexed_vector_collection, retrieve
from app.rag.context_assembler import assemble_context, DEFAULT_GROUNDING_INSTRUCTIONS


def generate_sample_augmented_prompt(
    query: str = "shipment rules in india",
    k: int = 3,
    max_context_tokens: int = 4096,
    response_reserve_tokens: int = 500
) -> dict:
    load_dotenv()
    collection = load_indexed_vector_collection()

    # Retrieve real chunks from vector collection
    retrieved_chunks = retrieve(query=query, k=k, collection=collection)

    # Assemble grounded context with token budget
    assembly_result = assemble_context(
        retrieved_chunks=retrieved_chunks,
        user_question=query,
        max_context_tokens=max_context_tokens,
        response_reserve_tokens=response_reserve_tokens,
        system_instructions=DEFAULT_GROUNDING_INSTRUCTIONS
    )

    budget = assembly_result.budget_info

    # Format text artifact matching required specification
    txt_lines = [
        "========================================",
        "SHIPRULE SAMPLE AUGMENTED PROMPT",
        "========================================",
        "Question:",
        query,
        "",
        "Retrieved Chunks:",
        str(budget.get("retrieved_chunks_count", len(retrieved_chunks))),
        "",
        "Included Chunks:",
        str(budget.get("included_chunks_count", assembly_result.selected_chunk_count)),
        "",
        "Configured Context Budget:",
        str(budget.get("max_context_tokens", max_context_tokens)),
        "",
        "Response Reserve:",
        str(budget.get("response_reserve_tokens", response_reserve_tokens)),
        "",
        "Instruction Tokens:",
        str(budget.get("instruction_tokens", 0)),
        "",
        "Question Tokens:",
        str(budget.get("question_tokens", 0)),
        "",
        "Context Tokens:",
        str(budget.get("context_tokens", assembly_result.token_count)),
        "",
        "Estimated Total:",
        str(budget.get("total_estimated_tokens", 0)),
        "",
        "Budget Check:",
        budget.get("budget_check", "PASS"),
        "",
        "AUGMENTED PROMPT",
        "----------------------------------------",
        "[GROUNDING INSTRUCTIONS]",
        DEFAULT_GROUNDING_INSTRUCTIONS,
        "",
        "[CONTEXT]",
        assembly_result.context,
        "",
        "[USER QUESTION]",
        query,
        "========================================"
    ]

    txt_content = "\n".join(txt_lines)

    json_payload = {
        "user_question": query,
        "retrieved_chunks_count": len(retrieved_chunks),
        "included_chunks_count": assembly_result.selected_chunk_count,
        "source_markers": [s["marker"] for s in assembly_result.source_mapping],
        "source_mapping": assembly_result.source_mapping,
        "token_budget": budget,
        "assembled_context": assembly_result.context,
        "final_augmented_prompt": txt_content,
        "retrieved_chunks": retrieved_chunks
    }

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, "sample_augmented_prompt.txt")
    json_path = os.path.join(output_dir, "sample_augmented_prompt.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Sample augmented prompt generated:")
    print(f"  TXT : {txt_path}")
    print(f"  JSON: {json_path}")
    print("\n--- SAMPLE OUTPUT PREVIEW ---")
    print(txt_content)

    return json_payload


if __name__ == "__main__":
    generate_sample_augmented_prompt()
