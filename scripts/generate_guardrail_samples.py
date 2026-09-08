"""
Script to generate sample output files demonstrating retrieval guardrail behavior:
1. outputs/sample_refusal_case.txt (Weak/Empty retrieval -> Safe Refusal)
2. outputs/sample_success_case.txt (Strong retrieval -> Grounded Answer with verified citations)
"""

import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.retrieval_guardrail import check_retrieval_quality


def generate_samples():
    os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)

    # --------------------------------------------------------------------------
    # SAMPLE 1: Weak Retrieval / Safe Refusal
    # --------------------------------------------------------------------------
    refusal_question = "What are the shipment regulations for importing live dolphins into Antarctica?"
    
    guardrail_weak = check_retrieval_quality(
        retrieved_chunks=[],
        max_distance_threshold=1.35,
        min_relevant_chunks=1
    )
    
    refusal_content = f"""Question:
{refusal_question}

Retrieved Chunks: {guardrail_weak['retrieved_count']}
Relevant Chunks: {guardrail_weak['relevant_count']}
Retrieval Quality: INSUFFICIENT
Reason: {guardrail_weak['reason']}
Guardrail Decision: {guardrail_weak['decision']}
LLM Called: No

Answer:
I don't know based on the available documents.

Citations:
None
"""

    refusal_path = os.path.join(project_root, "outputs", "sample_refusal_case.txt")
    with open(refusal_path, "w", encoding="utf-8") as f:
        f.write(refusal_content)
    print(f"[OK] Wrote sample refusal case to {refusal_path}")

    # --------------------------------------------------------------------------
    # SAMPLE 2: Strong Retrieval / Confident Grounded Answer
    # --------------------------------------------------------------------------
    success_question = "What agency is responsible for the customs requirements for laptop computers in India?"
    
    retrieved_chunks = [
        {
            "id": "c1",
            "chunk_text": "For importation of laptop computers into India under HS Code 8471, the Directorate General of Foreign Trade (DGFT) and Central Board of Indirect Taxes and Customs (CBIC) are the primary responsible regulatory agencies governing customs clearance and BIS registration requirements.",
            "source": "customs_requirements_india.txt",
            "chunk_index": 1,
            "section": "Electronics Regulations",
            "distance": 0.32,
            "similarity_score": 0.94,
            "rank": 1
        },
        {
            "id": "c2",
            "chunk_text": "All imported electronic goods and laptops into India require standard documentation including Commercial Invoice, Bill of Lading, and valid BIS registration certificate administered by CBIC customs ports.",
            "source": "shipping_rules_india.txt",
            "chunk_index": 2,
            "section": "Documentation",
            "distance": 0.58,
            "similarity_score": 0.82,
            "rank": 2
        },
        {
            "id": "c3",
            "chunk_text": "General warehousing standards for air cargo facilities.",
            "source": "warehouse_guide.txt",
            "chunk_index": 8,
            "section": "General Facilities",
            "distance": 1.65,
            "similarity_score": 0.15,
            "rank": 3
        }
    ]

    guardrail_success = check_retrieval_quality(
        retrieved_chunks=retrieved_chunks,
        max_distance_threshold=1.35,
        min_relevant_chunks=1
    )

    success_content = f"""Question:
{success_question}

Retrieved Chunks: {guardrail_success['retrieved_count']}
Relevant Chunks: {guardrail_success['relevant_count']}
Retrieval Quality: SUFFICIENT
Reason: {guardrail_success['reason']}
Guardrail Decision: {guardrail_success['decision']}
LLM Called: Yes

Answer:
According to the available shipment documentation, for laptop computers imported into India under HS Code 8471, the responsible regulatory agencies governing customs clearance and registration requirements are the Directorate General of Foreign Trade (DGFT) and the Central Board of Indirect Taxes and Customs (CBIC) [1]. Standard clearance requires a commercial invoice, bill of lading, and valid BIS registration administered by CBIC customs authorities [2].

Citations:
- [1] Source: customs_requirements_india.txt | Chunk Index: 1 | Section: Electronics Regulations | Distance: 0.32
- [2] Source: shipping_rules_india.txt | Chunk Index: 2 | Section: Documentation | Distance: 0.58
"""

    success_path = os.path.join(project_root, "outputs", "sample_success_case.txt")
    with open(success_path, "w", encoding="utf-8") as f:
        f.write(success_content)
    print(f"[OK] Wrote sample success case to {success_path}")


if __name__ == "__main__":
    generate_samples()
