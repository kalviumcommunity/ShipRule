"""
ShipRule CDLP - Conversational RAG CLI Demonstration Script
============================================================
Demonstrates multi-turn RAG dialogue:
1. Tracks conversation history across turns.
2. Rewrites ambiguous follow-up questions into standalone retrieval queries.
3. Retrieves relevant context using rewritten queries.
4. Answers user questions using retrieved context and enforces safe refusal for weak retrieval.
5. Saves dialogue artifacts to outputs/conversational_rag_output.json and outputs/conversational_dialogue.txt.
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
from app.rag.indexing import VectorCollection
from app.rag.embeddings import generate_query_embedding
from app.rag.conversational import conversational_answer, ConversationalRAGManager


def setup_demo_knowledge_base() -> VectorCollection:
    """
    Sets up a VectorCollection pre-populated with PRD & project guidelines context.
    """
    load_dotenv()
    collection = VectorCollection(name="conversational_rag_demo_kb")

    demo_docs = [
        {
            "text": "Project Submission Guidelines & Evidence Requirements: All project submissions must include two core deliverables: 1. GitHub PR link to the open pull request in team repository. 2. A 3-5 minute video explanation screen-share walkthrough. The video must cover: why follow-up questions break naive retrieval, how follow-ups are rewritten into standalone queries, multi-turn dialogue examples, how history and token limits are balanced, and how to keep long conversations grounded. Upload video to Google Drive with link set to 'Anyone with the link can view'.",
            "source": "submission_guidelines.md",
            "section": "Evidence Requirements",
            "doc_type": "Guidelines"
        },
        {
            "text": "Customs Record for Motor Vehicles / Cars in Italy: Destination Country: Italy. Commodity: Cars / Motor Vehicles under HS Code 8703. Duty Rate: 10.0% Basic Customs Duty + 22% Value Added Tax (VAT). Required Documents: Commercial Invoice, Certificate of Origin, Bill of Lading, EU Type-Approval Certificate. Source Agency: ADM Italy / EU TARIC. Last Confirmed Date: 2026-01-15.",
            "source": "customs_reg_italy.json",
            "section": "Motor Vehicles (Italy)",
            "doc_type": "RegulationData"
        },
        {
            "text": "General International Shipping Documentation Requirements: All international shipments require standard customs clearance documents: Commercial Invoice, Packing List, Certificate of Origin, and Bill of Lading or Air Waybill. Specific high-value goods or regulated categories may require import licenses or statutory compliance registration.",
            "source": "shipping_rules.txt",
            "section": "General Customs Clearance",
            "doc_type": "Policy"
        }
    ]

    records = []
    for i, item in enumerate(demo_docs, start=1):
        vec = generate_query_embedding(item["text"])
        records.append({
            "id": f"demo_chunk_{i}",
            "vector": vec,
            "text": item["text"],
            "metadata": {
                "source": item["source"],
                "section": item["section"],
                "doc_type": item["doc_type"],
                "chunk_index": i
            }
        })

    collection.upsert(records)
    return collection


def run_conversational_rag_demo() -> Dict[str, Any]:
    """
    Executes a multi-turn RAG conversation demonstration.
    """
    print("========================================================================")
    print("           SHIPRULE CDLP - CONVERSATIONAL RAG & FOLLOW-UP CONTEXT       ")
    print("========================================================================\n")

    collection = setup_demo_knowledge_base()
    manager = ConversationalRAGManager(collection=collection, max_history_tokens=1000)

    dialogue_turns = [
        "What evidence is required for project submission?",
        "What about the video?",
        "What shipping documents are required for international customs clearance?",
        "What is the duty rate for cars in Italy?",
        "Does it apply to space shuttles?"
    ]

    demo_results = {
        "dialogue_history": [],
        "turns_detail": []
    }

    report_lines = []
    report_lines.append("========================================================================")
    report_lines.append("           CONVERSATIONAL RAG MULTI-TURN DEMONSTRATION REPORT          ")
    report_lines.append("========================================================================\n")

    for turn_num, question in enumerate(dialogue_turns, start=1):
        print(f"------------------------------------------------------------------------")
        print(f" TURN {turn_num}: User Question")
        print(f" Original Input : '{question}'")
        print(f" History Turns  : {len(manager.history) // 2}")

        # Capture history state prior to processing turn
        history_before = list(manager.history)

        t0 = time.perf_counter()
        turn_response = manager.ask(question, candidate_k=5, final_k=3)
        t1 = time.perf_counter()
        elapsed_ms = round((t1 - t0) * 1000, 2)

        rewritten = turn_response["rewritten_query"]
        answer = turn_response["answer"]
        sources = turn_response["sources"]
        chunks = turn_response["retrieved_chunks"]

        print(f" Rewritten Query: '{rewritten}'")
        print(f" Retrieved Chunks: {len(chunks)}")
        print(f" Model Answer   : {answer}")
        print(f" Sources Cited  : {len(sources)}")
        print(f" Turn Latency   : {elapsed_ms} ms")
        print(f"------------------------------------------------------------------------\n")

        # Record report block
        report_lines.append(f"--- TURN {turn_num} ---")
        report_lines.append(f"Original Question : {question}")
        report_lines.append(f"Prior History     : {history_before}")
        report_lines.append(f"Rewritten Query   : {rewritten}")
        report_lines.append(f"Retrieved Chunks  : {len(chunks)}")
        if chunks:
            for c_idx, c in enumerate(chunks, start=1):
                app = c.get("metadata", {}).get("source", "unknown")
                txt = c.get("text", c.get("chunk_text", ""))[:100]
                report_lines.append(f"  [{c_idx}] Source: {app} | Snippet: {txt}...")
        report_lines.append(f"Generated Answer  : {answer}")
        report_lines.append(f"Sources Cited     : {sources}")
        report_lines.append(f"Turn Latency      : {elapsed_ms} ms\n")

        demo_results["turns_detail"].append({
            "turn": turn_num,
            "original_question": question,
            "history_before": history_before,
            "rewritten_query": rewritten,
            "retrieved_chunks_count": len(chunks),
            "sources": sources,
            "answer": answer,
            "latency_ms": elapsed_ms
        })

    demo_results["dialogue_history"] = manager.history

    # Save output artifacts
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "conversational_rag_output.json")
    txt_path = os.path.join(output_dir, "conversational_dialogue.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(demo_results, f, indent=2, ensure_ascii=False)

    full_txt_report = "\n".join(report_lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_txt_report)

    print(f"[SUCCESS] Conversational RAG demonstration completed.")
    print(f"  Artifact JSON : {json_path}")
    print(f"  Artifact TXT  : {txt_path}")
    print("========================================================================\n")

    return demo_results


if __name__ == "__main__":
    run_conversational_rag_demo()
