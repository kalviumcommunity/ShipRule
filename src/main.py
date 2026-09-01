


import os
import sys

# Ensure project root directory is in sys.path so `python src/main.py` works directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
import chromadb
from src.llm_completion import run_chat_completion
from src.context_manager import ContextManager, total_tokens
from src.scope_guard import is_in_scope, OUT_OF_SCOPE_RESPONSE
from src.token_counter import count_tokens, format_token_cost_report, INPUT_RATE, OUTPUT_RATE
from src.chunk_metadata import (
    tag_chunks,
    chunk_document_with_metadata,
    trace_chunk_source,
    create_metadata_dict,
)


def main():
    # Load environment variables from .env
    load_dotenv()

    print("=== RAG Application ===")

    # -----------------------------------------
    # 1. Load configuration
    # -----------------------------------------
    openai_base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://api.groq.com/openai/v1"
    )

    groq_api_key = os.getenv("GROQ_API_KEY", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    api_key_status = "Yes (Groq API Key)" if groq_api_key else ("Yes (OpenAI API Key)" if openai_api_key else "No (Placeholder)")

    chat_model = os.getenv(
        "CHAT_MODEL",
        "openai/gpt-oss-120b"
    )

    embed_model = os.getenv(
        "EMBED_MODEL",
        "text-embedding-3-small"
    )

    max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "4096"))
    response_reserve_tokens = int(os.getenv("RESPONSE_RESERVE_TOKENS", "500"))
    strategy = os.getenv("CONTEXT_STRATEGY", "trim")
    preserve_recent = int(os.getenv("NUM_RECENT_TURNS_PRESERVE", "2"))
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "600"))

    print(f"[Config] Chat Model: {chat_model}")
    print(f"[Config] Embed Model: {embed_model}")
    print(f"[Config] Base URL: {openai_base_url}")
    print(f"[Config] API Key Set: {api_key_status}")
    print(f"[Config] Temperature: {llm_temperature}")
    print(f"[Config] Max Output Tokens: {llm_max_tokens}")
    print(f"[Config] Max Context Budget: {max_context_tokens} tokens (Reserve: {response_reserve_tokens})")
    print(f"[Config] History Strategy: {strategy} (Preserve Recent: {preserve_recent} turns)")

    # -----------------------------------------
    # 2. Initialize ChromaDB
    # -----------------------------------------
    print("\n[Vector DB] Initializing ChromaDB client...")

    chroma_client = chromadb.Client()

    # Recreate collection to ensure clean PRD knowledge base
    try:
        chroma_client.delete_collection(name="knowledge_base_test")
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name="knowledge_base_test"
    )

    # Raw document specifications with source, section, and extra metadata for CDLP
    raw_doc_specs = [
        # Customs Regulation Records (Live Lookup Data)
        {
            "text": "Customs Record for Cars / Motor Vehicles in Italy: Destination Country: Italy. Commodity: Cars / Motor Vehicles principally designed for transport of persons. HS Code: 8703 (Motor cars and other motor vehicles under 8703). Duty Rate: 10.0% Basic Customs Duty + 22% Value Added Tax (VAT). Required Documents: Commercial Invoice, Certificate of Origin, Bill of Lading, EU Type-Approval Certificate, EUR.1 Movement Certificate. Restricted Status: Unrestricted (subject to EU emissions standards and vehicle type-approval). Source Agency: Agenzia delle Dogane e dei Monopoli (ADM), Italy / European Union TARIC. Source URL: https://ec.europa.eu/taxation_customs/dds2/taric. Last Confirmed Date: 2026-01-15.",
            "source": "customs_reg_italy.json",
            "section": "Customs Record - Motor Vehicles (Italy)",
            "doc_type": "RegulationData",
            "extra": {
                "country": "Italy",
                "hs_code": "8703",
                "source_agency": "Agenzia delle Dogane e dei Monopoli (ADM)",
                "source_url": "https://ec.europa.eu/taxation_customs/dds2/taric",
                "last_confirmed_date": "2026-01-15"
            }
        },
        {
            "text": "Customs Record for Laptop Computers in India: Destination Country: India. Commodity: Laptop Computers / Portable Automatic Data Processing Machines. HS Code: 8471.30. Duty Rate: 7.5% Basic Customs Duty + 10% Social Welfare Surcharge (SWS). Required Documents: Commercial Invoice, Bill of Lading, BIS Registration Certificate, Certificate of Origin, DGFT Import License. Restricted Status: Restricted (Import License Required under DGFT guidelines). Source Agency: Directorate General of Foreign Trade (DGFT) & CBIC, India. Source URL: https://www.cbic.gov.in. Last Confirmed Date: 2026-02-10.",
            "source": "customs_reg_india.json",
            "section": "Customs Record - Laptop Computers (India)",
            "doc_type": "RegulationData",
            "extra": {
                "country": "India",
                "hs_code": "8471.30",
                "source_agency": "Directorate General of Foreign Trade (DGFT) & CBIC",
                "source_url": "https://www.cbic.gov.in",
                "last_confirmed_date": "2026-02-10"
            }
        },
        {
            "text": "Customs Record for Pharmaceuticals in United States: Destination Country: United States. Commodity: Medicaments & Pharmaceuticals. HS Code: 3004.90. Duty Rate: 0.0% (Duty Free). Required Documents: Commercial Invoice, FDA Import Declaration, Certificate of Analysis, Bill of Lading, Prior Notice Confirmation. Restricted Status: Restricted (FDA Approval & Prior Notice Required). Source Agency: U.S. Customs and Border Protection (CBP) & Food and Drug Administration (FDA). Source URL: https://www.cbp.gov. Last Confirmed Date: 2026-01-20.",
            "source": "customs_reg_usa.json",
            "section": "Customs Record - Pharmaceuticals (USA)",
            "doc_type": "RegulationData",
            "extra": {
                "country": "United States",
                "hs_code": "3004.90",
                "source_agency": "U.S. Customs and Border Protection (CBP) & FDA",
                "source_url": "https://www.cbp.gov",
                "last_confirmed_date": "2026-01-20"
            }
        },
        {
            "text": "Customs Record for Cotton Textiles in United Kingdom: Destination Country: United Kingdom. Commodity: Trousers & Cotton Apparel. HS Code: 6204.62. Duty Rate: 12.0% Customs Duty. Required Documents: Commercial Invoice, Packing List, Certificate of Origin, Import Declaration (C88). Restricted Status: Unrestricted. Source Agency: HM Revenue & Customs (HMRC), UK. Source URL: https://www.gov.uk/trade-tariff. Last Confirmed Date: 2026-02-01.",
            "source": "customs_reg_uk.json",
            "section": "Customs Record - Cotton Textiles (UK)",
            "doc_type": "RegulationData",
            "extra": {
                "country": "United Kingdom",
                "hs_code": "6204.62",
                "source_agency": "HM Revenue & Customs (HMRC), UK",
                "source_url": "https://www.gov.uk/trade-tariff",
                "last_confirmed_date": "2026-02-01"
            }
        },
        {
            "text": "Customs Record for Solar Panels in Australia: Destination Country: Australia. Commodity: Photovoltaic Cells & Solar Panels. HS Code: 8541.40. Duty Rate: 5.0% Customs Duty. Required Documents: Commercial Invoice, Bill of Lading, Certificate of Origin, Electrical Safety Compliance Certificate. Restricted Status: Unrestricted. Source Agency: Australian Border Force (ABF). Source URL: https://www.abf.gov.au. Last Confirmed Date: 2026-02-05.",
            "source": "customs_reg_australia.json",
            "section": "Customs Record - Solar Panels (Australia)",
            "doc_type": "RegulationData",
            "extra": {
                "country": "Australia",
                "hs_code": "8541.40",
                "source_agency": "Australian Border Force (ABF)",
                "source_url": "https://www.abf.gov.au",
                "last_confirmed_date": "2026-02-05"
            }
        },

        # PRD Core Sections
        {
            "text": "CDLP (Customs Duty & Documentation Lookup Platform) is a centralized customs intelligence platform that consolidates public government customs data across countries and HS codes into a single, searchable interface. It enables searching customs requirements using destination country and HS code, viewing duty rates, required import documents, restricted-item status, source agency, source URL, and last confirmed date.",
            "source": "cdlp_prd_v1.0.md",
            "section": "1. Executive Summary",
            "doc_type": "PRD",
            "extra": {"page": 1}
        },
        {
            "text": "CDLP Business Problem: Customs regulations are fragmented across country-specific portals with varying website structures, data formats, HS-code classifications, documentation requirements, duty-rate presentations, and update frequencies. Operations staff spend ~15 minutes manually searching sources before processing shipments. CDLP reduces lookup time to 3 minutes or less with source traceability and data freshness.",
            "source": "cdlp_prd_v1.0.md",
            "section": "2. Business Problem",
            "doc_type": "PRD",
            "extra": {"page": 1}
        },
        {
            "text": "CDLP Core Data Bridge: CDLP connects Customs Regulation Data (duty rate, requirements), Government Source Agency, Source URL, Shipment Data (country, HS code, context), and Validation Data (verification status) to create a single auditable customs answer.",
            "source": "cdlp_prd_v1.0.md",
            "section": "2.2 The Core Gap",
            "doc_type": "PRD",
            "extra": {"page": 2}
        },
        {
            "text": "CDLP User Personas: Persona 1 - Operations Staff (Primary Goal: Quickly confirm duty, docs, restrictions in <=3 mins). Persona 2 - Trade Compliance Officer (Primary Goal: Verify and maintain customs records, re-validate stale records). Persona 3 - Operations Manager (Primary Goal: Monitor lookup adoption, coverage gaps, operational bottlenecks). Persona 4 - Regional Operations Lead (Primary Goal: Compare customs requirements side-by-side across countries).",
            "source": "cdlp_prd_v1.0.md",
            "section": "3. User Personas",
            "doc_type": "PRD",
            "extra": {"page": 2}
        },
        {
            "text": "CDLP Pain Points: P1-P3: Operations staff manual searching, missed import docs, incorrectly calculated duty rates (Critical). P4-P7/P10: Compliance officer repetitive queries, stale records, unmapped query tracking, centralized visibility (High). P8: Regional Ops Lead difficulty comparing country requirements (Medium). P9: Lack of single source of truth for public customs requirements (Critical).",
            "source": "cdlp_prd_v1.0.md",
            "section": "4. User Pain Points",
            "doc_type": "PRD",
            "extra": {"page": 3}
        },
        {
            "text": "CDLP Project Goals: G1 Customs Visibility (unified view of duty, docs, restrictions), G2 Fast Lookup (lookup time <=3 mins), G3 Source Traceability (display source agency, URL, last-confirmed date), G4 Coverage Management (identify unmapped country/HS queries), G5 Data Freshness (track re-verification), G6 Operational Reporting, G7 Country Comparison, G8 Compliance Efficiency, G9 Auditability, G10 Scalable Coverage.",
            "source": "cdlp_prd_v1.0.md",
            "section": "5. Project Goals",
            "doc_type": "PRD",
            "extra": {"page": 3}
        },
        {
            "text": "CDLP Dataset & Schema: Customs Regulation Dataset includes country, HS code, duty rate, required documents, restricted status, source agency, source URL, last confirmed date, validation status, validated_by_1, validated_by_2. Validation rules: Every record must contain source URL and source agency; published records require two-person validation.",
            "source": "cdlp_prd_v1.0.md",
            "section": "6. Dataset & Data Source Documentation",
            "doc_type": "PRD",
            "extra": {"page": 4}
        },
        {
            "text": "CDLP Success Metrics & SLAs: Rule Lookup Time <=3 mins, Duty/Doc Hold Rate <=1.5%, Weekly Active Adoption >=80%, Data Freshness >=95% within SLA, Top-20 Lane Coverage 100% by launch, Query Response Time <2 seconds. Business Metrics: >=80% reduction in manual lookup time, >=50% reduction in basic compliance queries.",
            "source": "cdlp_prd_v1.0.md",
            "section": "7. Success Metrics",
            "doc_type": "PRD",
            "extra": {"page": 4}
        },
        {
            "text": "CDLP Functional Requirements: FR-01 to FR-07 (Data Integration: store duty rates, import docs, restricted status, source agency, source URL, last confirmed date). FR-08 to FR-12 (Validation: two-person validation required before publishing, validation history, flag re-verification). FR-13 to FR-19 (Lookup: search by destination country + HS code, display duty/docs/restrictions/source, explicit Not Mapped state). FR-20 to FR-23 (Gap Detection: log unmatched queries and query frequency). FR-24 to FR-27 (Re-verification). FR-28 to FR-32 (Reporting & Country Comparison). FR-33 to FR-36 (Access Control & Audit Logs).",
            "source": "cdlp_prd_v1.0.md",
            "section": "8. Functional Requirements",
            "doc_type": "PRD",
            "extra": {"page": 5}
        },
        {
            "text": "CDLP Non-Functional Requirements: NFR-01 Lookup results within 2 seconds. NFR-02 Dashboard page load within 3 seconds. NFR-04 Distinguish mapped vs unmapped records. NFR-07 SSO authentication. NFR-08 Role-based access control (RBAC). NFR-09 Audit logs for critical changes. NFR-10 No sensitive credentials stored in source code.",
            "source": "cdlp_prd_v1.0.md",
            "section": "9. Non-Functional Requirements",
            "doc_type": "PRD",
            "extra": {"page": 6}
        },
        {
            "text": "CDLP MVP Scope: Included in MVP: Public customs data integration, two-person validation, country + HS code lookup, duty rate, required docs, restriction status, source agency & URL, last confirmed date, unmatched query logging, re-verification alerts, weekly reports, RBAC, country comparison. Explicitly Excluded from MVP: Automated government scraping/API, FTA/preferential rates, carrier restrictions, internal policy overlays, TMS/ERP integration, mobile app, multi-language UI, AI chatbot.",
            "source": "cdlp_prd_v1.0.md",
            "section": "11. MVP Scope",
            "doc_type": "PRD",
            "extra": {"page": 7}
        },
        {
            "text": "CDLP Future Scope & Roadmap: Phase 2 Intelligence Layer (AI Customs Assistant, Smart HS-Code Suggestions, Gap Prioritization), Phase 3 Integration & Automation (Government API, TMS/ERP integration, Automated refresh, Teams/Email alerts), Phase 4 Enterprise Readiness (Multi-language UI, Advanced Audit Trail, Enterprise Analytics, Change Detection).",
            "source": "cdlp_prd_v1.0.md",
            "section": "12. Future Scope",
            "doc_type": "PRD",
            "extra": {"page": 8}
        },
        {
            "text": "CDLP Risks & Mitigation: R1 Government websites change without notice (Mitigation: fixed re-verification cycle and visible last confirmed date). R2 Incorrect HS code entered (Mitigation: autocomplete & training). R3 Manual data entry errors (Mitigation: two-person validation). R4 Incomplete coverage (Mitigation: prioritize top 20 shipment lanes).",
            "source": "cdlp_prd_v1.0.md",
            "section": "13. Risks and Assumptions",
            "doc_type": "PRD",
            "extra": {"page": 8}
        }
    ]

    # Process all raw specs through chunk_document_with_metadata to produce tagged chunks
    tagged_corpus = []
    for spec in raw_doc_specs:
        chunks = chunk_document_with_metadata(
            text=spec["text"],
            source=spec["source"],
            section=spec["section"],
            doc_type=spec["doc_type"],
            chunk_size=350,
            chunk_overlap=50,
            page=spec.get("extra", {}).get("page", 1),
            extra_metadata=spec.get("extra", {})
        )
        tagged_corpus.extend(chunks)

    prd_docs = [c["text"] for c in tagged_corpus]
    prd_metadatas = [c["metadata"] for c in tagged_corpus]
    prd_ids = [f"chunk_id_{i+1}" for i in range(len(tagged_corpus))]

    collection.add(
        documents=prd_docs,
        metadatas=prd_metadatas,
        ids=prd_ids
    )

    # Test retrieval and trace sample chunk
    results = collection.query(
        query_texts=["Customs Duty & Documentation Lookup Platform PRD"],
        n_results=1
    )

    sample_meta = results["metadatas"][0][0]
    sample_trace = trace_chunk_source(sample_meta)

    print(
        f"[Vector DB] Stored {len(tagged_corpus)} metadata-tagged chunks in ChromaDB!"
    )
    print(f"[Source Traceback Sample] {sample_trace['formatted_citation']}")

    print("\n[Status] RAG foundation with CDLP PRD is ready!")

    # -----------------------------------------
    # 3. Initialize Context Manager
    # -----------------------------------------
    ctx_manager = ContextManager(
        max_context_tokens=max_context_tokens,
        response_reserve_tokens=response_reserve_tokens,
        strategy=strategy,
        num_recent_turns_preserve=preserve_recent,
        model_name=chat_model
    )

    # -----------------------------------------
    # 4. Ask questions from terminal
    # -----------------------------------------
    print("\n========================================")
    print("       RAG QUESTION & ANSWER")
    print("========================================")
    print("Type your question below.")
    print("Type 'reset' to clear conversation history.")
    print("Type 'exit' to stop the application.")
    print("========================================")

    while True:

        question = input("\nAsk your question: ").strip()

        # Exit
        if question.lower() == "exit":
            print("\nExiting RAG Application...")
            break

        # Reset history
        if question.lower() == "reset":
            ctx_manager.reset_history()
            print("\n[Context Manager] Conversation history reset.")
            continue

        # Empty input
        if not question:
            print("Please enter a question.")
            continue

        # -----------------------------------------
        # Strict Scope Guard Check
        # -----------------------------------------
        if not is_in_scope(question):
            print(f"\n{OUT_OF_SCOPE_RESPONSE}")
            continue

        print("\n[Vector DB] Querying ChromaDB for relevant context...")
        query_res = collection.query(
            query_texts=[question],
            n_results=2
        )

        retrieved_docs = query_res.get("documents", [[]])[0]
        retrieved_metas = query_res.get("metadatas", [[]])[0]

        citations = []
        for idx, (doc_text, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
            trace_info = trace_chunk_source(meta)
            citations.append(trace_info["formatted_citation"])
            print(f"[Chunk #{idx+1} Traceback] {trace_info['formatted_citation']}")

        context = "\n".join(retrieved_docs) if retrieved_docs else ""

        print(f"[Vector DB] Retrieved {len(retrieved_docs)} context snippet(s).")

        # Prepare context payload via ContextManager
        prep = ctx_manager.get_prepared_payload(question, retrieved_context=context)
        print(f"[Context Budget] Tokens: {prep['total_tokens']}/{prep['budget']} | Strategy: {prep['strategy_applied']}")
        print(f"[LLM Completion] Querying Groq API (Temperature: {llm_temperature}, Max Tokens: {llm_max_tokens})...")

        try:
            # Send prepared messages to LLM
            response = run_chat_completion(
                messages_override=prep["messages"],
                model_override=chat_model,
                temperature_override=llm_temperature,
                max_tokens_override=llm_max_tokens
            )

            if response:
                # Handle structured JSON response dict
                if isinstance(response, dict):
                    answer_text = response.get("answer", "")
                    source_info = response.get("source", "CDLP System")
                else:
                    answer_text = str(response)
                    source_info = "CDLP System"

                # Add to persistent context manager history
                ctx_manager.history = list(prep["messages"])
                ctx_manager.add_assistant_message(answer_text)

                print("\n--- Model Response ---")
                print(answer_text)
                if source_info:
                    print(f"\n[Source Citation: {source_info}]")
                print("----------------------")
                print(f"[History Stats] Current History Turns: {(len(ctx_manager.history) - 1) // 2} turn(s)")

                # Calculate Output Tokens & Report Token Usage & Cost
                input_tokens = prep["total_tokens"]
                output_tokens = count_tokens(answer_text)
                cost_report = format_token_cost_report(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    input_rate=INPUT_RATE,
                    output_rate=OUTPUT_RATE
                )
                print(cost_report)
            else:
                print("\n[ERROR] No response received from model.")

        except Exception as e:
            print("\n[ERROR] Failed to get response.")
            print(f"Details: {e}")


if __name__ == "__main__":
    main()