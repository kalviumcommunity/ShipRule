


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

    # Add complete CDLP Customs Regulation Dataset + PRD Knowledge Base
    prd_docs = [
        # Customs Regulation Records (Live Lookup Data)
        "Customs Record for Cars / Motor Vehicles in Italy: Destination Country: Italy. Commodity: Cars / Motor Vehicles principally designed for transport of persons. HS Code: 8703 (Motor cars and other motor vehicles under 8703). Duty Rate: 10.0% Basic Customs Duty + 22% Value Added Tax (VAT). Required Documents: Commercial Invoice, Certificate of Origin, Bill of Lading, EU Type-Approval Certificate, EUR.1 Movement Certificate. Restricted Status: Unrestricted (subject to EU emissions standards and vehicle type-approval). Source Agency: Agenzia delle Dogane e dei Monopoli (ADM), Italy / European Union TARIC. Source URL: https://ec.europa.eu/taxation_customs/dds2/taric. Last Confirmed Date: 2026-01-15.",
        "Customs Record for Laptop Computers in India: Destination Country: India. Commodity: Laptop Computers / Portable Automatic Data Processing Machines. HS Code: 8471.30. Duty Rate: 7.5% Basic Customs Duty + 10% Social Welfare Surcharge (SWS). Required Documents: Commercial Invoice, Bill of Lading, BIS Registration Certificate, Certificate of Origin, DGFT Import License. Restricted Status: Restricted (Import License Required under DGFT guidelines). Source Agency: Directorate General of Foreign Trade (DGFT) & CBIC, India. Source URL: https://www.cbic.gov.in. Last Confirmed Date: 2026-02-10.",
        "Customs Record for Pharmaceuticals in United States: Destination Country: United States. Commodity: Medicaments & Pharmaceuticals. HS Code: 3004.90. Duty Rate: 0.0% (Duty Free). Required Documents: Commercial Invoice, FDA Import Declaration, Certificate of Analysis, Bill of Lading, Prior Notice Confirmation. Restricted Status: Restricted (FDA Approval & Prior Notice Required). Source Agency: U.S. Customs and Border Protection (CBP) & Food and Drug Administration (FDA). Source URL: https://www.cbp.gov. Last Confirmed Date: 2026-01-20.",
        "Customs Record for Cotton Textiles in United Kingdom: Destination Country: United Kingdom. Commodity: Trousers & Cotton Apparel. HS Code: 6204.62. Duty Rate: 12.0% Customs Duty. Required Documents: Commercial Invoice, Packing List, Certificate of Origin, Import Declaration (C88). Restricted Status: Unrestricted. Source Agency: HM Revenue & Customs (HMRC), UK. Source URL: https://www.gov.uk/trade-tariff. Last Confirmed Date: 2026-02-01.",
        "Customs Record for Solar Panels in Australia: Destination Country: Australia. Commodity: Photovoltaic Cells & Solar Panels. HS Code: 8541.40. Duty Rate: 5.0% Customs Duty. Required Documents: Commercial Invoice, Bill of Lading, Certificate of Origin, Electrical Safety Compliance Certificate. Restricted Status: Unrestricted. Source Agency: Australian Border Force (ABF). Source URL: https://www.abf.gov.au. Last Confirmed Date: 2026-02-05.",

        # PRD Core Sections
        "CDLP (Customs Duty & Documentation Lookup Platform) is a centralized customs intelligence platform that consolidates public government customs data across countries and HS codes into a single, searchable interface. It enables searching customs requirements using destination country and HS code, viewing duty rates, required import documents, restricted-item status, source agency, source URL, and last confirmed date.",
        "CDLP Business Problem: Customs regulations are fragmented across country-specific portals with varying website structures, data formats, HS-code classifications, documentation requirements, duty-rate presentations, and update frequencies. Operations staff spend ~15 minutes manually searching sources before processing shipments. CDLP reduces lookup time to 3 minutes or less with source traceability and data freshness.",
        "CDLP Core Data Bridge: CDLP connects Customs Regulation Data (duty rate, requirements), Government Source Agency, Source URL, Shipment Data (country, HS code, context), and Validation Data (verification status) to create a single auditable customs answer.",
        "CDLP User Personas: Persona 1 - Operations Staff (Primary Goal: Quickly confirm duty, docs, restrictions in <=3 mins). Persona 2 - Trade Compliance Officer (Primary Goal: Verify and maintain customs records, re-validate stale records). Persona 3 - Operations Manager (Primary Goal: Monitor lookup adoption, coverage gaps, operational bottlenecks). Persona 4 - Regional Operations Lead (Primary Goal: Compare customs requirements side-by-side across countries).",
        "CDLP Pain Points: P1-P3: Operations staff manual searching, missed import docs, incorrectly calculated duty rates (Critical). P4-P7/P10: Compliance officer repetitive queries, stale records, unmapped query tracking, centralized visibility (High). P8: Regional Ops Lead difficulty comparing country requirements (Medium). P9: Lack of single source of truth for public customs requirements (Critical).",
        "CDLP Project Goals: G1 Customs Visibility (unified view of duty, docs, restrictions), G2 Fast Lookup (lookup time <=3 mins), G3 Source Traceability (display source agency, URL, last-confirmed date), G4 Coverage Management (identify unmapped country/HS queries), G5 Data Freshness (track re-verification), G6 Operational Reporting, G7 Country Comparison, G8 Compliance Efficiency, G9 Auditability, G10 Scalable Coverage.",
        "CDLP Dataset & Schema: Customs Regulation Dataset includes country, HS code, duty rate, required documents, restricted status, source agency, source URL, last confirmed date, validation status, validated_by_1, validated_by_2. Validation rules: Every record must contain source URL and source agency; published records require two-person validation.",
        "CDLP Success Metrics & SLAs: Rule Lookup Time <=3 mins, Duty/Doc Hold Rate <=1.5%, Weekly Active Adoption >=80%, Data Freshness >=95% within SLA, Top-20 Lane Coverage 100% by launch, Query Response Time <2 seconds. Business Metrics: >=80% reduction in manual lookup time, >=50% reduction in basic compliance queries.",
        "CDLP Functional Requirements: FR-01 to FR-07 (Data Integration: store duty rates, import docs, restricted status, source agency, source URL, last confirmed date). FR-08 to FR-12 (Validation: two-person validation required before publishing, validation history, flag re-verification). FR-13 to FR-19 (Lookup: search by destination country + HS code, display duty/docs/restrictions/source, explicit Not Mapped state). FR-20 to FR-23 (Gap Detection: log unmatched queries and query frequency). FR-24 to FR-27 (Re-verification). FR-28 to FR-32 (Reporting & Country Comparison). FR-33 to FR-36 (Access Control & Audit Logs).",
        "CDLP Non-Functional Requirements: NFR-01 Lookup results within 2 seconds. NFR-02 Dashboard page load within 3 seconds. NFR-04 Distinguish mapped vs unmapped records. NFR-07 SSO authentication. NFR-08 Role-based access control (RBAC). NFR-09 Audit logs for critical changes. NFR-10 No sensitive credentials stored in source code.",
        "CDLP MVP Scope: Included in MVP: Public customs data integration, two-person validation, country + HS code lookup, duty rate, required docs, restriction status, source agency & URL, last confirmed date, unmatched query logging, re-verification alerts, weekly reports, RBAC, country comparison. Explicitly Excluded from MVP: Automated government scraping/API, FTA/preferential rates, carrier restrictions, internal policy overlays, TMS/ERP integration, mobile app, multi-language UI, AI chatbot.",
        "CDLP Future Scope & Roadmap: Phase 2 Intelligence Layer (AI Customs Assistant, Smart HS-Code Suggestions, Gap Prioritization), Phase 3 Integration & Automation (Government API, TMS/ERP integration, Automated refresh, Teams/Email alerts), Phase 4 Enterprise Readiness (Multi-language UI, Advanced Audit Trail, Enterprise Analytics, Change Detection).",
        "CDLP Risks & Mitigation: R1 Government websites change without notice (Mitigation: fixed re-verification cycle and visible last confirmed date). R2 Incorrect HS code entered (Mitigation: autocomplete & training). R3 Manual data entry errors (Mitigation: two-person validation). R4 Incomplete coverage (Mitigation: prioritize top 20 shipment lanes)."
    ]

    prd_metadatas = [
        {"section": "customs_rec_cars_italy", "doc_type": "RegulationData"},
        {"section": "customs_rec_laptops_india", "doc_type": "RegulationData"},
        {"section": "customs_rec_pharma_usa", "doc_type": "RegulationData"},
        {"section": "customs_rec_textiles_uk", "doc_type": "RegulationData"},
        {"section": "customs_rec_solar_australia", "doc_type": "RegulationData"},
        {"section": "1_executive_summary", "doc_type": "PRD"},
        {"section": "2_business_problem", "doc_type": "PRD"},
        {"section": "2.2_core_gap", "doc_type": "PRD"},
        {"section": "3_user_personas", "doc_type": "PRD"},
        {"section": "4_user_pain_points", "doc_type": "PRD"},
        {"section": "5_project_goals", "doc_type": "PRD"},
        {"section": "6_dataset_schema", "doc_type": "PRD"},
        {"section": "7_success_metrics", "doc_type": "PRD"},
        {"section": "8_functional_requirements", "doc_type": "PRD"},
        {"section": "9_non_functional_requirements", "doc_type": "PRD"},
        {"section": "11_mvp_scope", "doc_type": "PRD"},
        {"section": "12_future_scope", "doc_type": "PRD"},
        {"section": "13_risks_and_assumptions", "doc_type": "PRD"}
    ]

    prd_ids = [f"prd_sec_{i+1}" for i in range(len(prd_docs))]

    collection.add(
        documents=prd_docs,
        metadatas=prd_metadatas,
        ids=prd_ids
    )

    # Test retrieval
    results = collection.query(
        query_texts=["Customs Duty & Documentation Lookup Platform PRD"],
        n_results=1
    )

    print(
        "[Vector DB] Successfully stored CDLP PRD Knowledge Base in ChromaDB! Sample doc: "
        f"'{results['documents'][0][0][:100]}...'"
    )

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