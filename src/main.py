


import os
import sys

# Ensure project root directory is in sys.path so `python src/main.py` works directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
import chromadb
from src.document_loader import load_directory, chunk_documents
from src.llm_completion import run_chat_completion
from src.context_manager import ContextManager, total_tokens
from src.scope_guard import is_in_scope, OUT_OF_SCOPE_RESPONSE
from src.token_counter import count_tokens, format_token_cost_report, INPUT_RATE, OUTPUT_RATE

# Maximum distance threshold for ChromaDB retrieval relevance (L2 distance)
MAX_DISTANCE_THRESHOLD = 1.35


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
    # 2. Initialize ChromaDB & Ingest Corpus
    # -----------------------------------------
    print("\n[Vector DB] Initializing ChromaDB client...")

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="shiprule_knowledge_base"
    )

    # Ingest document corpus from data/sample_corpus or data directory
    corpus_dir = os.path.join(project_root, "data", "sample_corpus")
    if not os.path.exists(corpus_dir):
        corpus_dir = os.path.join(project_root, "data")

    raw_docs = load_directory(corpus_dir, verbose=False)
    chunks = chunk_documents(raw_docs, max_chunk_size=500, overlap=100)

    if chunks:
        # Avoid duplicate additions if collection persists
        if collection.count() == 0:
            collection.add(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks]
            )
        print(f"[Vector DB] Ingested {len(raw_docs)} document(s) into {len(chunks)} searchable chunk(s).")
    else:
        print("[WARNING] No document chunks found to ingest into vector store.")

    print("\n[Status] RAG foundation is ready!")

    # Load system prompt v2 constrained for CDLP / ShipRule
    prompt_file = os.path.join(project_root, "prompts", "system_prompt_v2_constrained.txt")
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            sys_prompt = f.read().strip()
    else:
        sys_prompt = None

    # -----------------------------------------
    # 3. Initialize Context Manager
    # -----------------------------------------
    ctx_manager = ContextManager(
        max_context_tokens=max_context_tokens,
        response_reserve_tokens=response_reserve_tokens,
        strategy=strategy,
        num_recent_turns_preserve=preserve_recent,
        model_name=chat_model,
        system_prompt=sys_prompt
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
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        doc_list = query_res.get("documents", [[]])[0]
        meta_list = query_res.get("metadatas", [[]])[0]
        dist_list = query_res.get("distances", [[]])[0]

        # Filter chunks by distance threshold to reject irrelevant matches
        relevant_snippets = []
        retrieved_sources = []
        for doc_text, meta, dist in zip(doc_list, meta_list, dist_list):
            if dist <= MAX_DISTANCE_THRESHOLD:
                relevant_snippets.append(doc_text)
                retrieved_sources.append(meta)

        if relevant_snippets:
            context = "\n\n".join(relevant_snippets)
            print(f"[Vector DB] Retrieved {len(relevant_snippets)} relevant context snippet(s).")
        else:
            context = ""
            print("[Vector DB] No relevant context snippets found in knowledge base (distance threshold exceeded).")

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
                if isinstance(response, dict):
                    answer_text = response.get("answer", "")
                    sources_list = response.get("sources", [])
                    confidence = response.get("confidence", "low")
                    has_answer = response.get("has_answer", False)
                else:
                    answer_text = str(response)
                    sources_list = []
                    confidence = "low"
                    has_answer = False

                # Add to persistent context manager history
                ctx_manager.history = list(prep["messages"])
                ctx_manager.add_assistant_message(answer_text)

                print("\n--- Model Response ---")
                print(answer_text)
                print(f"\n[Confidence: {confidence.upper()} | Has Answer: {has_answer}]")
                if sources_list:
                    print("Sources:")
                    formatted_sources = set()
                    for src in sources_list:
                        s_name = src.get("source", "Unknown")
                        s_page = src.get("page", "1")
                        formatted_sources.add(f"  - {s_name}, page {s_page}")
                    for f_src in sorted(formatted_sources):
                        print(f_src)
                else:
                    print("Sources: None (No matching knowledge base documents cited)")
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