


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

    print(f"[Config] Chat Model: {chat_model}")
    print(f"[Config] Embed Model: {embed_model}")
    print(f"[Config] Base URL: {openai_base_url}")
    print(f"[Config] API Key Set: {api_key_status}")
    print(f"[Config] Max Context Budget: {max_context_tokens} tokens (Reserve: {response_reserve_tokens})")
    print(f"[Config] History Strategy: {strategy} (Preserve Recent: {preserve_recent} turns)")

    # -----------------------------------------
    # 2. Initialize ChromaDB
    # -----------------------------------------
    print("\n[Vector DB] Initializing ChromaDB client...")

    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(
        name="knowledge_base_test"
    )

    # Add initial knowledge base documents
    collection.add(
        documents=[
            "ShipRule CDLP is an automated customs compliance and duty classification platform.",
            "Source traceability ensures every duty lookup is verified against official trade and customs regulations.",
            "RAG Application Day 1 Setup Initialized Successfully."
        ],
        metadatas=[
            {"source": "platform_overview"},
            {"source": "compliance_rules"},
            {"source": "setup"}
        ],
        ids=[
            "id1",
            "id2",
            "id3"
        ]
    )

    # Test retrieval
    results = collection.query(
        query_texts=["Setup"],
        n_results=1
    )

    print(
        "[Vector DB] Successfully stored and queried ChromaDB doc: "
        f"'{results['documents'][0][0]}'"
    )

    print("\n[Status] RAG foundation is ready!")

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
        print("[LLM Completion] Querying Groq API...")

        try:
            # Send prepared messages to LLM
            response = run_chat_completion(
                messages_override=prep["messages"],
                model_override=chat_model
            )

            if response:
                # Add to persistent context manager history
                ctx_manager.history = list(prep["messages"])
                ctx_manager.add_assistant_message(response)

                print("\n--- Model Response ---")
                print(response)
                print("----------------------")
                print(f"[History Stats] Current History Turns: {(len(ctx_manager.history) - 1) // 2} turn(s)")
            else:
                print("\n[ERROR] No response received from model.")

        except Exception as e:
            print("\n[ERROR] Failed to get response.")
            print(f"Details: {e}")


if __name__ == "__main__":
    main()