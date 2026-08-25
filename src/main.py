


import os
import sys

# Ensure project root directory is in sys.path so `python src/main.py` works directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
import chromadb
from src.llm_completion import run_chat_completion


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

    print(f"[Config] Chat Model: {chat_model}")
    print(f"[Config] Embed Model: {embed_model}")
    print(f"[Config] Base URL: {openai_base_url}")
    print(f"[Config] API Key Set: {api_key_status}")

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
    # 3. Ask questions from terminal
    # -----------------------------------------
    print("\n========================================")
    print("       RAG QUESTION & ANSWER")
    print("========================================")
    print("Type your question below.")
    print("Type 'exit' to stop the application.")
    print("========================================")

    while True:

        question = input("\nAsk your question: ").strip()

        # Exit
        if question.lower() == "exit":
            print("\nExiting RAG Application...")
            break

        # Empty input
        if not question:
            print("Please enter a question.")
            continue

        print("\n[Vector DB] Querying ChromaDB for relevant context...")
        query_res = collection.query(
            query_texts=[question],
            n_results=2
        )

        retrieved_docs = query_res.get("documents", [[]])[0]
        context = "\n".join(retrieved_docs) if retrieved_docs else ""

        print(f"[Vector DB] Retrieved {len(retrieved_docs)} context snippet(s).")
        print("[LLM Completion] Querying Groq API...")

        try:
            # Send question and retrieved context to LLM
            response = run_chat_completion(question, context=context)

            if response:
                print("\n--- Model Response ---")
                print(response)
                print("----------------------")
            else:
                print("\n[ERROR] No response received from model.")

        except Exception as e:
            print("\n[ERROR] Failed to get response.")
            print(f"Details: {e}")


if __name__ == "__main__":
    main()