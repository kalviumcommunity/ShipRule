import os
from dotenv import load_dotenv
import chromadb
from src.llm_completion import run_chat_completion

def main():
    # Load environment variables from .env
    load_dotenv()

    print("=== RAG Application Starter ===")
    
    # Retrieve environment variables
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    print(f"[Config] Chat Model: {chat_model}")
    print(f"[Config] Embed Model: {embed_model}")
    print(f"[Config] Base URL: {openai_base_url}")
    print(f"[Config] API Key Set: {'Yes' if openai_api_key else 'No (Placeholder)'}")

    # Demonstrate ChromaDB initialization
    print("\n[Vector DB] Initializing ChromaDB client...")
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="knowledge_base_test")
    collection.add(
        documents=["RAG Application Day 1 Setup Initialized Successfully."],
        metadatas=[{"source": "setup"}],
        ids=["id1"]
    )
    results = collection.query(query_texts=["Setup"], n_results=1)
    print(f"[Vector DB] Successfully stored and queried ChromaDB doc: '{results['documents'][0][0]}'")

    print("\n[LLM Completion] Executing chat completion call...")
    run_chat_completion()

    print("\n[Status] RAG Application Foundation environment successfully verified!")

if __name__ == "__main__":
    main()

