import os
import sys

project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

import chromadb
from src.document_loader import load_directory, chunk_documents
from src.llm_completion import run_chat_completion
from src.context_manager import ContextManager
from src.scope_guard import is_in_scope, OUT_OF_SCOPE_RESPONSE

# Setup ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="test_eval_kb")
corpus_dir = os.path.join(project_root, "data", "sample_corpus")
raw_docs = load_directory(corpus_dir, verbose=False)
chunks = chunk_documents(raw_docs)
if collection.count() == 0:
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )

test_queries = [
    "What documents are required for international shipments?",
    "shipping cost to send a macbook to iran",
    "Who won the 2022 World Cup?",
    "What are the requirements for commercial invoices and customs duties?",
    "shipping cost to send a macbook to iran"
]

MAX_DIST = 1.35

# Load system prompt v2 constrained for CDLP / ShipRule
prompt_file = os.path.join(project_root, "prompts", "system_prompt_v2_constrained.txt")
if os.path.exists(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        sys_prompt = f.read().strip()
else:
    sys_prompt = None

for idx, q in enumerate(test_queries, 1):
    print(f"\n========================================")
    print(f"TEST {idx}: \"{q}\"")
    print(f"========================================")
    if not is_in_scope(q):
        print(f"[Scope Guard]: OUT OF SCOPE")
        print(f"{OUT_OF_SCOPE_RESPONSE}")
        continue
    
    ctx_manager = ContextManager(system_prompt=sys_prompt)
    res = collection.query(query_texts=[q], n_results=3, include=["documents", "metadatas", "distances"])
    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    dists = res["distances"][0] if res["distances"] else []
    
    rel_docs, rel_metas = [], []
    for d, m, dist in zip(docs, metas, dists):
        if dist <= MAX_DIST:
            rel_docs.append(d)
            rel_metas.append(m)
            
    print(f"[Vector DB]: Retrieved {len(rel_docs)} snippet(s) passing threshold (out of {len(docs)} candidate(s)).")
    context = "\n\n".join(rel_docs) if rel_docs else ""
    
    prep = ctx_manager.get_prepared_payload(q, retrieved_context=context)
    resp = run_chat_completion(messages_override=prep["messages"])
    
    print(f"[Raw JSON Response]: {resp}")
    print(f"--- Formatted Answer ---")
    print(f"Answer: {resp.get('answer')}")
    print(f"Confidence: {resp.get('confidence', '').upper()} | Has Answer: {resp.get('has_answer')}")
    sources = resp.get("sources", [])
    if sources:
        print("Sources:")
        for s in sources:
            print(f"  - {s.get('source')}, page {s.get('page')}")
    else:
        print("Sources: None")
