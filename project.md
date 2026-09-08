# ShipRule CDLP — Project Documentation

**ShipRule CDLP** (Customs Duty & Documentation Lookup Platform) is a production-ready, grounded Retrieval-Augmented Generation (RAG) system built for customs duty lookup, shipping documentation rules, and international trade compliance.

---

## 1. Project Architecture Overview

```text
                                  ┌────────────────────────┐
                                  │   User / Streamlit UI  │
                                  │   (app.py & static/)   │
                                  └───────────┬────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FastAPI Backend API (src/api.py)                        │
├─────────────────────────────────────────────┬───────────────────────────────────────────┤
│  POST /documents (Upload & Runtime Index)   │    POST /query (RAG Grounded Search)      │
└──────────────────────┬──────────────────────┴─────────────────────┬─────────────────────┘
                       │                                            │
                       ▼                                            ▼
┌──────────────────────────────────────────────┐ ┌────────────────────────────────────────┐
│ Document Loader & Cleaner                    │ │ Query Embedder                         │
│ (src/document_loader.py, src/text_cleaner.py)│ │ (src/embeddings.py)                    │
└──────────────────────┬───────────────────────┘ └──────────────────┬─────────────────────┘
                       │                                            │
                       ▼                                            ▼
┌──────────────────────────────────────────────┐ ┌────────────────────────────────────────┐
│ Token Chunker & Metadata Tagging             │ │ Vector / Hybrid Retrieval & Re-Ranker │
│ (src/chunker.py, src/chunk_metadata.py)      │ │ (src/retrieval.py, src/reranker.py)    │
└──────────────────────┬───────────────────────┘ └──────────────────┬─────────────────────┘
                       │                                            │
                       ▼                                            ▼
┌──────────────────────────────────────────────┐ ┌────────────────────────────────────────┐
│ Embeddings Generator (OpenAI / Local Fallback)│ │ Retrieval Quality Guardrail            │
│ (src/embeddings.py)                          │ │ (src/retrieval_guardrail.py)           │
└──────────────────────┬───────────────────────┘ └──────────────────┬─────────────────────┘
                       │                                            │
                       ▼                                            ▼
┌──────────────────────────────────────────────┐ ┌────────────────────────────────────────┐
│ Vector Database Storage                      │ │ Grounded LLM Context Assembly & Generator│
│ (src/indexing.py, src/vector_store.py)       │ │ (src/context_assembler.py, src/rag_pipeline.py)
└──────────────────────────────────────────────┘ └────────────────────────────────────────┘
```

---

## 2. Directory & File Breakdown

### Root Directory Executables & Config

* **`api_server.py`**: Entry point to launch the FastAPI backend server using Uvicorn (`http://127.0.0.1:8000`).
* **`app.py`**: Interactive Streamlit web application & CLI query interface.
* **`api_demo.py`**: Terminal demonstration script testing all REST API endpoints via FastAPI `TestClient`.
* **`conversational_rag_demo.py`**: Terminal demonstration of multi-turn conversational follow-up query rewriting and grounded QA.
* **`evaluate_rag_system.py`**: Evaluation script running grounding, source citation accuracy, and point-scoring benchmarks.
* **`evaluate_retrieval.py`**: Evaluation harness for top-k retrieval recall, precision, and re-ranking performance.
* **`ingest.py`**: CLI script executing end-to-end corpus ingestion and manifest generation.
* **`rerank_demo.py`**: Demonstration script comparing initial vector retrieval against cross-encoder re-ranked results.
* **`sanity_check.py`**: Automated pipeline sanity check verifier.
* **`requirements.txt`**: Python dependencies list (`fastapi`, `uvicorn`, `pydantic`, `chromadb`, `openai`, `groq`, `tiktoken`, `pypdf`, `fpdf2`, `pytest`, `streamlit`).
* **`.env` / `.env.example`**: Environment variable configurations (`OPENAI_API_KEY`, `GROQ_API_KEY`, `EMBED_MODEL`, `CHAT_MODEL`, `MAX_UPLOAD_SIZE_MB`).

---

### Source Package (`src/`) — Core Modules Point-to-Point

1. **`src/api.py`**
   * Exposes FastAPI REST backend endpoints.
   * `POST /query`: Accepts a question string, runs the RAG pipeline, returns grounded answer, source list, status (`answered` vs `refused`), token metrics, and guardrail decisions.
   * `POST /documents`: Accepts `.txt`, `.md`, `.pdf` files, validates file type and max size (`MAX_UPLOAD_SIZE_MB`), securely stores files in `uploads/`, loads/cleans/chunks text, generates embeddings, and updates the vector store for immediate searchability.
   * `GET /health`: Returns service health status and active configuration.
   * `GET /`: Returns API root service metadata.

2. **`src/rag_pipeline.py`**
   * Main orchestrator connecting query embedding, vector retrieval, re-ranking, retrieval guardrail checking, context assembly, grounded answer generation, and citation verification.
   * Implements anti-hallucination rules ensuring queries without sufficient context yield clear refusal responses.

3. **`src/retrieval_guardrail.py`**
   * Evaluates candidate retrieved chunks against relevance and distance thresholds.
   * Prevents hallucination by triggering a safe refusal status (`REFUSE`) when context is insufficient.

4. **`src/conversational_rag.py`**
   * Manages conversational state across multi-turn interactions.
   * Rewrites follow-up questions using conversation history into standalone search queries (`rewrite_followup`).

5. **`src/rag_evaluation.py` & `src/evaluation.py`**
   * Automated benchmark suite judging grounding accuracy, citation presence, and expected factual point coverage using LLM-as-a-Judge and rule-based checks.

6. **`src/document_loader.py`**
   * Multi-format document loader supporting `.txt`, `.md`, and `.pdf` (via `pypdf`).
   * Includes token-aware chunking (`token_chunks`, `chunk_document`) powered by `tiktoken`.

7. **`src/text_cleaner.py`**
   * Text sanitization pipeline removing excessive whitespace, corrupt characters, and formatting noise (`clean`).

8. **`src/chunker.py`**
   * Modular text chunking strategies (`paragraph`, `sentence`, `fixed_token`, `semantic_heading`).

9. **`src/chunk_metadata.py`**
   * Enriches text chunks with structured metadata (document type, section, country, HS code, chunk index, page number).

10. **`src/embeddings.py`**
    * Generates dense numerical vector embeddings.
    * Features automatic provider client initialization (`OpenAI`) with zero-cost `LocalEmbeddingClient` (ChromaDB ONNX) fallback when Groq/offline mode is configured.

11. **`src/indexing.py` & `src/vector_store.py`**
    * Vector database storage layer wrapping ChromaDB and in-memory `VectorCollection`.
    * Supports batch upsert, dimension validation, metadata normalization, and spot-check readback.

12. **`src/retrieval.py` & `src/reranker.py`**
    * Implements vector similarity search, metadata filtering, keyword match scoring, hybrid retrieval, and cross-encoder re-ranking.

13. **`src/context_assembler.py` & `src/context_manager.py`**
    * Formats retrieved context blocks with deterministic citation markers (`[1]`, `[2]`) and manages token budget bounds.

14. **`src/citation_manager.py` & `src/answer_validator.py`**
    * Builds citation mapping registries and verifies that claims made in generated answers are explicitly supported by cited sources.

15. **`src/llm_completion.py`**
    * Provides unified wrapper for Groq/OpenAI chat completion API calls.

16. **`src/prompt_templates.py`**
    * Centralized prompt system instructions, grounded response templates, and anti-hallucination constraints.

17. **`src/scope_guard.py`**
    * Classifies incoming user queries as in-scope (customs duty, shipping documentation) vs out-of-scope.

18. **`src/token_counter.py`**
    * Token count estimation and cost calculation utilities.

19. **`src/sanity_checker.py`**
    * System sanity verifier for pipeline components and embedding dimension integrity.

---

### Corpus & Storage Directories

* **`data/sample_corpus/`**: Pre-loaded document corpus containing:
  * `shipping_rules.txt`: Commercial invoice, packing list, and ocean bill of lading guidelines.
  * `customs_requirements.txt`: Country-specific customs duty rates (India HS 8471.30, Italy HS 8703, Germany HS 8541.43).
  * `international_shipping_guide.pdf`: PDF shipping guide covering container inspection and hazardous material protocols.
* **`uploads/`**: Server-side directory where user-uploaded documents from `POST /documents` are securely stored.
* **`outputs/`**: Generated system artifacts:
  * `embedded_chunks.json`: Main indexed vector store containing chunk text, vectors, and metadata.
  * `processed_chunks.json`: Pre-processed text chunks.
  * `ingestion_report.json` / `corpus_manifest.json`: Corpus ingestion tracking.
  * `rag_evaluation_results.json`: Evaluation scoring outputs.

---

## 3. REST API Endpoint Specification

### 1. `POST /documents` (Document Upload & Indexing)
* **Request**: `multipart/form-data` with `file: UploadFile` (`.txt`, `.md`, `.pdf`).
* **Validation**: Checks file extension, enforces `MAX_UPLOAD_SIZE_MB` limit, prevents path traversal, and rejects empty files.
* **Response**:
  ```json
  {
    "status": "indexed",
    "filename": "new-policy.md",
    "summary": {
      "document": "uploads/20260908_155145_e3484628_new-policy.md",
      "chunks": 12,
      "indexed": 12
    }
  }
  ```

### 2. `POST /query` (Grounded Query Search)
* **Request**:
  ```json
  {
    "question": "What shipping documentation is required for international customs clearance?",
    "use_reranking": true,
    "final_k": 3
  }
  ```
* **Response**:
  ```json
  {
    "answer": "International shipments require a commercial invoice and packing list [Source: shipping_rules.txt].",
    "sources": [
      {
        "source": "shipping_rules.txt",
        "chunk_id": "shipping_rules_chunk_1",
        "score": 0.85
      }
    ],
    "status": "answered",
    "metadata": {
      "token_usage": { "input_tokens": 350, "output_tokens": 40, "total_tokens": 390 },
      "timing": { "total_pipeline_time_ms": 450.2 },
      "guardrail_decision": "ALLOW"
    }
  }
  ```

### 3. `GET /health`
* **Response**:
  ```json
  {
    "status": "healthy",
    "service": "ShipRule RAG API",
    "environment": {
      "embedding_model_configured": true,
      "chat_model_configured": true,
      "vector_db_path": "data/vector_db",
      "collection_name": "rag_chunks",
      "max_upload_size_mb": 10
    }
  }
  ```

---

## 4. Automated Test Suite (`tests/`)

The repository includes a consolidated test suite containing **309 unit, integration, API, RAG, retrieval, and evaluation tests** in a single test file with 100% pass rate:

* `tests/test_all.py`: Unified test file covering API endpoints (`/health`, `/query`, `/documents`), document loaders, text cleaning, tiktoken chunking, chunk metadata, embeddings, vector store & indexing, similarity retrieval, re-ranking, retrieval guardrails, RAG pipeline, conversational RAG, context assembly, citation management, answer validation, RAG evaluation metrics, and end-to-end integration sanity checks.

To execute the test suite:
```bash
pytest tests/test_all.py
```

---

## 5. Key Implemented Features & Web Interface

### 1. Streamlit Web Chat & Query UI (`app.py` & `static/`)
* **Interactive Frontend Chat Interface**: Modern web app allowing users to type natural-language customs duty and shipping compliance questions.
* **Grounded Answers & Source Citation Drawer**: Formats answers with inline source tags and expandable cards showing source filename, chunk ID, relevance scores, and matched text.
* **Status Badges & Safe Refusal Display**: Highlights query processing state, latency metrics, and clear refusal alerts when questions cannot be answered from available context.

### 2. Real-Time Document Upload & Indexing (`POST /documents`)
* **Multi-Format Ingestion**: Supports `.txt`, `.md`, and `.pdf` uploads up to `MAX_UPLOAD_SIZE_MB` (default 10 MB).
* **Automated Runtime Pipeline**: Automatically sanitizes text, splits into tiktoken chunks, assigns metadata, generates embeddings, and updates vector index for immediate searchability via `/query`.

### 3. Multi-Turn Conversational RAG & Query Rewriting (`src/conversational_rag.py`)
* **Stateful Dialogue History**: Tracks rolling multi-turn user/assistant exchanges.
* **Follow-up Question Rewriting**: Uses past context to convert ambiguous user follow-ups (e.g., *"What about the invoice?"*) into standalone search queries prior to vector embedding.

### 4. Zero-Cost Local Embedding Fallback (`src/embeddings.py`)
* **Provider Flexibility**: Seamlessly uses OpenAI `text-embedding-3-small` when API keys are available, and automatically falls back to `LocalEmbeddingClient` for zero-cost offline embedding generation.

### 5. Grounding & Anti-Hallucination Guardrails (`src/retrieval_guardrail.py`)
* **Relevance & Similarity Thresholding**: Evaluates retrieved chunks against strict distance thresholds.
* **Explicit Refusal Protocol**: Refuses to output answers when vector context is irrelevant, preventing hallucinations.

### 6. Consolidated 309-Test Automation (`tests/test_all.py`)
* **Single-File Execution**: Consolidates 309 tests into 17 structured sections for fast, reliable CI/CD test runs.
