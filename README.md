# ShipRule: Customs Duty & Documentation Lookup Platform (CDLP)

A production-ready Retrieval-Augmented Generation (RAG) system engineered for shipping rules, customs requirements, tariff classification, and international freight documentation lookup using LLMs and ChromaDB.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Quick Start & Setup Guide](#quick-start--setup-guide)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create & Activate Virtual Environment](#2-create--activate-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Environment Variables Setup](#4-environment-variables-setup)
- [Architecture & Implemented Modules](#architecture--implemented-modules)
  - [1. LLM Chat Completion & API Error Handling](#1-llm-chat-completion--api-error-handling)
  - [2. Tokens, Tokenization & Cost Estimation](#2-tokens-tokenization--cost-estimation)
  - [3. Multi-Format Document Loader & Intake Engine](#3-multi-format-document-loader--intake-engine)
  - [4. Document Chunking Strategies](#4-document-chunking-strategies)
  - [5. Corpus Preparation & Ingestion Validation](#5-corpus-preparation--ingestion-validation)
  - [6. Generating Embeddings via API](#6-generating-embeddings-via-api)
- [End-to-End RAG Execution](#end-to-end-rag-execution)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Security Guidelines](#security-guidelines)
- [Reproducibility & Best Practices](#reproducibility--best-practices)

---

## Project Overview

ShipRule delivers deterministic, hallucination-resistant query answering over complex shipping directives, customs compliance codes (HS Codes), and international logistics guides.

The application architecture encompasses:
- **Multi-Format Document Intake**: Robust ingestion of `.txt`, `.pdf`, and `.md` policy files with whitespace normalization and failure isolation.
- **Configurable Chunking Engine**: Fixed-size, paragraph-based, and sentence-based chunking with boundary inspection and metadata preservation.
- **Corpus Ingestion & Reconciliation**: End-to-end manifest tracking with mathematical zero-drop reconciliation assertions ($Discovered = Success + Failed + Skipped$).
- **API-Driven Embeddings**: OpenAI-compatible embedding generation converting textual chunks into dense numerical vectors while preserving source traceability.
- **Context & Token Budget Management**: History trimming, token tracking via `tiktoken`, and real-time per-query API cost calculation.
- **Vector Storage Foundation**: ChromaDB integration for dense semantic similarity retrieval.

---

## Folder Structure

```
ShipRule/
├── app.py                      # Main RAG interactive terminal session entry point
├── chunker.py                  # Document chunking CLI entry point
├── ingest.py                   # Corpus ingestion & validation pipeline CLI entry point
├── embed.py                    # Embedding generation CLI entry point
├── document_loader.py          # Document intake demonstration CLI entry point
├── data/
│   └── sample_corpus/          # ShipRule knowledge base documents (TXT, PDF)
│       ├── customs_requirements.txt
│       ├── international_shipping_guide.pdf
│       └── shipping_rules.txt
├── src/                        # Core modular engine components
│   ├── __init__.py
│   ├── main.py                 # Interactive RAG CLI loop with scope guard & ChromaDB
│   ├── chunker.py              # Chunking strategies, stats & boundary inspection
│   ├── ingestion.py            # Corpus ingestion, manifest & reconciliation validation
│   ├── embeddings.py           # Embeddings API client, vector generation & validation
│   ├── document_loader.py      # Multi-format plain-text extraction (PDF, TXT, MD)
│   ├── text_cleaner.py         # Whitespace normalization & text sanitization
│   ├── token_counter.py        # Token counting & model cost estimation
│   ├── context_manager.py      # Multi-turn history management & token budgets
│   ├── llm_completion.py       # OpenAI/Groq API client wrapper & error diagnostics
│   ├── prompt_templates.py     # System & user prompt templates
│   └── scope_guard.py          # Query guardrails for out-of-scope filtering
├── prompts/                    # System prompt specifications & constraints
│   ├── system_prompt_v1.txt
│   └── system_prompt_v2_constrained.txt
├── outputs/                    # Generated manifests, chunks, embeddings, and reports
│   ├── corpus_manifest.json
│   ├── ingestion_report.json
│   ├── ingestion_failures.json
│   ├── processed_chunks.json
│   ├── ingestion_log.txt
│   ├── embedded_chunks.json
│   ├── embedding_report.json
│   ├── sample_embedding_output.json
│   ├── chunks_fixed.json
│   ├── chunks_paragraph.json
│   ├── chunks_sentence.json
│   └── chunking_report.json
├── tests/                      # Comprehensive unit test suite (115 passing tests)
│   ├── test_embeddings.py
│   ├── test_ingestion.py
│   ├── test_chunking.py
│   ├── test_document_loader.py
│   ├── test_text_cleaner.py
│   ├── test_context_manager.py
│   ├── test_token_cost.py
│   ├── test_prompt_templates.py
│   ├── test_scope_guard.py
│   └── test_structured_output.py

├── .env.example                # Environment variable configuration template
├── .gitignore                  # Git ignore rules for secrets, venv, and local outputs
├── requirements.txt            # Pinned dependency specifications
└── README.md                   # Complete project documentation
```

---

## Prerequisites

- **Python**: Version 3.10, 3.11, 3.12, or 3.13
- **Git**: Installed and configured
- **API Key**: Groq API Key or OpenAI API Key


---

## Quick Start & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/kalviumcommunity/ShipRule.git
cd ShipRule
git checkout feature/projectsetup
```

### 2. Create & Activate Virtual Environment

It is recommended to use an isolated Python virtual environment (`.venv`) to manage dependencies.

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the pinned dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Environment Variables Setup

Copy `.env.example` to create your local `.env` file:

**On Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**On macOS / Linux:**
```bash
cp .env.example .env
```

Edit your `.env` file with your configuration:
```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_openai_api_key_here
CHAT_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
```

> ⚠️ **Security Warning**: The `.env` file is excluded from Git tracking in `.gitignore`. Never commit API keys or sensitive secrets to version control.

---

---

## Architecture & Implemented Modules

### 1. LLM Chat Completion & API Error Handling

Dedicated LLM client wrapper interfacing with OpenAI/Groq endpoints with automated error diagnostics and JSON response parsing.

- **Module**: `src/llm_completion.py`
- **Features**:
  - System and user prompt role separation.
  - Formatted request/response logging with token metrics.
  - Actionable diagnostics for HTTP 401 (invalid/missing API key) and HTTP 429 (rate limits/quota exhaustion).
- **Execution Command**:
  ```bash
  python src/llm_completion.py
  ```

---

### 2. Tokens, Tokenization & Cost Estimation

Implements deterministic token counting using `tiktoken` (`cl100k_base` / `o200k_base`) and dynamic cost modeling for prompt inputs vs. generated outputs.

- **Module**: `src/token_counter.py`
- **Features**:
  - Exact token calculation for single strings and multi-turn message arrays.
  - Per-model pricing tracking ($/1M tokens).
  - Terminal-formatted token cost reports generated for every user query turn.
- **Execution Command**:
  ```bash
  python src/token_counter.py
  ```

---

### 3. Multi-Format Document Loader & Intake Engine

Multi-format document extractor that normalizes heterogeneous file formats into sanitized plain-text representations with source identity preservation.

- **Module**: `src/document_loader.py` (CLI wrapper: `document_loader.py`)
- **Supported Formats**: `.txt` (Plain text), `.pdf` (PDF via `pypdf`), and `.md` (Markdown).
- **Sample Corpus**: `data/sample_corpus/` (`shipping_rules.txt`, `customs_requirements.txt`, `international_shipping_guide.pdf`).
- **Features**:
  - Plain-text conversion with page marker preservation (`[Page N]`).
  - Whitespace cleaning and regex-based artifact stripping.
  - Graceful handling of missing, corrupt, or unsupported files with warnings without interrupting batch processing.
- **Execution Command**:
  ```bash
  python document_loader.py
  ```
  or
  ```bash
  python -m src.document_loader
  ```

---

### 4. Document Chunking Strategies

Partitions extracted text into structured semantic units suitable for dense vector embeddings and semantic retrieval in ChromaDB.

- **Module**: `src/chunker.py` (CLI wrapper: `chunker.py`)
- **Strategies**:
  1. **Fixed-Size Chunking (`fixed`)**: Uniform character window slices (`size=500`, `overlap=50`).
  2. **Paragraph-Based Chunking (`paragraph`)**: Splits along natural paragraph boundaries (`\n\n+`), keeping thematic clauses intact.
  3. **Sentence-Based Chunking (`sentence`)**: Splits on grammatical sentence boundaries without cutting clauses in the middle.
- **Execution Command**:
  ```bash
  python chunker.py
  ```
  or
  ```bash
  python -m src.chunker
  ```
- **Generated Artifacts in `outputs/`**:
  - `chunks_fixed.json`, `chunks_paragraph.json`, `chunks_sentence.json`, `chunking_report.json`.

#### Corpus Comparison Benchmark

| Strategy | Total Chunks | Avg Chunk Size | Context Preservation | Main Advantage | Main Limitation |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Fixed-Size** | 8 | 411.8 chars | Low - Moderate | Uniform, predictable bounds | Slices sentences & ideas mid-clause |
| **Paragraph-Based** | 8 | 379.2 chars | High | Preserves complete semantic sections | Variable chunk lengths |
| **Sentence-Based** | 21 | 143.9 chars | Moderate - High | Preserves clause syntax | Generates many small fragments |

> **Corpus Recommendation**: **Paragraph-Based Chunking** is recommended because shipping and customs documents consist of structured policy sections where preserving complete paragraphs maintains required context for vector retrieval.

---

### 5. Corpus Preparation & Ingestion Validation

An automated, resumable ingestion pipeline that connects file discovery, loading, cleaning, chunking, metadata tagging, and zero-drop reconciliation validation.

- **Module**: `src/ingestion.py` (CLI wrapper: `ingest.py`)
- **Features**:
  - **Recursive Discovery**: Scans all subdirectories while ignoring hidden files.
  - **Manifest Tracking (`corpus_manifest.json`)**: Every file is tracked with status (`SUCCESS`, `FAILED`, or `SKIPPED`), character count, chunk count, and SHA-256 hash.
  - **Zero-Drop Reconciliation Check**: Enforces $\text{Discovered} = \text{Success} + \text{Failed} + \text{Skipped}$.
  - **Metadata & Chunk Validation**: Verifies unique chunk IDs, complete metadata keys, positive character counts, and non-empty text.
  - **Resumable Ingestion (`--resume`)**: Uses SHA-256 hashes to reuse existing chunks for unchanged files, accelerating batch updates.
- **Execution Command**:
  ```bash
  python ingest.py
  ```
  or
  ```bash
  python ingest.py --resume
  ```
- **Generated Artifacts in `outputs/`**:
---

### 6. Concept 13 — Token-Aware Chunk Sizing & Overlap

#### What is Token-Based Chunking?
Token-based chunking is the process of splitting text into chunks based on **token count** as determined by a tokenizer (`tiktoken` `cl100k_base`), rather than arbitrary character or word counts.

#### Why Characters Are Not Reliable for Model Context
Large Language Models (LLMs) and embedding models operate exclusively on **tokens**, not characters or words. Character count varies wildly based on text structure:
- **Standard English**: ~4 characters per token.
- **Code & Punctuation**: ~1.5 to 2.5 characters per token.
- **Non-Latin Scripts (e.g. Hindi, Chinese)**: Multiple bytes/tokens per character.

Chunking by character length risks either overflowing the model's context window (for token-dense text) or producing unnecessarily tiny chunks (for token-sparse text). Determining chunk sizes directly with `tiktoken` guarantees strict context window compliance.

#### What are Tokens?
Tokens are the fundamental subword units into which text is broken down before processing by neural networks. Common words may be a single token (e.g., `" shipping"`), while rare terms, code, or non-English scripts are split into subword fragments.

#### Why Overlap is Needed
When documents are split into discrete chunks, information located near the edge/boundary of a chunk can be cleanly severed. Overlap repeats a fixed window of tokens across consecutive chunks, ensuring that key context and semantic relationships spanning chunk boundaries are preserved in both chunks.

---

#### How the Implementation Works
The core function `token_chunks()` in [`src/document_loader.py`](file:///c:/Users/hp/Desktop/ShipRule/src/document_loader.py) uses `tiktoken` (`cl100k_base`):

```python
def token_chunks(text: str, size: int = 400, overlap: int = 60, encoding_name: str = "cl100k_base") -> List[Dict[str, Any]]:
```

##### Step-by-Step Movement
1. The input string is encoded into token IDs via `enc.encode(text)`.
2. A sliding window iterates over token IDs with step size `step = size - overlap` (`400 - 60 = 340`).
3. Each token range `tokens[start:end]` is decoded back into text using `enc.decode()`.
4. Detailed metadata is returned for each chunk:
   ```json
   {
     "chunk_id": 1,
     "text": "...",
     "token_count": 400,
     "start_token": 0,
     "end_token": 400,
     "overlap": 60
   }
   ```

---

#### Overlap Demonstration (400-Token Chunks, 60-Token Overlap)
- **Chunk 1**: Token range `0 – 399` (400 tokens)
- **Chunk 2**: Token range `340 – 739` (400 tokens)
- **Shared Context Window**: Tokens `340 – 399` (60 tokens repeated in both chunks).

Information near token 380 is available in both Chunk 1 and Chunk 2, so retrieval matches either chunk without losing boundary context.

---

#### Overlap Value Comparison

Running an 865-token document across different overlap values yields empirical metrics:

| Overlap (Tokens) | Chunks Generated | Total Tokens | Duplicated Tokens | Overhead (%) |
| :--- | :--- | :--- | :--- | :--- |
| **0 tokens** | 3 | 865 | 0 | 0.00% |
| **20 tokens** | 3 | 905 | 40 | 4.62% |
| **40 tokens** | 3 | 945 | 80 | 9.25% |
| **60 tokens** | 3 | 985 | 120 | 13.87% |

---

#### Cost of Overlap vs. Context Preservation

Higher overlap creates a direct tradeoff:

$$\text{More Repeated Text} \longrightarrow \text{More Chunks} \longrightarrow \text{More Embeddings} \longrightarrow \text{Higher Vector Storage \& Retrieval Cost}$$

- **Too Little Overlap (e.g. 0 tokens)**: Lowest cost, but critical facts across boundaries are split and missed by vector search.
- **Too Much Overlap (e.g. > 30%)**: Excellent context preservation, but excessive duplicate embeddings increase database size and search latency.
- **Balance**: Overlap balances **Context Preservation $\leftrightarrow$ Processing Cost**.

---

#### Boundary Context Example (Case A vs. Case B)

##### Case A — No Overlap (0 tokens)
A critical rule line `"CRITICAL REQUIREMENT: For all high-value electronic imports... MUST be attached to Commercial Invoice"` falls exactly at token 400.
- **Chunk 1 Tail**: `"...in verifying cross-border documentation. All shipments passing through international trade corridors must strictly"`
- **Chunk 2 Head**: `"adhere to import regulations. ShipRule CDLP is an automated customs compliance..."`
- **Result**: The requirement sentence is broken. Neither chunk contains the full context required to answer a search query.

##### Case B — 60-Token Overlap
- **Chunk 1 Tail**: Contains `tokens 340-400`.
- **Chunk 2 Head**: Starts at `token 340`, repeating `tokens 340-400` as context before continuing.
- **Result**: The entire sentence and its preceding context are completely preserved inside Chunk 2. RAG vector retrieval successfully matches Chunk 2.

---

#### Relationship Between Chunk Size, Top-k, and Context Window

Chunk size cannot be chosen independently of vector retrieval `top_k` and the model's max context limit:

$$\text{(Chunk Size } \times \text{ top\_k) } + \text{ System Instructions } + \text{ User Query } \le \text{ Model Context Limit}$$

For example:
- **Chunk Size**: `400` tokens
- **Top-K Chunks**: `5`
- **Retrieved Context**: $400 \times 5 = 2,000$ tokens.
- Adding prompt instructions (~500 tokens) and user query (~100 tokens) gives ~2,600 tokens total, comfortably fitting within standard context windows (e.g. 4,096 or 128k).

---

#### Rationale for 400 Tokens / 60 Tokens Initial Configuration

- **Overlap Ratio**: $60 / 400 \times 100 = 15\%$.
- **Starting Benchmark**: 10%–15% overlap is the standard recommended baseline for RAG pipelines.
- **Flexibility**: 400 tokens is large enough to contain complete technical paragraphs while small enough to allow retrieving multiple distinct chunks (`top_k=3 to 5`) within token budget.
- *Note*: 400/60 is a starting default and should be tuned based on specific document types, embedding models, and context limits.

---

#### How to Run Implementation & Tests

##### Run Token Chunking Demonstration:
```bash
python src/token_chunk_demo.py
```
Output is printed to the terminal and saved to [`outputs/token_chunking_results.txt`](file:///c:/Users/hp/Desktop/ShipRule/outputs/token_chunking_results.txt).

##### Run Unit Tests:
```bash
pytest tests/test_token_chunks.py
```

---

### 6. Generating Embeddings via API

Transforms validated text chunks into dense mathematical vector representations ($d=1536$ for `text-embedding-3-small`) to enable cosine similarity and dense semantic retrieval in ChromaDB.

#### Why ShipRule Converts Chunks into Vectors
- **Semantic Understanding**: Keyword search fails when users ask questions with synonyms (e.g. searching *"air waybill"* vs. *"freight manifest"*). Dense embeddings capture semantic proximity in high-dimensional vector space.
- **Traceability Preservation**: Every generated vector retains its full source lineage (`chunk_id`, `source`, `document_type`, `strategy`, `chunk_index`, `character_count`, `chunk_text`).
- **Mathematical Consistency**: Validates that all vectors returned by the model have identical dimensionality, non-zero magnitudes, and numeric float values.

#### Complete Pipeline Flow

```
Shipping Documents (.txt, .pdf, .md)
      ↓
Document Loading & Sanitization
      ↓
Document Chunking (Paragraph-based)
      ↓
Metadata Tagging & Validation
      ↓
Validated Chunks (outputs/processed_chunks.json)
      ↓
Embeddings API (text-embedding-3-small)
      ↓
Text → Dense Numerical Vectors
      ↓
Vectors + Source Text + Metadata (outputs/embedded_chunks.json)
      ↓
NEXT: ChromaDB Semantic Search & Vector Storage
```

#### Separate Providers: Chat Generation vs. Embeddings API

ShipRule supports configuring dedicated, independent providers for text generation and vector embeddings:

```text
Chat / LLM Generation
        ↓
    Groq API
        ↓
   CHAT_MODEL
(e.g., openai/gpt-oss-120b)

Embeddings Generation
        ↓
 Embedding API Provider
        ↓
   EMBED_MODEL
(e.g., text-embedding-3-small)
        ↓
 Dense Embedding Vectors
```

> **Why Separate Providers?**
> An OpenAI-compatible API endpoint (such as Groq) offers specialized LLM inference for chat completions but does not host proprietary OpenAI embedding models (such as `text-embedding-3-small`). ShipRule cleanly separates these configurations to avoid routing embedding requests to chat-only endpoints.

#### Environment Configuration

Configure both providers in your local `.env` file (copied from `.env.example`):

```env
# Chat / LLM Generation Provider (e.g., Groq)
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
CHAT_MODEL=openai/gpt-oss-120b

# Embeddings API Provider (e.g., OpenAI)
EMBEDDING_API_KEY=your_embedding_provider_key_here
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBED_MODEL=text-embedding-3-small
```

> ⚠️ **Security Notice**: `.env` is listed in `.gitignore` and must never be committed to version control. Never hardcode API keys or provider URLs in source code.

#### Execution Command

Run embedding generation over validated chunks:
```bash
python embed.py
```
or
```bash
python -m src.embeddings
```

Cost control and custom options:
```bash
# Process only the first 5 chunks for cost control
python embed.py --max-chunks 5

# Override provider endpoint or model via CLI
python embed.py --model text-embedding-3-small --base-url https://api.openai.com/v1
```

#### Generated Embedding Artifacts

Saved in [`outputs/`](file:///outputs):
- [`outputs/embedded_chunks.json`](file:///outputs/embedded_chunks.json): Full embedded chunk collection containing complete vectors, original text, and complete metadata.
- [`outputs/embedding_report.json`](file:///outputs/embedding_report.json): Execution timestamp, model identifier, chunk counts, vector dimension, and validation audit.
- [`outputs/sample_embedding_output.json`](file:///outputs/sample_embedding_output.json): Lightweight sample containing metadata and trimmed vector previews (`[v0, v1, v2, ...]`) suitable for version control.


---

## End-to-End RAG Execution

To launch the full interactive ShipRule Customs Duty & Documentation Lookup session:

```bash
python app.py
```

### Session Features:
- **Scope Guard Filtering**: Automatically filters out-of-scope queries (e.g. non-shipping questions) with courteous redirection.
- **ChromaDB Semantic Search**: Vector distance filtering (L2 distance threshold $\le 1.35$) to ensure retrieved context relevance.
- **Adaptive Context Budgeting**: History management (`trim` / `sliding` / `summary`) respecting the configured context window budget.
- **Real-Time Cost Reporting**: Prints token usage and dollar billing estimate per query turn.
- **Commands**: Type `reset` to clear history, or `exit` / `quit` to end the session.

---

## Testing & Quality Assurance

ShipRule includes a unit test suite testing loaders, cleaners, chunking strategies, token estimators, context managers, scope guards, the ingestion pipeline, and mocked embedding API generation.

Run the test suite:
```bash
python -m unittest discover tests
```

**Test Suite Coverage (115 passing tests)**:
- `test_embeddings.py` (14 tests): Loading prepared chunks, mocked vector generation, numerical validation, vector dimension detection, consistent dimension validation, metadata preservation, error handling, report generation.
- `test_ingestion.py` (13 tests): Recursive discovery, reconciliation checks, metadata validation, failure isolation, resumability.
- `test_chunking.py` (13 tests): Fixed-size, paragraph, sentence chunking, overlap mechanics, boundary inspection, sample corpus execution.
- `test_document_loader.py` (10 tests): Multi-format intake (TXT, PDF), corrupted file handling, unsupported file skipping.
- `test_text_cleaner.py` (9 tests): Whitespace normalization and line break cleanup.
- `test_token_cost.py` (6 tests): Token calculation and model cost reporting.
- `test_context_manager.py` (13 tests): Context budgeting, history trimming, and turn retention.
- `test_prompt_templates.py` (7 tests): System & user template formatting.
- `test_scope_guard.py` (7 tests): In-scope and out-of-scope query guardrails.
- `test_structured_output.py` (11 tests): Structured JSON extraction and model response retries.
- `test_embedding_demo.py` (6 tests): Dimension verification and cosine similarity calculations.
- `test_chunk_metadata.py` (6 tests): Metadata schema and source traceability.

---

## Security Guidelines

1. **No API Keys in Repository**: `.env` is listed in `.gitignore`. Always inspect `.env.example` to ensure no sensitive values are present before committing.
2. **Untracked Local Data & Outputs**: Document files placed inside `data/` and generated outputs in `outputs/` are ignored by default (except approved sample outputs) to prevent accidental data leaks.
3. **Environment Isolation**: `.venv/` is ignored to ensure environment dependencies remain isolated per developer system.

---

## Reproducibility & Best Practices

- **Pinned Requirements**: `requirements.txt` contains strict version pins generated via `pip freeze` to ensure deterministic builds across all platforms.
- **Fresh Install Verification**: Teammates can clone the repo, run `python -m venv .venv`, run `pip install -r requirements.txt`, and immediately start development without dependency conflicts.

---

## Sprint-Level Analytical Tasks (Workflow Placeholders)

Three sprint-level analytical tasks have been created as issues in the repository and assigned to the user (`minnu04`):

1. **[Analyze and design optimal document chunking strategy for ChromaDB ingestion](https://github.com/kalviumcommunity/ShipRule/issues/2)**
   - **Label**: `data-pipeline`
   - **Assignee**: `minnu04`
   - **Description**: Analyze chunking sizes (e.g., 500, 1000 tokens) and overlap ratios (10%, 20%) to optimize similarity search precision and avoid LLM context window pollution.
2. **[Evaluate embedding model performance on domain-specific dataset](https://github.com/kalviumcommunity/ShipRule/issues/3)**
   - **Label**: `data-pipeline`
   - **Assignee**: `minnu04`
   - **Description**: Benchmark `text-embedding-3-small` against domain-specific queries and calculate retrieval metrics (Hit Rate @ K, MRR).
3. **[Design evaluation framework for LLM response quality assessment](https://github.com/kalviumcommunity/ShipRule/issues/4)**
   - **Label**: `feature`
   - **Assignee**: `minnu04`
   - **Description**: Formulate metrics (faithfulness, answer relevance, context recall) and design the structure for the evaluation framework inside the codebase.

A screenshot of the created issues is available at [docs/issues_list.png](file:///c:/Users/HP/OneDrive/Desktop/ShipRule/docs/issues_list.png).


