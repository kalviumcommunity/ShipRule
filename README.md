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
- **Context & Token Budget Management**: History trimming, token tracking via `tiktoken`, and real-time per-query API cost calculation.
- **Vector Storage Foundation**: ChromaDB integration for dense semantic similarity retrieval.

---

## Folder Structure

```
ShipRule/
├── app.py                      # Main RAG interactive terminal session entry point
├── chunker.py                  # Document chunking CLI entry point
├── ingest.py                   # Corpus ingestion & validation pipeline CLI entry point
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
├── outputs/                    # Generated manifests, chunks, and analytical reports
│   ├── corpus_manifest.json
│   ├── ingestion_report.json
│   ├── ingestion_failures.json
│   ├── processed_chunks.json
│   ├── ingestion_log.txt
│   ├── chunks_fixed.json
│   ├── chunks_paragraph.json
│   ├── chunks_sentence.json
│   └── chunking_report.json
├── tests/                      # Comprehensive unit test suite (89 passing tests)
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
  - `corpus_manifest.json`, `ingestion_report.json`, `ingestion_failures.json`, `processed_chunks.json`, `ingestion_log.txt`.

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

ShipRule includes a unit test suite testing loaders, cleaners, chunking strategies, token estimators, context managers, scope guards, and the ingestion pipeline.

Run the test suite:
```bash
python -m unittest discover tests
```

**Test Suite Coverage (89 passing tests)**:
- `test_ingestion.py` (13 tests): Recursive discovery, reconciliation checks, metadata validation, failure isolation, resumability.
- `test_chunking.py` (13 tests): Fixed-size, paragraph, sentence chunking, overlap mechanics, boundary inspection, sample corpus execution.
- `test_document_loader.py` (10 tests): Multi-format intake (TXT, PDF), corrupted file handling, unsupported file skipping.
- `test_text_cleaner.py` (9 tests): Whitespace normalization and line break cleanup.
- `test_token_cost.py` (6 tests): Token calculation and model cost reporting.
- `test_context_manager.py` (13 tests): Context budgeting, history trimming, and turn retention.
- `test_prompt_templates.py` (7 tests): System & user template formatting.
- `test_scope_guard.py` (7 tests): In-scope and out-of-scope query guardrails.
- `test_structured_output.py` (11 tests): Structured JSON extraction and model response retries.

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


