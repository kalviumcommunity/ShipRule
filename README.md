# RAG Application Starter (`rag-app-starter`)

A production-ready foundation for building Retrieval-Augmented Generation (RAG) applications using OpenAI models and ChromaDB vector store.

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
- [Running the Project](#running-the-project)
  - [LLM Chat Completion & Error Handling Script](#llm-chat-completion--error-handling-script)
  - [Multi-Format Document Loader & Intake Engine](#multi-format-document-loader--intake-engine)
  - [Document Chunking Strategies](#document-chunking-strategies)
- [Security Guidelines](#security-guidelines)
- [Reproducibility & Best Practices](#reproducibility--best-practices)

---

## Project Overview

This repository provides an isolated, modular environment for developing RAG pipelines. It integrates:
- **OpenAI**: Large Language Models (LLMs) and embedding models for generation and retrieval.
- **ChromaDB**: In-memory vector storage for document embeddings and similarity search.
- **python-dotenv**: Secure environment management without exposing secret API keys.

---

## Folder Structure

```
rag-app-starter/
├── data/           # Documents and local knowledge base files (git-ignored, .gitkeep preserved)
├── src/            # Core Python source code (main pipeline, components, helpers)
│   ├── __init__.py
│   └── main.py
├── prompts/        # Prompt templates and system instructions (.gitkeep preserved)
├── outputs/        # Generated response outputs and logs (git-ignored, .gitkeep preserved)
├── .env.example    # Environment variable template (no real secrets)
├── .gitignore      # Git ignore rules for venv, secrets, and data
├── requirements.txt # Pinned dependency versions for reproducible builds
└── README.md       # Project documentation
```

---

## Prerequisites

- **Python**: Version 3.10 or higher
- **Git**: Installed and configured
- **OpenAI API Key**: (Optional for local dry-run, required for live LLM API calls)

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

## Running the Project

To execute the application foundation and verify that OpenAI, ChromaDB, and environment configurations load cleanly:

```bash
python src/main.py
```

### LLM Chat Completion & Error Handling Script

You can also run the dedicated LLM completion script directly:

```bash
python src/llm_completion.py
```

This script:
- Configures an OpenAI-compatible client from `.env` (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `CHAT_MODEL`).
- Sends a chat completion request with system & user roles.
- Logs outgoing request payloads, response text, and token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`).
- Catches and explains common errors:
  - **401 Unauthorized**: Explains missing/invalid `OPENAI_API_KEY`.
  - **429 Too Many Requests**: Explains rate limits/quota exhaustion.

Sample run logs are captured and saved in [`outputs/sample_output.txt`](file:///c:/Users/dhars/OneDrive/Desktop/ShipRule/outputs/sample_output.txt).

### Multi-Format Document Loader & Intake Engine

ShipRule includes a document loader supporting multiple document formats:
- **Supported Formats**: `.txt` (Plain text), `.pdf` (PDF documents via `pypdf`), and `.md` (Markdown).
- **Sample Corpus**: Located in `data/sample_corpus/` (`shipping_rules.txt`, `customs_requirements.txt`, `international_shipping_guide.pdf`).
- **Execution Command**:
  ```bash
  python document_loader.py
  ```
  or
  ```bash
  python -m src.document_loader
  ```
- **Intake Demonstration**: Extracts plain text, preserves source identity (`filename`), reports character lengths, displays formatted text snippet samples, and gracefully handles missing, corrupt, or unsupported files with warnings without interrupting batch processing.

### Document Chunking Strategies

Chunking partitions raw extracted document text into structured semantic units suitable for dense vector embeddings and semantic retrieval in ChromaDB.

#### Why Chunking is Required
- **Embedding Model Limits**: Embedding models operate within strict token input limits (e.g. 512–8192 tokens).
- **Retrieval Precision vs. Context Balance**: Granular chunks ensure that vector queries retrieve specific, highly relevant facts without diluting cosine similarity across multiple unrelated topics.
- **LLM Context Budget & API Cost**: Injecting tightly scoped chunks into the prompt context window optimizes LLM comprehension and minimizes per-query billing costs.

#### Implemented Strategies

1. **Fixed-Size Chunking (`fixed_size`)**:
   - Divides text into fixed-length character slices (default: `size=500`, `overlap=50`).
   - Slices sequentially using a sliding window step (`size - overlap`).
   - Overlap carries context across boundary cuts, but may slice sentences or words in half.

2. **Paragraph-Based Chunking (`paragraph`)**:
   - Splits extracted text along natural paragraph breaks (`\n\n+`).
   - Removes empty lines and preserves complete thematic clauses.
   - Ideal for structured policies, compliance guides, and legal rules.

3. **Sentence-Based Chunking (`sentence`)**:
   - Parses grammatical sentence boundaries using abbreviation-aware NLP rules.
   - Avoids cutting clauses mid-sentence, preserving exact factual assertions.

#### Execution Command

Run the chunking pipeline over the sample corpus:
```bash
python chunker.py
```
or
```bash
python -m src.chunker
```

#### Output Files

Generated chunks and analytical reports are saved inside `outputs/`:
- [`outputs/chunks_fixed.json`](file:///outputs/chunks_fixed.json): Standardized chunk objects generated via fixed-size strategy.
- [`outputs/chunks_paragraph.json`](file:///outputs/chunks_paragraph.json): Standardized chunk objects generated via paragraph strategy.
- [`outputs/chunks_sentence.json`](file:///outputs/chunks_sentence.json): Standardized chunk objects generated via sentence strategy.
- [`outputs/chunking_report.json`](file:///outputs/chunking_report.json): Consolidated per-document and corpus-wide statistical report with boundary inspections and recommendation.

#### Corpus Comparison Results

| Strategy | Total Chunks | Avg Chunk Size | Context Preservation | Main Advantage | Main Limitation |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Fixed-Size** | 8 | 411.8 chars | Low - Moderate | Predictable & uniform chunk bounds | May split sentences/ideas mid-clause |
| **Paragraph-Based** | 8 | 379.2 chars | High | Preserves complete semantic sections | Uneven chunk sizes across sections |
| **Sentence-Based** | 21 | 143.9 chars | Moderate - High | Preserves grammatical sentence syntax | Produces many small, fragmented chunks |

#### Recommended Strategy for ShipRule

> **Recommendation**: **Paragraph-Based Chunking**
> 
> *Rationale*: Paragraph-based chunking is recommended for the current ShipRule corpus because shipping and customs documents contain structured policy sections where preserving complete paragraphs helps maintain the context required for retrieval.

#### Connection to the Next Learning Unit (Embeddings & ChromaDB)

The output of the chunking module directly feeds into the upcoming Embeddings API stage:
```
Document Loader
      ↓
Extracted Documents (Plain Text + Metadata)
      ↓
Chunking Strategy (Unified Chunks: chunk_id, source, document_type, chunk_text)
      ↓
Validated Chunks + Metadata
      ↓
Embeddings API (text-embedding-3-small)  ← NEXT LU
      ↓
ChromaDB Vector Store
      ↓
Semantic Retrieval & Top-K RAG Generation
```

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

