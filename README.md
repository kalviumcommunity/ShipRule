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

---

## Security Guidelines

1. **No API Keys in Repository**: `.env` is listed in `.gitignore`. Always inspect `.env.example` to ensure no sensitive values are present before committing.
2. **Untracked Local Data & Outputs**: Document files placed inside `data/` and generated outputs in `outputs/` are ignored by default (except approved sample outputs) to prevent accidental data leaks.
3. **Environment Isolation**: `.venv/` is ignored to ensure environment dependencies remain isolated per developer system.

---

## Reproducibility & Best Practices

- **Pinned Requirements**: `requirements.txt` contains strict version pins generated via `pip freeze` to ensure deterministic builds across all platforms.
- **Fresh Install Verification**: Teammates can clone the repo, run `python -m venv .venv`, run `pip install -r requirements.txt`, and immediately start development without dependency conflicts.

