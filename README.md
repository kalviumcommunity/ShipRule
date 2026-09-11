# ShipRule CDLP — Customs Duty & Documentation Lookup Platform

ShipRule is an AI-powered Customs Duty & Shipping Documentation Lookup Platform (CDLP) delivering grounded answers, source citation auditing, cross-encoder re-ranking, and anti-hallucination guardrails for maritime and international shipping compliance.

---

## 🏗️ Project Architecture

```text
ShipRule/
│
├── server/       # FastAPI + Production RAG Backend
│   ├── main.py   # Single FastAPI production server entry point
│   ├── app/
│   │   ├── api/          # REST API routes (health, query, documents)
│   │   ├── core/         # Configuration & logging settings
│   │   ├── rag/          # Hybrid retrieval, reranking, embeddings, pipeline
│   │   ├── documents/    # Loading, cleaning, chunking, ingestion
│   │   ├── llm/          # Groq completion engine
│   │   ├── guardrails/   # Scope and retrieval safeguards
│   │   ├── validation/   # Grounded answer & citation verification
│   │   └── prompts/      # System prompts & template definitions
│   ├── evaluation/       # RAG and retrieval quality evaluation suite
│   ├── scripts/          # Corpus ingestion & evaluation scripts
│   ├── tests/            # Automated test suite (303 tests passing)
│   ├── data/             # Document corpus (data/sample_corpus)
│   ├── uploads/          # User-uploaded document store
│   └── outputs/          # Processing reports & vector indexes
│
└── client/       # Next.js 16 + TypeScript + Tailwind CSS Frontend
    ├── src/
    │   ├── app/          # App Router pages (Dashboard, Ask ShipRule, Documents, Settings)
    │   ├── components/   # UI components, layout, query interface, document manager
    │   ├── services/     # API service client connecting to FastAPI
    │   ├── types/        # Strongly typed API interfaces
    │   └── config/       # Environment configuration
    └── package.json
```

---

## 🚀 Quick Start Guide

### 1. Start FastAPI Backend Server

```bash
cd server
python main.py
```

* Swagger API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
* Health Endpoint: [http://localhost:8000/health](http://localhost:8000/health)

### 2. Start Next.js Frontend Client

```bash
cd client
npm install
npm run dev
```

* Next.js Web Interface: [http://localhost:3000](http://localhost:3000)

---

## 🧪 Automated Testing

To run the complete backend test suite:

```bash
cd server
python -m pytest
```

**Status**: `303 / 303 passed`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status & version metadata |
| `GET` | `/health` | Service health & environment configuration check |
| `POST` | `/query` | Execute grounded RAG query with source citations |
| `GET` | `/documents` | List uploaded & indexed documents |
| `POST` | `/documents` | Upload, clean, chunk, embed, and index `.txt`, `.md`, or `.pdf` file |

---

## 🚀 Live Deployment

For complete instructions on deploying the **FastAPI backend on Render** and the **Next.js frontend on Vercel**, see the [Deployment Guide](deployment.md).

