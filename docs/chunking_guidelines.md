# Document Chunking Guidelines

This document provides guidelines on selecting chunk size and overlap parameters for various types of document sources within the RAG pipeline.

## Parameters Analysis

### 1. Chunk Size
- **Small Chunks (100–300 characters)**: Better for factual retrieval where answers are short. However, can lose contextual continuity.
- **Medium Chunks (500–1000 characters)**: Recommended baseline for general documents. Offers a balance of specific facts and surrounding context.
- **Large Chunks (1000+ characters)**: Useful when synthesizing complex concepts, but increases token usage and risks context dilution.

### 2. Chunk Overlap
- An overlap of **10%–20%** of the chunk size (e.g., 50–100 characters overlap for a 500-character chunk) ensures that semantic meaning is preserved across chunk transitions.

## Recommendations by Document Type

| Document Type | Recommended Size | Recommended Overlap | Strategy |
| :--- | :--- | :--- | :--- |
| Text Manuals | 500 chars | 50 chars | Character split |
| Source Code | 1000 chars | 100 chars | AST/logical split |
| FAQ Q&As | Keep whole | 0 chars | No split |
