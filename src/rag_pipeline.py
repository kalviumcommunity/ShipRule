"""
ShipRule CDLP - End-to-End RAG Pipeline Module
================================================
Implements a modular, testable End-to-End RAG Pipeline architecture:

  User Query
      ↓
  embed_query()
      ↓
  retrieve_context() (Metadata Filtering, Vector/Hybrid Search, Re-Ranking)
      ↓
  assemble_context() (Deterministic Citation Formatting)
      ↓
  generate_answer() (Grounded System Prompt & Anti-Hallucination)
      ↓
  answer_query() (Pipeline Orchestrator, Timing Metrics, Debug Tracing)

Each stage has a single clear responsibility and is independently unit testable.
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# UTF-8 stdout configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from src.indexing import VectorCollection
from src.embeddings import generate_query_embedding
from src.retrieval import retrieve, hybrid_retrieve, load_indexed_vector_collection
from src.reranker import rerank, retrieve_and_rerank
from src.prompt_templates import ANSWER_TEMPLATE, PromptTemplate, DEFAULT_GROUNDING_INSTRUCTIONS, GROUNDED_AUGMENTED_PROMPT_TEMPLATE
from src.context_assembler import assemble_context as assemble_grounded_context, format_budget_report, build_augmented_prompt

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")


# ==============================================================================
# 1. QUERY EMBEDDING STAGE
# ==============================================================================

def embed_query(
    query: str,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> List[float]:
    """
    Validates user query and generates vector embedding using the exact same model used during document indexing.

    Args:
        query: Non-empty query string.
        client: Optional embedding client instance.
        model: Optional embedding model name.

    Returns:
        List of float embedding values.
    """
    if query is None or not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    load_dotenv()
    selected_model = model or os.getenv("EMBED_MODEL", "text-embedding-3-small")

    try:
        query_vector = generate_query_embedding(query.strip(), client=client, model=selected_model)
    except Exception as e:
        raise RuntimeError(f"Query embedding generation failed: {e}")

    if not query_vector:
        raise RuntimeError("Failed to generate vector embedding for query.")

    return query_vector


# ==============================================================================
# 2. RETRIEVAL & RE-RANKING STAGE
# ==============================================================================

def retrieve_context(
    query: str,
    candidate_k: int = 10,
    final_k: int = 3,
    metadata_filter: Optional[Dict[str, Any]] = None,
    use_reranking: bool = True,
    use_hybrid: bool = False,
    keywords: Optional[List[str]] = None,
    collection: Optional[VectorCollection] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Retrieves and re-ranks candidate document chunks for a query using existing retrieval modules.

    Args:
        query: Non-empty search query.
        candidate_k: Number of candidates to retrieve in Stage 1.
        final_k: Final Top-K context chunks to return.
        metadata_filter: Optional metadata filter dict.
        use_reranking: Bool flag to enable Stage 2 re-ranking (default: True).
        use_hybrid: Bool flag to enable hybrid vector + keyword search (default: False).
        keywords: Optional list of keyword strings for hybrid search.
        collection: Optional VectorCollection instance.
        client: Optional embedding client.
        model: Optional embedding model name.

    Returns:
        Tuple of (final_top_k_chunks, timing_dict).
    """
    if query is None or not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k <= 0:
        raise ValueError("candidate_k must be a positive integer greater than 0.")

    if isinstance(final_k, bool) or not isinstance(final_k, int) or final_k <= 0:
        raise ValueError("final_k must be a positive integer greater than 0.")

    if collection is None:
        collection = load_indexed_vector_collection()

    total_chunks = collection.count()
    if total_chunks == 0:
        return [], {"vector_retrieval_ms": 0.0, "reranking_ms": 0.0}

    effective_final_k = min(final_k, total_chunks)
    effective_candidate_k = min(candidate_k, total_chunks)

    t0 = time.perf_counter()

    # Stage 1: Candidate Retrieval
    if use_hybrid:
        kw_list = keywords or [t for t in query.split() if len(t) > 2]
        candidates = hybrid_retrieve(
            query=query,
            keywords=kw_list,
            k=effective_candidate_k,
            metadata_filter=metadata_filter,
            collection=collection,
            client=client,
            model=model
        )
    else:
        candidates = retrieve(
            query=query,
            k=effective_candidate_k,
            metadata_filter=metadata_filter,
            collection=collection,
            client=client,
            model=model
        )

    t1 = time.perf_counter()
    retrieval_ms = round((t1 - t0) * 1000, 2)

    # Stage 2: Re-Ranking
    t2 = time.perf_counter()
    if use_reranking and len(candidates) > 0:
        final_chunks = rerank(query=query, candidates=candidates, final_k=effective_final_k)
    else:
        final_chunks = candidates[:effective_final_k]
        for idx, item in enumerate(final_chunks, start=1):
            item["rank"] = idx

    t3 = time.perf_counter()
    reranking_ms = round((t3 - t2) * 1000, 2)

    timing = {
        "retrieval_ms": retrieval_ms,
        "reranking_ms": reranking_ms
    }

    return final_chunks, timing


# ==============================================================================
# 3. CONTEXT ASSEMBLY STAGE
# ==============================================================================

def assemble_context(chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Converts retrieved candidate chunks into clean, formatted context for the LLM prompt.
    Attaches stable citation identifiers e.g. [1] Source: document.txt | Chunk Index: 1.

    Args:
        chunks: List of retrieved chunk dictionaries.

    Returns:
        Tuple of (assembled_context_string, structured_sources_list).
    """
    if not chunks or not isinstance(chunks, list):
        return "", []

    context_blocks = []
    sources_list = []

    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.get("source") or chunk.get("metadata", {}).get("source", "unknown_source")
        chunk_idx = chunk.get("chunk_index") or chunk.get("metadata", {}).get("chunk_index", idx)
        text = chunk.get("chunk_text") or chunk.get("text", "").strip()

        citation_id = f"[{idx}]"
        header = f"{citation_id} Source: {source} | Chunk Index: {chunk_idx}"
        block = f"{header}\n{text}"
        context_blocks.append(block)

        sources_list.append({
            "citation_id": citation_id,
            "source": str(source),
            "chunk_index": chunk_idx,
            "rank": chunk.get("rank", idx),
            "score": chunk.get("rerank_score", chunk.get("similarity_score", 0.0))
        })

    assembled_text = "\n\n".join(context_blocks)
    return assembled_text, sources_list


# ==============================================================================
# 4. GROUNDED LLM GENERATION STAGE
# ==============================================================================

def generate_answer(
    query: str,
    context: str,
    sources_info: Optional[List[Dict[str, Any]]] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a grounded answer to the user query using ONLY the supplied context.
    Enforces strict anti-hallucination rules.

    Args:
        query: User search query.
        context: Formatted context string from assemble_context().
        sources_info: Optional list of source dicts.
        client: Optional LLM client (e.g. Groq instance).
        model: Optional LLM model name string.

    Returns:
        Dict with 'answer' and 'sources'.
    """
    if not query or not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if not context or not isinstance(context, str) or not context.strip():
        return {
            "answer": "I could not find enough relevant information in the available knowledge base to answer this question. The provided context is insufficient to answer this question.",
            "sources": []
        }

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    # If Groq API key is present, attempt live LLM completion
    if api_key and client is not None or api_key and not api_key.startswith("gsk_local_key"):
        try:
            from groq import Groq
            groq_client = client or Groq(api_key=api_key)
            llm_model = model or os.getenv("CHAT_MODEL", "groq/compound-mini")

            system_instruction = (
                "You are a grounded customs-information assistant.\n"
                "Answer the user's question using ONLY the information contained in the provided context.\n"
                "Do not use outside knowledge or make unsupported assumptions.\n"
                "Do not invent customs rules, rates, documents, agencies, dates, URLs, or other facts.\n"
                "When the provided context does not contain enough information to answer, state clearly:\n"
                "'The provided context is insufficient to answer this question.'\n"
                "When making factual claims, cite the relevant source marker such as [1] or [2].\n"
                "Never invent or fabricate source markers."
            )

            prompt_user = f"Context:\n{context}\n\nQuestion:\n{query}"

            resp = groq_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt_user}
                ],
                temperature=0.1,
                max_tokens=500
            )

            reply_text = resp.choices[0].message.content.strip()
            sources_out = [s["source"] for s in (sources_info or [])]
            return {
                "answer": reply_text,
                "sources": list(dict.fromkeys(sources_out))
            }

        except Exception as e:
            logging.warning("Live LLM generation API call encountered: %s. Falling back to grounded context synthesis.", e)

    # Fallback / Offline Grounded Synthesis Engine (Guarantees zero API failure breakages)
    sources_out = [s["source"] for s in (sources_info or [])]
    unique_sources = list(dict.fromkeys(sources_out))

    # Clean context preview for grounded response
    cleaned_paragraphs = []
    for block in context.split("\n\n"):
        lines = block.split("\n")
        if len(lines) > 1:
            body = " ".join(lines[1:])
            cleaned_paragraphs.append(body)
        else:
            cleaned_paragraphs.append(block)

    synthesized_answer = " ".join(cleaned_paragraphs)
    if len(synthesized_answer) > 400:
        synthesized_answer = synthesized_answer[:397] + "..."

    formatted_answer = f"Based on retrieved documentation:\n{synthesized_answer}"

    return {
        "answer": formatted_answer,
        "sources": unique_sources
    }


# ==============================================================================
# 5. PIPELINE ORCHESTRATOR
# ==============================================================================

def answer_query(
    query: str,
    candidate_k: int = 10,
    final_k: int = 3,
    metadata_filter: Optional[Dict[str, Any]] = None,
    use_reranking: bool = True,
    use_hybrid: bool = False,
    debug: bool = False,
    collection: Optional[VectorCollection] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Connects all RAG pipeline stages end-to-end:
      User Query -> embed_query() -> retrieve_context() -> assemble_context() -> generate_answer() -> Answer + Sources.

    Handles empty retrieval safely without calling the LLM or hallucinating facts.
    Measures timing metrics and supports optional debug tracing mode.

    Args:
        query: User question string.
        candidate_k: Number of candidate chunks to fetch in Stage 1 retrieval.
        final_k: Top-K context chunks to pass to LLM context assembly.
        metadata_filter: Optional metadata filtering criteria.
        use_reranking: Bool flag to enable re-ranking.
        use_hybrid: Bool flag to enable hybrid vector + keyword search.
        debug: Bool flag to include execution trace in returned response.
        collection: Optional VectorCollection instance.
        client: Optional embedding/LLM client.
        model: Optional model name string.

    Returns:
        Structured RAG answer dict containing 'answer', 'sources', 'retrieved_chunks', 'timing', and optional 'debug_info'.
    """
    t_start = time.perf_counter()

    # 1. Validate Query
    if query is None or not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # 2. Stage 1 & 2: Embedding & Retrieval
    t_embed_start = time.perf_counter()
    query_vector = embed_query(query, client=client, model=model)
    t_embed_end = time.perf_counter()
    embed_ms = round((t_embed_end - t_embed_start) * 1000, 2)

    chunks, ret_timing = retrieve_context(
        query=query,
        candidate_k=candidate_k,
        final_k=final_k,
        metadata_filter=metadata_filter,
        use_reranking=use_reranking,
        use_hybrid=use_hybrid,
        collection=collection,
        client=client,
        model=model
    )

    # 3. Handle Empty / Insufficient Retrieval Safely
    if not chunks:
        t_total_end = time.perf_counter()
        total_ms = round((t_total_end - t_start) * 1000, 2)

        safe_response = {
            "answer": "I could not find enough relevant information in the available knowledge base to answer this question.",
            "sources": [],
            "retrieved_chunks": [],
            "timing": {
                "embedding_ms": embed_ms,
                "retrieval_ms": ret_timing["retrieval_ms"],
                "reranking_ms": ret_timing["reranking_ms"],
                "context_assembly_ms": 0.0,
                "generation_ms": 0.0,
                "total_pipeline_time_ms": total_ms
            }
        }
        if debug:
            safe_response["debug_info"] = {
                "query": query,
                "candidate_k": candidate_k,
                "final_k": final_k,
                "metadata_filter": metadata_filter,
                "status": "empty_retrieval_fallback"
            }
        return safe_response

    # 4. Stage 3: Context Assembly
    t_asm_start = time.perf_counter()
    assembled_context, sources_info = assemble_context(chunks)
    t_asm_end = time.perf_counter()
    asm_ms = round((t_asm_end - t_asm_start) * 1000, 2)

    # 5. Stage 4: Grounded LLM Generation
    t_gen_start = time.perf_counter()
    gen_result = generate_answer(
        query=query,
        context=assembled_context,
        sources_info=sources_info,
        client=client,
        model=model
    )
    t_gen_end = time.perf_counter()
    gen_ms = round((t_gen_end - t_gen_start) * 1000, 2)

    t_total_end = time.perf_counter()
    total_ms = round((t_total_end - t_start) * 1000, 2)

    pipeline_result = {
        "answer": gen_result["answer"],
        "sources": gen_result["sources"],
        "retrieved_chunks": chunks,
        "timing": {
            "embedding_ms": embed_ms,
            "retrieval_ms": ret_timing["retrieval_ms"],
            "reranking_ms": ret_timing["reranking_ms"],
            "context_assembly_ms": asm_ms,
            "generation_ms": gen_ms,
            "total_pipeline_time_ms": total_ms
        }
    }

    if debug:
        grounded_assembly = assemble_grounded_context(
            retrieved_chunks=chunks,
            user_question=query
        )
        pipeline_result["debug_info"] = {
            "query": query,
            "candidate_k": candidate_k,
            "final_k": final_k,
            "metadata_filter": metadata_filter,
            "use_reranking": use_reranking,
            "use_hybrid": use_hybrid,
            "query_vector_dim": len(query_vector),
            "retrieved_chunks_count": len(chunks),
            "sources_citation_info": sources_info,
            "assembled_context_prompt": assembled_context,
            "budget_info": grounded_assembly.budget_info,
            "budget_report": grounded_assembly.budget_report,
            "augmented_prompt": grounded_assembly.augmented_prompt
        }

    return pipeline_result

