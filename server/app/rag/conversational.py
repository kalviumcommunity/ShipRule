"""
ShipRule CDLP - Conversational RAG & Follow-Up Context Module
=============================================================
Tracks multi-turn conversation history across dialogue turns, rewrites ambiguous
follow-up questions into standalone retrieval queries, retrieves relevant context
using the rewritten query, and generates grounded answers with token budget management.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
from groq import Groq
from app.rag.indexing import VectorCollection
from app.rag.retrieval import retrieve, load_indexed_vector_collection
from app.guardrails.retrieval import check_retrieval_quality, SAFE_REFUSAL_MESSAGE
from app.rag.pipeline import retrieve_context, assemble_context, generate_answer
from app.rag.context_manager import trim_history, total_tokens, count_tokens

logger = logging.getLogger(__name__)


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 150
) -> str:
    """
    Utility function to send a prompt to the LLM and return raw text output.
    Used for query rewriting, reference resolution, and history summarization.

    Args:
        prompt: Primary user prompt for LLM.
        system_prompt: Optional system instruction.
        client: Optional Groq / OpenAI client instance.
        model: Optional model override.
        temperature: Temperature sampling parameter.
        max_tokens: Maximum output tokens allowed.

    Returns:
        Stripped output text from LLM completion.
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    selected_model = model or os.getenv("CHAT_MODEL", "groq/compound-mini")

    # Initialize Groq client if not provided
    if client is None and api_key:
        try:
            client = Groq(api_key=api_key)
        except Exception as err:
            logger.warning(f"Failed to initialize Groq client: {err}")
            client = None

    if client is None:
        logger.warning("No LLM client available; returning prompt fallback.")
        if "Latest question:" in prompt:
            return prompt.split("Latest question:")[-1].strip()
        return prompt.strip()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # List of candidate models to try
    models_to_try = [selected_model]
    for fallback in ["groq/compound-mini", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    for m in models_to_try:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            raw_output = response.choices[0].message.content or ""
            return raw_output.strip()
        except Exception as e:
            logger.warning(f"Model {m} text completion failed: {e}. Trying fallback...")

    # Final fallback if all API attempts fail
    if "Latest question:" in prompt:
        return prompt.split("Latest question:")[-1].strip()
    return prompt.strip()


def format_history_for_rewriter(history: List[Dict[str, str]], max_turns: int = 3) -> str:
    """
    Formats recent history turns into a clean, token-efficient text string for query rewriting.
    """
    if not history or not isinstance(history, list):
        return "None"

    recent = history[-max_turns * 2:] if len(history) > max_turns * 2 else history
    lines = []
    for turn in recent:
        role = turn.get("role", "user").capitalize()
        content = str(turn.get("content", "")).strip()
        if len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def rewrite_followup(
    history: List[Dict[str, str]],
    question: str,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> str:
    """
    Rewrites a user's follow-up question into a standalone search query using conversation history.
    Uses the conversation history ONLY to resolve references (pronouns, ambiguous phrases).
    Does NOT answer the question.

    Args:
        history: List of conversation turns, e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        question: User's latest question string.
        client: Optional Groq/OpenAI client instance.
        model: Optional model name.

    Returns:
        Standalone retrieval query string.
    """
    if not question or not isinstance(question, str):
        raise ValueError("Question must be a non-empty string.")

    cleaned_q = question.strip()
    if not history or not isinstance(history, list) or len(history) == 0:
        return cleaned_q

    formatted_history = format_history_for_rewriter(history, max_turns=3)

    prompt = f"""Rewrite the user's latest question as a standalone search query. Use the conversation history only to resolve references. Do not answer the question.

History:
{formatted_history}

Latest question: {cleaned_q}"""

    system_instruction = (
        "You are a search query rewriting assistant. "
        "Your sole task is to rewrite ambiguous follow-up questions into clear, self-contained standalone search queries "
        "by resolving references (like 'it', 'that', 'the video', 'the deadline') using prior conversation history. "
        "Do NOT answer the question. Return ONLY the rewritten standalone search query text."
    )

    rewritten = call_llm(
        prompt=prompt,
        system_prompt=system_instruction,
        client=client,
        model=model,
        temperature=0.0,
        max_tokens=80
    ).strip()

    # Clean common response prefixes first, then quotes
    prefixes_to_strip = [
        "Standalone Query:", "Standalone query:", "Rewritten Query:",
        "Rewritten question:", "Query:", "Standalone search query:"
    ]
    for prefix in prefixes_to_strip:
        if rewritten.lower().startswith(prefix.lower()):
            rewritten = rewritten[len(prefix):].strip()

    if (rewritten.startswith('"') and rewritten.endswith('"')) or (rewritten.startswith("'") and rewritten.endswith("'")):
        rewritten = rewritten[1:-1].strip()

    return rewritten or cleaned_q


def retrieval_is_strong(
    chunks: List[Dict[str, Any]],
    max_distance_threshold: Optional[float] = None,
    min_relevant_chunks: int = 1
) -> bool:
    """
    Evaluates whether retrieved context chunks provide reliable, relevant context.

    Args:
        chunks: List of retrieved context chunk dictionaries.
        max_distance_threshold: Maximum allowed vector distance.
        min_relevant_chunks: Minimum number of required relevant chunks.

    Returns:
        True if retrieval is strong, False if weak or insufficient.
    """
    if not chunks or not isinstance(chunks, list) or len(chunks) < min_relevant_chunks:
        return False

    guardrail_res = check_retrieval_quality(
        retrieved_chunks=chunks,
        max_distance_threshold=max_distance_threshold,
        min_relevant_chunks=min_relevant_chunks
    )

    return guardrail_res.get("is_sufficient", False)


def conversational_answer(
    history: List[Dict[str, str]],
    user_question: str,
    candidate_k: int = 10,
    final_k: int = 4,
    metadata_filter: Optional[Dict[str, Any]] = None,
    use_reranking: bool = True,
    collection: Optional[Any] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    max_history_tokens: int = 1000
) -> Dict[str, Any]:
    """
    Executes Conversational RAG pipeline turn:
    1. Trims history to balance context and token limits.
    2. Rewrites follow-up user question into a standalone query.
    3. Retrieves context chunks using the rewritten query.
    4. Evaluates retrieval strength (retrieval_is_strong).
    5. Answers user question using retrieved context (or safe refusal if weak).
    6. Updates history with user and assistant turns.

    Args:
        history: Mutable list of dialogue turns.
        user_question: Latest user question.
        candidate_k: Candidate pool size for retrieval.
        final_k: Top-K chunks to assemble into context.
        metadata_filter: Optional metadata filter.
        use_reranking: Whether to re-rank chunks.
        collection: Optional VectorCollection instance.
        client: Optional Groq/OpenAI client instance.
        model: Optional model name.
        max_history_tokens: Maximum token budget reserved for conversation history.

    Returns:
        Dict containing:
          - "rewritten_query": Standalone query used for vector search.
          - "answer": Final grounded answer text or safe refusal.
          - "sources": List of metadata dictionaries for cited chunks.
          - "retrieved_chunks": Raw retrieved chunk objects.
          - "history": Updated conversation history list.
    """
    if user_question is None or not isinstance(user_question, str) or not user_question.strip():
        raise ValueError("User question must be a non-empty string.")

    cleaned_question = user_question.strip()

    # 1. Balance History & Token Limits
    trimmed_history, history_tokens, _ = trim_history(
        messages=history,
        budget=max_history_tokens
    )

    # 2. Rewrite follow-up question before retrieval
    standalone_query = rewrite_followup(
        history=trimmed_history,
        question=cleaned_question,
        client=client,
        model=model
    )

    # 3. Retrieve context using the rewritten query
    chunks, ret_timing = retrieve_context(
        query=standalone_query,
        candidate_k=candidate_k,
        final_k=final_k,
        metadata_filter=metadata_filter,
        use_reranking=use_reranking,
        collection=collection,
        client=client,
        model=model
    )

    # 4. Check retrieval strength
    if not retrieval_is_strong(chunks):
        answer = "I don't have enough reliable context to answer that."
        sources = []
    else:
        # 5. Generate grounded answer for original user_question using retrieved context
        assembled_context, sources_info = assemble_context(chunks)
        ans_result = generate_answer(
            query=cleaned_question,
            context=assembled_context,
            sources_info=sources_info,
            client=client,
            model=model
        )
        answer = ans_result.get("answer", "I don't have enough reliable context to answer that.")
        
        # Extract metadata from chunks for output structure
        sources = []
        for chunk in chunks:
            meta = chunk.get("metadata")
            if not meta:
                meta = {
                    "source": chunk.get("source", "unknown"),
                    "section": chunk.get("section", "unknown"),
                    "doc_type": chunk.get("doc_type", "unknown")
                }
            sources.append(meta)

    # 6. Update history in-place
    history.append({"role": "user", "content": cleaned_question})
    history.append({"role": "assistant", "content": answer})

    return {
        "rewritten_query": standalone_query,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": chunks,
        "history": history
    }


class ConversationalRAGManager:
    """
    Manages multi-turn Conversational RAG sessions, history tracking,
    query rewriting, and context window token limits.
    """

    def __init__(
        self,
        collection: Optional[VectorCollection] = None,
        max_history_tokens: int = 1000,
        model: Optional[str] = None
    ):
        self.collection = collection or load_indexed_vector_collection()
        self.max_history_tokens = max_history_tokens
        self.model = model
        self.history: List[Dict[str, str]] = []

    def reset_history(self) -> None:
        """Clears current conversation history."""
        self.history.clear()

    def ask(
        self,
        question: str,
        candidate_k: int = 10,
        final_k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_reranking: bool = True
    ) -> Dict[str, Any]:
        """
        Processes a single user question in the conversational context.
        """
        return conversational_answer(
            history=self.history,
            user_question=question,
            candidate_k=candidate_k,
            final_k=final_k,
            metadata_filter=metadata_filter,
            use_reranking=use_reranking,
            collection=self.collection,
            model=self.model,
            max_history_tokens=self.max_history_tokens
        )
