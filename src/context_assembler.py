"""
ShipRule CDLP - Grounded Context Assembler & Prompt Augmentation Module
========================================================================
Assembles retrieved vector/ChromaDB chunks into a structured, token-budgeted,
and grounded context block for LLM prompt augmentation.

Features:
1. Formats retrieved chunks with unique source markers ([1], [2], [3]...).
2. Extracts and displays all available chunk metadata (source, section, chunk, page, etc.).
3. Enforces strict token budgets (max_context_tokens, response_reserve_tokens).
4. Employs greedy chunk inclusion in ranking order without arbitrary string truncation.
5. Injects strict grounding and anti-hallucination instructions.
6. Formats reproducible augmented prompt and budget reports.
"""

import os
import sys
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

try:
    from src.token_counter import count_tokens
except ImportError:
    from token_counter import count_tokens


# Configure logger
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. GROUNDING INSTRUCTIONS & TEMPLATES
# ==============================================================================

DEFAULT_GROUNDING_INSTRUCTIONS = (
    "You are a grounded customs-information assistant.\n"
    "Answer the user's question using ONLY the information contained in the provided context.\n"
    "Do not use outside knowledge or make unsupported assumptions.\n"
    "Do not invent customs rules, rates, documents, agencies, dates, URLs, or other facts.\n"
    "When the provided context does not contain enough information to answer, state clearly:\n"
    "'The provided context is insufficient to answer this question.'\n"
    "When making factual claims, cite the relevant source marker such as [1] or [2].\n"
    "Never invent or fabricate source markers."
)


# ==============================================================================
# 2. CHUNK FORMATTER
# ==============================================================================

def format_chunk(
    chunk: Dict[str, Any],
    marker_index: int
) -> str:
    """
    Formats a single retrieved chunk with its source marker and all available metadata.

    Args:
        chunk: Chunk dictionary (can contain top-level or nested 'metadata' fields).
        marker_index: 1-indexed sequential integer for the citation marker.

    Returns:
        Cleanly formatted string block for the chunk.
    """
    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}

    source = chunk.get("source") or meta.get("source") or "unknown_source"
    section = chunk.get("section") or meta.get("section")

    chunk_idx = chunk.get("chunk_index")
    if chunk_idx is None:
        chunk_idx = chunk.get("chunk")
    if chunk_idx is None and meta:
        chunk_idx = meta.get("chunk_index", meta.get("chunk"))

    page = chunk.get("page")
    if page is None and meta:
        page = meta.get("page")

    country = chunk.get("country")
    if country is None and meta:
        country = meta.get("country")

    hs_code = chunk.get("hs_code")
    if hs_code is None and meta:
        hs_code = meta.get("hs_code")

    source_agency = chunk.get("source_agency")
    if source_agency is None and meta:
        source_agency = meta.get("source_agency")

    source_url = chunk.get("source_url")
    if source_url is None and meta:
        source_url = meta.get("source_url")

    last_confirmed_date = chunk.get("last_confirmed_date")
    if last_confirmed_date is None and meta:
        last_confirmed_date = meta.get("last_confirmed_date")

    text = chunk.get("chunk_text") or chunk.get("text") or ""
    text = str(text).strip()

    def is_valid_meta(val: Any) -> bool:
        if val is None:
            return False
        s = str(val).strip()
        return bool(s) and s.lower() not in ("n/a", "none", "null")

    lines = [f"[{marker_index}] Source: {source}"]

    if is_valid_meta(section):
        lines.append(f"Section: {section}")
    if is_valid_meta(chunk_idx):
        lines.append(f"Chunk: {chunk_idx}")
    if is_valid_meta(page):
        lines.append(f"Page: {page}")
    if is_valid_meta(country):
        lines.append(f"Country: {country}")
    if is_valid_meta(hs_code):
        lines.append(f"HS Code: {hs_code}")
    if is_valid_meta(source_agency):
        lines.append(f"Agency: {source_agency}")
    if is_valid_meta(source_url):
        lines.append(f"URL: {source_url}")
    if is_valid_meta(last_confirmed_date):
        lines.append(f"Date: {last_confirmed_date}")

    header = "\n".join(lines)
    return f"{header}\n{text}" if text else header


# ==============================================================================
# 3. CONTEXT ASSEMBLY RESULT CLASS
# ==============================================================================

class AssembledContext(tuple):
    """
    Immutable tuple container supporting 4-tuple unpacking:
      `(formatted_context, selected_chunk_count, token_count, source_mapping)`
    while also supporting attribute and dictionary key access.
    """
    def __new__(
        cls,
        context: str,
        selected_chunk_count: int,
        token_count: int,
        source_mapping: List[Dict[str, Any]],
        budget_info: Optional[Dict[str, Any]] = None,
        budget_report: str = "",
        augmented_prompt: str = ""
    ):
        return super(AssembledContext, cls).__new__(
            cls,
            (context, selected_chunk_count, token_count, source_mapping)
        )

    def __init__(
        self,
        context: str,
        selected_chunk_count: int,
        token_count: int,
        source_mapping: List[Dict[str, Any]],
        budget_info: Optional[Dict[str, Any]] = None,
        budget_report: str = "",
        augmented_prompt: str = ""
    ):
        self.context = context
        self.formatted_context = context
        self.selected_chunk_count = selected_chunk_count
        self.included_chunks_count = selected_chunk_count
        self.token_count = token_count
        self.context_tokens = token_count
        self.source_mapping = source_mapping
        self.sources = source_mapping
        self.budget_info = budget_info or {}
        self.budget_report = budget_report
        self.augmented_prompt = augmented_prompt

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            mapping = {
                "context": self.context,
                "formatted_context": self.formatted_context,
                "selected_chunk_count": self.selected_chunk_count,
                "included_chunks_count": self.included_chunks_count,
                "token_count": self.token_count,
                "context_tokens": self.context_tokens,
                "source_mapping": self.source_mapping,
                "sources": self.sources,
                "budget_info": self.budget_info,
                "budget_report": self.budget_report,
                "augmented_prompt": self.augmented_prompt
            }
            if item in mapping:
                return mapping[item]
            raise KeyError(f"Key '{item}' not found in AssembledContext.")
        return super().__getitem__(item)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


# ==============================================================================
# 4. BUDGET REPORT FORMATTER
# ==============================================================================

def format_budget_report(
    max_context_tokens: int,
    response_reserve_tokens: int,
    instruction_tokens: int,
    question_tokens: int,
    available_context: int,
    retrieved_chunks_count: int,
    included_chunks_count: int,
    context_tokens: int,
    total_estimated: int,
    budget_check: str = "PASS"
) -> str:
    """Formats standardized context budget report block."""
    lines = [
        "[Context Budget]",
        f"Configured Budget : {max_context_tokens}",
        f"Response Reserve : {response_reserve_tokens}",
        f"Instruction Tokens : {instruction_tokens}",
        f"Question Tokens : {question_tokens}",
        f"Available Context : {available_context}",
        f"Retrieved Chunks : {retrieved_chunks_count}",
        f"Included Chunks : {included_chunks_count}",
        f"Context Tokens : {context_tokens}",
        f"Total Estimated : {total_estimated}",
        f"Budget Check : {budget_check}"
    ]
    return "\n".join(lines)


# ==============================================================================
# 5. AUGMENTED PROMPT BUILDER
# ==============================================================================

def build_augmented_prompt(
    user_question: str,
    context: str,
    system_instructions: Optional[str] = None,
    budget_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Assembles the final structured prompt containing instructions, context,
    user question, and token budget metadata.

    Args:
        user_question: The natural language user query.
        context: The formatted context block with source markers.
        system_instructions: Optional grounding instructions.
        budget_info: Optional dictionary of token budget metrics.

    Returns:
        Complete augmented prompt string ready for LLM input.
    """
    instructions = (system_instructions or DEFAULT_GROUNDING_INSTRUCTIONS).strip()
    ctx_text = (context or "").strip()
    q_text = (user_question or "").strip()

    prompt_sections = [
        "SYSTEM / INSTRUCTIONS",
        instructions,
        "",
        "CONTEXT",
        ctx_text if ctx_text else "[No relevant context available]",
        "",
        "USER QUESTION",
        q_text
    ]

    if budget_info:
        prompt_sections.extend([
            "",
            "CONTEXT BUDGET",
            f"Instruction Tokens: {budget_info.get('instruction_tokens', 0)}",
            f"Question Tokens: {budget_info.get('question_tokens', 0)}",
            f"Context Tokens: {budget_info.get('context_tokens', 0)}",
            f"Response Reserve: {budget_info.get('response_reserve_tokens', 500)}",
            f"Total Estimated Tokens: {budget_info.get('total_estimated_tokens', 0)}",
            f"Configured Budget: {budget_info.get('max_context_tokens', 4096)}",
            f"Budget Check: {budget_info.get('budget_check', 'PASS')}"
        ])

    return "\n".join(prompt_sections)


# ==============================================================================
# 6. CORE CONTEXT ASSEMBLY FUNCTION
# ==============================================================================

def assemble_context(
    retrieved_chunks: List[Dict[str, Any]],
    user_question: str = "",
    max_context_tokens: Optional[int] = None,
    response_reserve_tokens: Optional[int] = None,
    system_instructions: Optional[str] = None,
    encoding_name: str = "cl100k_base",
    print_report: bool = False,
    **kwargs: Any
) -> AssembledContext:
    """
    Assembles retrieved ChromaDB / vector search chunks into a token-budgeted,
    grounded context block with unique sequential source markers ([1], [2], ...).

    Enforces token budget:
      `instruction_tokens + question_tokens + context_tokens + response_reserve_tokens <= max_context_tokens`

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries from ChromaDB.
        user_question: The natural language user query.
        max_context_tokens: Maximum allowed total token budget (defaults to env MAX_CONTEXT_TOKENS or 4096).
        response_reserve_tokens: Reserved tokens for the LLM answer (defaults to env RESPONSE_RESERVE_TOKENS or 500).
        system_instructions: Optional custom grounding instructions.
        encoding_name: Tokenizer encoding name (cl100k_base).
        print_report: Whether to print ASCII budget report to stdout.

    Returns:
        AssembledContext tuple: `(formatted_context, selected_chunk_count, token_count, source_mapping)`
    """
    load_dotenv()

    # 1. Resolve Token Limits
    if max_context_tokens is None:
        try:
            max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "4096"))
        except (ValueError, TypeError):
            max_context_tokens = 4096

    if response_reserve_tokens is None:
        try:
            response_reserve_tokens = int(os.getenv("RESPONSE_RESERVE_TOKENS", "500"))
        except (ValueError, TypeError):
            response_reserve_tokens = 500

    instructions = system_instructions or DEFAULT_GROUNDING_INSTRUCTIONS
    q_str = str(user_question or "").strip()

    # 2. Count Static Tokens (Grounding Instructions + User Question)
    instruction_tokens = count_tokens(instructions, encoding_name=encoding_name)
    question_tokens = count_tokens(q_str, encoding_name=encoding_name)

    # 3. Calculate Available Context Budget
    available_context = max_context_tokens - response_reserve_tokens - instruction_tokens - question_tokens
    if available_context < 0:
        available_context = 0

    # 4. Greedy Chunk Inclusion in Ranking Order
    included_blocks: List[str] = []
    source_mapping: List[Dict[str, Any]] = []
    current_context_tokens = 0

    if retrieved_chunks and isinstance(retrieved_chunks, list):
        for candidate_chunk in retrieved_chunks:
            if not isinstance(candidate_chunk, dict):
                continue

            marker_idx = len(included_blocks) + 1
            formatted_chunk_str = format_chunk(candidate_chunk, marker_idx)
            chunk_token_cost = count_tokens(formatted_chunk_str, encoding_name=encoding_name)

            # Check if this chunk fits within remaining available context budget
            if current_context_tokens + chunk_token_cost <= available_context:
                included_blocks.append(formatted_chunk_str)
                current_context_tokens += chunk_token_cost

                # Record source metadata
                meta = candidate_chunk.get("metadata") if isinstance(candidate_chunk.get("metadata"), dict) else {}
                src = candidate_chunk.get("source") or meta.get("source") or "unknown_source"
                sec = candidate_chunk.get("section") or meta.get("section")
                c_idx = candidate_chunk.get("chunk_index")
                if c_idx is None:
                    c_idx = candidate_chunk.get("chunk")
                if c_idx is None and meta:
                    c_idx = meta.get("chunk_index", meta.get("chunk"))
                pg = candidate_chunk.get("page")
                if pg is None and meta:
                    pg = meta.get("page")

                source_mapping.append({
                    "marker": f"[{marker_idx}]",
                    "citation_id": f"[{marker_idx}]",
                    "source": str(src),
                    "section": str(sec) if sec is not None else None,
                    "chunk_index": c_idx,
                    "page": str(pg) if pg is not None else None,
                    "rank": candidate_chunk.get("rank", marker_idx),
                    "score": candidate_chunk.get("rerank_score", candidate_chunk.get("similarity_score", 0.0)),
                    "token_count": chunk_token_cost
                })
            else:
                # Chunk exceeds remaining budget: skip it and attempt smaller subsequent chunks
                logger.debug(
                    "Skipping chunk %s (size %d tokens) exceeding remaining context budget (%d tokens).",
                    candidate_chunk.get("source", "unknown"),
                    chunk_token_cost,
                    available_context - current_context_tokens
                )

    # 5. Formulate Final Context Block
    formatted_context = "\n\n".join(included_blocks)
    context_tokens = count_tokens(formatted_context, encoding_name=encoding_name) if formatted_context else 0

    # 6. Budget Check & Verification
    total_estimated_tokens = instruction_tokens + question_tokens + context_tokens + response_reserve_tokens
    budget_check = "PASS" if total_estimated_tokens <= max_context_tokens else "FAIL"

    budget_info = {
        "max_context_tokens": max_context_tokens,
        "response_reserve_tokens": response_reserve_tokens,
        "instruction_tokens": instruction_tokens,
        "question_tokens": question_tokens,
        "available_context": available_context,
        "retrieved_chunks_count": len(retrieved_chunks) if retrieved_chunks else 0,
        "included_chunks_count": len(included_blocks),
        "context_tokens": context_tokens,
        "total_estimated_tokens": total_estimated_tokens,
        "budget_check": budget_check
    }

    budget_report = format_budget_report(
        max_context_tokens=max_context_tokens,
        response_reserve_tokens=response_reserve_tokens,
        instruction_tokens=instruction_tokens,
        question_tokens=question_tokens,
        available_context=available_context,
        retrieved_chunks_count=budget_info["retrieved_chunks_count"],
        included_chunks_count=budget_info["included_chunks_count"],
        context_tokens=context_tokens,
        total_estimated=total_estimated_tokens,
        budget_check=budget_check
    )

    augmented_prompt = build_augmented_prompt(
        user_question=q_str,
        context=formatted_context,
        system_instructions=instructions,
        budget_info=budget_info
    )

    if print_report:
        print(budget_report)

    return AssembledContext(
        context=formatted_context,
        selected_chunk_count=len(included_blocks),
        token_count=context_tokens,
        source_mapping=source_mapping,
        budget_info=budget_info,
        budget_report=budget_report,
        augmented_prompt=augmented_prompt
    )
