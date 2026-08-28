"""
ShipRule CDLP - Context Window & Message History Management Module
==================================================================
Provides context window budgeting, token counting, history trimming,
summarization, and RAG context management for multi-turn LLM sessions.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple

# Ensure project root is in sys.path for direct imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.token_counter import count_tokens
except ImportError:
    from token_counter import count_tokens

try:
    from src.prompt_templates import render, ANSWER_TEMPLATE
except ImportError:
    from prompt_templates import render, ANSWER_TEMPLATE


# Configure module logger
logger = logging.getLogger(__name__)


def total_tokens(messages: List[Dict[str, str]], encoding_name: str = "cl100k_base") -> int:
    """
    Calculates total tokens for a list of chat messages including ChatML / OpenAI formatting overhead.
    Standard ChatML overhead: ~4 tokens per message (role, delimiters) + 2 tokens for assistant priming.
    """
    if not messages:
        return 0

    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Overhead per message (<|im_start|>{role}\n{content}<|im_end|>)
        for key, value in message.items():
            if value:
                num_tokens += count_tokens(str(value), encoding_name=encoding_name)
    
    num_tokens += 2  # Primer overhead for assistant reply
    return num_tokens


def trim_history(
    messages: List[Dict[str, str]],
    budget: int,
    preserve_recent: int = 2,
    encoding_name: str = "cl100k_base"
) -> Tuple[List[Dict[str, str]], int, str]:
    """
    Trims conversation history to fit within the specified token budget.
    - Always preserves the system prompt (if present at index 0).
    - Prefers removing complete user/assistant turn pairs.
    - Preserves up to `preserve_recent` recent turns if budget allows, but trims further if needed.
    - If a single message exceeds budget, gracefully truncates its content.

    Returns:
        (trimmed_messages, tokens_after_trim, strategy_log)
    """
    current_tokens = total_tokens(messages, encoding_name=encoding_name)
    if current_tokens <= budget:
        return list(messages), current_tokens, "none"

    if not messages:
        return [], 0, "none"

    # Separate system prompt from conversational turns
    system_message = None
    turns_start_idx = 0
    if messages[0].get("role") == "system":
        system_message = messages[0]
        turns_start_idx = 1

    conversation = list(messages[turns_start_idx:])
    if not conversation:
        # Only system message exists and it's over budget
        truncated_content = _truncate_text_to_tokens(
            system_message["content"], budget - 10, encoding_name
        )
        trimmed = [{"role": "system", "content": truncated_content}]
        return trimmed, total_tokens(trimmed, encoding_name), "system_truncated"

    # Identify user/assistant turn pairs or individual messages
    # Attempt trimming from oldest turns first
    modified = False
    
    while total_tokens(([system_message] if system_message else []) + conversation, encoding_name) > budget:
        if len(conversation) <= 1:
            # Only one conversational message left, but still over budget.
            # Truncate content of the single remaining message gracefully.
            last_msg = conversation[0]
            sys_tok = total_tokens([system_message], encoding_name) if system_message else 0
            available_tok = budget - sys_tok - 10
            if available_tok > 20:
                truncated_content = _truncate_text_to_tokens(
                    last_msg["content"], available_tok, encoding_name
                )
                conversation[0] = {"role": last_msg["role"], "content": truncated_content + "\n[Truncated to fit context window budget]"}
            break

        # Check if the first two messages form a pair (user -> assistant)
        if len(conversation) >= 2 and conversation[0].get("role") == "user" and conversation[1].get("role") == "assistant":
            # Remove full pair
            conversation.pop(0)
            conversation.pop(0)
            modified = True
        else:
            # Remove single oldest message
            conversation.pop(0)
            modified = True

    result_messages = ([system_message] if system_message else []) + conversation
    final_tokens = total_tokens(result_messages, encoding_name)
    
    # Final safety check: if still over budget (e.g. single message content was huge)
    if final_tokens > budget and result_messages:
        for idx in range(len(result_messages)):
            if result_messages[idx].get("role") != "system":
                avail = max(50, budget - total_tokens([m for i, m in enumerate(result_messages) if i != idx], encoding_name))
                result_messages[idx] = {
                    "role": result_messages[idx]["role"],
                    "content": _truncate_text_to_tokens(result_messages[idx]["content"], avail - 15, encoding_name) + "\n[Truncated to fit context window budget]"
                }
        final_tokens = total_tokens(result_messages, encoding_name)

    strategy_note = "trim" if modified else "truncate"
    return result_messages, final_tokens, strategy_note


def summarize_history(
    messages: List[Dict[str, str]],
    budget: int,
    summarize_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    preserve_recent: int = 2,
    encoding_name: str = "cl100k_base"
) -> Tuple[List[Dict[str, str]], int, str]:
    """
    Summarizes older conversation turns into a compact summary message when history exceeds budget.
    - Always preserves the system prompt (index 0).
    - Preserves `preserve_recent` recent user/assistant turns intact.
    - Condenses older turns into a summary block.
    - If total tokens still exceed budget after summarization, falls back to `trim_history`.

    Returns:
        (summarized_messages, final_tokens, strategy_log)
    """
    current_tokens = total_tokens(messages, encoding_name=encoding_name)
    if current_tokens <= budget:
        return list(messages), current_tokens, "none"

    if not messages:
        return [], 0, "none"

    system_message = None
    turns = list(messages)
    if turns and turns[0].get("role") == "system":
        system_message = turns.pop(0)

    # Determine split point: preserve recent turns (each turn = user + assistant pair, so preserve_recent * 2 messages)
    preserve_count = preserve_recent * 2
    if len(turns) <= preserve_count:
        # Not enough older history to summarize, trim directly
        return trim_history(messages, budget, preserve_recent=preserve_recent, encoding_name=encoding_name)

    older_turns = turns[:-preserve_count]
    recent_turns = turns[-preserve_count:]

    # Generate summary of older turns
    if summarize_fn is not None:
        try:
            summary_text = summarize_fn(older_turns)
        except Exception as e:
            logger.warning(f"Summarization function failed: {e}. Falling back to default summary generator.")
            summary_text = _default_summarize_turns(older_turns)
    else:
        summary_text = _default_summarize_turns(older_turns)

    summary_message = {
        "role": "system",
        "content": f"[Context Summary of Prior Conversation: {summary_text}]"
    }

    candidate_messages = ([system_message] if system_message else []) + [summary_message] + recent_turns
    cand_tokens = total_tokens(candidate_messages, encoding_name=encoding_name)

    if cand_tokens <= budget:
        return candidate_messages, cand_tokens, "summarize"

    # If summary + recent turns still exceed budget, apply trim_history on the candidate
    return trim_history(candidate_messages, budget, preserve_recent=preserve_recent, encoding_name=encoding_name)


def prepare_context(
    history: List[Dict[str, str]],
    retrieved_context: Optional[str] = None,
    user_message: str = "",
    max_tokens: int = 4096,
    reserve_tokens: int = 500,
    strategy: str = "trim",
    preserve_recent: int = 2,
    system_prompt: Optional[str] = None,
    summarize_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    encoding_name: str = "cl100k_base"
) -> Dict[str, Any]:
    """
    Prepares the full conversation payload before sending to the model:
    `system prompt + conversation history + retrieved context + user message + response reserve` <= max_tokens.

    Returns dict with:
        - "messages": final list of messages
        - "total_tokens": token count
        - "budget": safe context budget
        - "strategy_applied": strategy log string
    """
    budget = max_tokens - reserve_tokens
    if budget <= 0:
        budget = max(100, max_tokens // 2)

    # Assemble messages base
    built_messages = []
    
    # 1. System prompt
    if system_prompt:
        built_messages.append({"role": "system", "content": system_prompt})
    elif history and history[0].get("role") == "system":
        built_messages.append(history[0])

    # 2. Append existing history (excluding system prompt if already added)
    hist_start = 1 if (history and history[0].get("role") == "system") else 0
    candidate_history = list(history[hist_start:])

    # 3. Handle history trimming/summarization if system + history alone exceeds or leaves insufficient room
    temp_messages = list(built_messages) + candidate_history
    strategy_applied = "none"

    # Pre-check if system + history needs trimming/summarization
    user_msg_tokens = count_tokens(user_message, encoding_name=encoding_name) + 10
    allowed_hist_budget = budget - user_msg_tokens

    if total_tokens(temp_messages, encoding_name=encoding_name) > allowed_hist_budget:
        if strategy.lower() == "summarize":
            temp_messages, _, strategy_applied = summarize_history(
                temp_messages,
                budget=allowed_hist_budget,
                summarize_fn=summarize_fn,
                preserve_recent=preserve_recent,
                encoding_name=encoding_name
            )
        else:
            temp_messages, _, strategy_applied = trim_history(
                temp_messages,
                budget=allowed_hist_budget,
                preserve_recent=preserve_recent,
                encoding_name=encoding_name
            )
        built_messages = temp_messages
    else:
        built_messages = temp_messages

    # 4. Formulate RAG context and user message
    effective_user_content = user_message
    if retrieved_context and retrieved_context.strip():
        current_tok = total_tokens(built_messages, encoding_name=encoding_name)
        avail_for_rag = budget - current_tok - user_msg_tokens - 15

        rag_text = retrieved_context.strip()
        if avail_for_rag < count_tokens(rag_text, encoding_name=encoding_name):
            if avail_for_rag > 20:
                rag_text = _truncate_text_to_tokens(rag_text, avail_for_rag, encoding_name=encoding_name) + "\n[Truncated to fit context window budget]"
            else:
                rag_text = "[Context Omitted to fit context budget]"
            strategy_applied = (strategy_applied + "+rag_trimmed") if strategy_applied != "none" else "rag_trimmed"

        effective_user_content = render(ANSWER_TEMPLATE, context=rag_text, question=user_message)

    if user_message:
        built_messages.append({"role": "user", "content": effective_user_content})

    final_tok = total_tokens(built_messages, encoding_name=encoding_name)

    # Safety catch if still over budget
    if final_tok > budget:
        built_messages, final_tok, extra_strat = trim_history(
            built_messages,
            budget=budget,
            preserve_recent=preserve_recent,
            encoding_name=encoding_name
        )
        strategy_applied = (strategy_applied + "+" + extra_strat) if strategy_applied != "none" else extra_strat

    logger.info(f"[Context Manager] Prepared {len(built_messages)} messages | Tokens: {final_tok}/{budget} | Strategy: {strategy_applied}")

    return {
        "messages": built_messages,
        "total_tokens": final_tok,
        "budget": budget,
        "strategy_applied": strategy_applied
    }


def _truncate_text_to_tokens(text: str, max_tok: int, encoding_name: str = "cl100k_base") -> str:
    """Helper to safely truncate a text string to a target token limit."""
    if max_tok <= 0:
        return ""
    if count_tokens(text, encoding_name=encoding_name) <= max_tok:
        return text

    # Binary search or character slice estimate
    # Start with standard char ratio (~4 chars per token)
    char_limit = int(max_tok * 4)
    truncated = text[:char_limit]
    
    while count_tokens(truncated, encoding_name=encoding_name) > max_tok and len(truncated) > 0:
        char_limit = int(char_limit * 0.9)
        truncated = text[:char_limit]
        
    return truncated


def _default_summarize_turns(turns: List[Dict[str, str]]) -> str:
    """
    Default deterministic summarizer for conversation turns.
    Extracts key points, user requests, and assistant responses cleanly.
    """
    user_queries = []
    topics = []
    for msg in turns:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        # Clean RAG prefix if present in older messages
        if "Question:\n" in content:
            content = content.split("Question:\n")[-1]
        
        # Take first 100 characters of each message for compact representation
        short_snippet = content[:100].replace("\n", " ")
        if role == "user":
            user_queries.append(short_snippet)
        elif role == "assistant":
            topics.append(short_snippet)

    summary_parts = []
    if user_queries:
        summary_parts.append(f"User asked about: {'; '.join(user_queries[:3])}")
    if topics:
        summary_parts.append(f"Key responses provided: {'; '.join(topics[:3])}")

    return " | ".join(summary_parts) if summary_parts else "Prior user-assistant discussion on customs compliance and logistics."


class ContextManager:
    """
    High-level class for tracking conversation history, enforcing context budgets,
    and managing trim/summarization state across multi-turn sessions.
    """
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_context_tokens: int = 4096,
        response_reserve_tokens: int = 500,
        strategy: str = "trim",
        num_recent_turns_preserve: int = 2,
        model_name: str = "openai/gpt-oss-120b",
        encoding_name: str = "cl100k_base",
        summarize_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None
    ):
        self.max_context_tokens = max_context_tokens
        self.response_reserve_tokens = response_reserve_tokens
        self.strategy = strategy
        self.num_recent_turns_preserve = num_recent_turns_preserve
        self.model_name = model_name
        self.encoding_name = encoding_name
        self.summarize_fn = summarize_fn

        self.history: List[Dict[str, str]] = []
        if system_prompt:
            self.set_system_prompt(system_prompt)

    def set_system_prompt(self, system_prompt: str) -> None:
        """Sets or updates the system prompt at index 0 of history."""
        sys_msg = {"role": "system", "content": system_prompt}
        if self.history and self.history[0].get("role") == "system":
            self.history[0] = sys_msg
        else:
            self.history.insert(0, sys_msg)

    def add_user_message(self, content: str) -> None:
        """Appends a user message to history."""
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Appends an assistant response message to history."""
        self.history.append({"role": "assistant", "content": content})

    def get_prepared_payload(
        self,
        user_message: str,
        retrieved_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepares messages for an API request without mutating persistent history prematurely.
        """
        sys_prompt = self.history[0]["content"] if (self.history and self.history[0].get("role") == "system") else None
        
        # Existing conversation history excluding system prompt and current pending user message
        hist_start = 1 if sys_prompt else 0
        current_history = list(self.history[hist_start:])

        prep_result = prepare_context(
            history=current_history,
            retrieved_context=retrieved_context,
            user_message=user_message,
            max_tokens=self.max_context_tokens,
            reserve_tokens=self.response_reserve_tokens,
            strategy=self.strategy,
            preserve_recent=self.num_recent_turns_preserve,
            system_prompt=sys_prompt,
            summarize_fn=self.summarize_fn,
            encoding_name=self.encoding_name
        )
        return prep_result

    def ask(
        self,
        user_message: str,
        retrieved_context: Optional[str] = None,
        llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None
    ) -> Dict[str, Any]:
        """
        Executes a turn:
        1. Prepares budget-constrained context.
        2. Updates internal history with user message (and trimmed history if strategy was applied).
        3. Calls `llm_fn` to get model completion response.
        4. Adds assistant response to history.
        """
        prep = self.get_prepared_payload(user_message, retrieved_context)
        prepared_messages = prep["messages"]

        # Sync persistent history to the prepared messages state (preserving system + trimmed turns)
        self.history = list(prepared_messages)

        response_text = None
        if llm_fn:
            response_text = llm_fn(prepared_messages)
            if response_text:
                self.add_assistant_message(response_text)

        return {
            "response": response_text,
            "prepared_messages": prepared_messages,
            "total_tokens": prep["total_tokens"],
            "budget": prep["budget"],
            "strategy_applied": prep["strategy_applied"]
        }

    def reset_history(self) -> None:
        """Clears user/assistant turns while preserving system prompt."""
        sys_msg = self.history[0] if (self.history and self.history[0].get("role") == "system") else None
        self.history = [sys_msg] if sys_msg else []
