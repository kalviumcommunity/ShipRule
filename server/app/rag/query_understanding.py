"""
ShipRule CDLP - Query Understanding & Normalization Module
===========================================================
Performs language correction, typo resolution, and phrasing normalization
on user queries before vector embedding and semantic retrieval.

STRICT PRINCIPLE:
Correct language without changing meaning or intent. Never hardcode synonyms
or query-specific rules. Gracefully fall back to original query if API fails.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

QUERY_CORRECTION_SYSTEM_PROMPT = (
    "You are a linguistic query normalizer for a RAG retrieval system.\n"
    "Your task is to correct spelling mistakes, grammatical errors, typos, and awkward phrasing in the user's query.\n\n"
    "STRICT RULES:\n"
    "1. Correct ONLY language, spelling, grammar, and phrasing.\n"
    "2. NEVER answer the question or invent information.\n"
    "3. PRESERVE ALL original intent, entities, countries, products, dates, numbers, actions, and constraints.\n"
    "4. DO NOT add extra facts, categories, or technical terms that were not present or implied in the user's query.\n"
    "5. If the query is already clear and grammatically correct, set normalized_query equal to original_query and changed to false.\n"
    "6. You MUST respond ONLY with a valid JSON object in this exact schema:\n"
    '{"original_query": "<original input>", "normalized_query": "<corrected query>", "changed": true}'
)


def normalize_query(
    query: str,
    client: Optional[Any] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Normalizes a user query by fixing typos, spelling errors, and awkward phrasing
    via the configured LLM, while preserving intent and constraints.

    Fallback behavior:
    If API call fails, times out, returns malformed JSON, or is offline,
    gracefully returns normalized_query equal to original_query with changed=False.

    Args:
        query: Raw user search query.
        client: Optional LLM client instance (e.g. Groq client).
        model: Optional model name string.

    Returns:
        Dict with keys: 'original_query', 'normalized_query', 'changed', 'error'.
    """
    if query is None or not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    cleaned_query = query.strip()

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    # If no API key or dummy local key and no mock client passed, perform offline clean fallback
    if not (api_key or client) or (api_key and api_key.startswith("gsk_local_key") and client is None):
        return {
            "original_query": cleaned_query,
            "normalized_query": cleaned_query,
            "changed": False,
            "error": None
        }

    try:
        from groq import Groq
        groq_client = client or Groq(api_key=api_key)
        llm_model = model or os.getenv("CHAT_MODEL", "groq/compound-mini")

        user_prompt = f"Normalize this user query:\n{cleaned_query}"

        resp = groq_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": QUERY_CORRECTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=250,
            response_format={"type": "json_object"} if hasattr(groq_client, "chat") else None
        )

        reply_content = resp.choices[0].message.content.strip()
        data = json.loads(reply_content)

        normalized = data.get("normalized_query")
        if isinstance(normalized, str) and normalized.strip():
            norm_str = normalized.strip()
            is_changed = bool(data.get("changed", norm_str.lower() != cleaned_query.lower()))
            return {
                "original_query": cleaned_query,
                "normalized_query": norm_str,
                "changed": is_changed,
                "error": None
            }
        else:
            logger.warning("Query correction model returned empty normalized_query field.")
            return {
                "original_query": cleaned_query,
                "normalized_query": cleaned_query,
                "changed": False,
                "error": "empty_normalized_query"
            }

    except Exception as exc:
        logger.warning(f"Query normalization API call failed: {exc}. Falling back to original query.")
        return {
            "original_query": cleaned_query,
            "normalized_query": cleaned_query,
            "changed": False,
            "error": str(exc)
        }
