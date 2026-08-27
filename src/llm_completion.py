

import os
import sys
import logging

from dotenv import load_dotenv
from groq import (
    Groq,
    AuthenticationError as GroqAuthError,
    RateLimitError as GroqRateError,
    APIConnectionError as GroqConnError,
    APIError as GroqAPIError,
)


# Ensure UTF-8 output handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Configure logging - use WARNING by default to keep CLI interaction clean
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def run_chat_completion(
    question,
    context=None,
    model_override=None,
    api_key_override=None,
    system_prompt_override=None
):
    """
    Send a question and optional RAG context to Groq/LLM and return the model response.
    """

    # Load .env
    load_dotenv()

    # -----------------------------------------
    # Get Groq configuration
    # -----------------------------------------
    api_key = (
        api_key_override
        if api_key_override is not None
        else (os.getenv("GROQ_API_KEY")  )
    )

    model = (
        model_override
        or os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
    )

    logging.info("=== Initializing Groq / LLM Client ===")
    logging.info("[Config] Model: %s", model)
    logging.info(
        "[Config] API Key Configured: %s",
        "Yes" if api_key else "No"
    )

    # -----------------------------------------
    # Check API key
    # -----------------------------------------
    if not api_key:
        print(
            "\n[ERROR] GROQ_API_KEY is not configured."
        )
        print(
            "Please add GROQ_API_KEY to your .env file."
        )
        return None

    # -----------------------------------------
    # Initialize Groq client
    # -----------------------------------------
    client = Groq(api_key=api_key)

    # -----------------------------------------
    # Create messages with RAG Context & System Role
    # -----------------------------------------
    if system_prompt_override:
        system_prompt = system_prompt_override
    else:
        # Load constrained production system prompt for CDLP / ShipRule
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_file = os.path.join(base_dir, "prompts", "system_prompt_v2_constrained.txt")
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        else:
            system_prompt = (
                "You are an official AI Support Assistant for the Customs Duty & Documentation Lookup Platform (CDLP / ShipRule). "
                "Answer ONLY questions related to logistics, customs duties, import documents, HS codes, and shipment rules in 2-3 sentences. "
                "Refuse non-logistics queries strictly."
            )

    if context:
        user_content = f"Retrieved Context:\n{context}\n\nQuestion:\n{question}"
    else:
        user_content = question

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    logging.info("[Request] Target Model: %s", model)
    logging.info("[Request] User Question: %s", question)

    # List of candidate models to try in case of model error or availability fallback
    models_to_try = [model]
    for fallback in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    last_error = None
    for current_model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=messages
            )

            # Get answer
            reply_text = response.choices[0].message.content
            return reply_text

        except GroqAPIError as e:
            last_error = e
            # If rate limit or auth error, don't keep retrying other models
            if hasattr(e, 'status_code') and e.status_code in (401, 429):
                break
            logging.warning("Model %s failed with %s. Trying fallback model...", current_model, e)
        except Exception as e:
            last_error = e
            break

    # -----------------------------------------
    # Error handling
    # -----------------------------------------
    if isinstance(last_error, GroqAuthError):
        print(
            "\n[ERROR] Authentication failed (401). "
            "Check your GROQ_API_KEY."
        )
        logging.error("AuthenticationError: %s", last_error)

    elif isinstance(last_error, GroqRateError):
        print(
            "\n[ERROR] Rate limit reached (429). "
            "Please wait and try again."
        )
        logging.error("RateLimitError: %s", last_error)

    elif isinstance(last_error, GroqConnError):
        print(
            "\n[ERROR] Could not connect to Groq."
        )
        logging.error("APIConnectionError: %s", last_error)

    elif isinstance(last_error, GroqAPIError):
        print(
            f"\n[ERROR] Groq API error: {last_error}"
        )
        logging.error("APIError: %s", last_error)

    elif last_error is not None:
        print(
            f"\n[ERROR] Unexpected error: {last_error}"
        )
        logging.error(
            "Unexpected Exception: %s",
            last_error,
            exc_info=True
        )

    return None


if __name__ == "__main__":
    question = input("Ask your question: ")
    run_chat_completion(question)
