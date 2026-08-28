

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


import json
from src.context_manager import prepare_context, total_tokens

# Default LLM Output Control Parameters
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 600


def parse_and_validate_json_response(raw_text: str) -> dict:
    """
    Parses raw AI completion text into JSON and validates required fields ('answer' and 'source').
    Safely extracts JSON objects from Markdown code fences (```json ... ```) or surrounding text.
    Raises ValueError with descriptive standard error message on validation failure.
    """
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
        logging.warning("[RAW MODEL RESPONSE]: %s", repr(raw_text))
        raise ValueError("Malformed AI response")

    text = raw_text.strip()
    logging.info("[RAW MODEL RESPONSE]: %s", repr(text))

    # Safely extract JSON object bounds '{' ... '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        logging.warning("[RAW MODEL RESPONSE NO JSON BOUNDS]: %s", repr(text))
        raise ValueError("Malformed AI response")

    json_candidate = text[first_brace : last_brace + 1]

    try:
        data = json.loads(json_candidate)
    except Exception as parse_err:
        logging.warning("[RAW MODEL RESPONSE PARSE FAIL] %s: %s", repr(json_candidate), parse_err)
        raise ValueError("Malformed AI response")

    if not isinstance(data, dict):
        logging.warning("[RAW MODEL RESPONSE NOT DICT]: %s", repr(data))
        raise ValueError("Malformed AI response")

    if "answer" not in data or data["answer"] is None:
        raise ValueError("Missing required field: answer")
    if not isinstance(data["answer"], str) or not data["answer"].strip():
        raise ValueError("Missing required field: answer")

    if "source" not in data or data["source"] is None:
        raise ValueError("Missing required field: source")
    if not isinstance(data["source"], str):
        raise ValueError("Missing required field: source")

    return {
        "answer": data["answer"].strip(),
        "source": data["source"].strip()
    }


def run_chat_completion(
    question=None,
    context=None,
    model_override=None,
    api_key_override=None,
    system_prompt_override=None,
    messages_override=None,
    history=None,
    max_context_tokens=4096,
    response_reserve_tokens=500,
    strategy="trim",
    preserve_recent=2,
    temperature_override=None,
    max_tokens_override=None
):
    """
    Send a question and optional RAG context to Groq/LLM and return structured JSON model response.
    Supports multi-turn history, context budgeting, output control parameters, defensive parsing,
    and single retry on malformed JSON outputs.
    """

    # Load .env
    load_dotenv()

    # -----------------------------------------
    # Get Groq configuration & parameters
    # -----------------------------------------
    api_key = (
        api_key_override
        if api_key_override is not None
        else (os.getenv("GROQ_API_KEY"))
    )

    model = (
        model_override
        or os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
    )

    temperature = (
        temperature_override
        if temperature_override is not None
        else float(os.getenv("LLM_TEMPERATURE", "0.0"))
    )

    max_tokens = (
        max_tokens_override
        if max_tokens_override is not None
        else int(os.getenv("LLM_MAX_TOKENS", str(LLM_MAX_TOKENS)))
    )

    logging.info("=== Initializing Groq / LLM Client ===")
    logging.info("[Config] Model: %s | Temperature: %.2f | Max Output Tokens: %d", model, temperature, max_tokens)
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
    # Create or prepare messages with Context Budgeting
    # -----------------------------------------
    if messages_override:
        messages = messages_override
    else:
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
                    "You are an official AI Support Assistant for the Customs Duty & Documentation Lookup Platform (CDLP / ShipRule).\n"
                    "Return ONLY valid JSON:\n"
                    "{\n  \"answer\": \"concise answer\",\n  \"source\": \"source name\"\n}\n"
                    "Keep the answer concise and within approximately 150 words.\n"
                    "Do not use Markdown.\nDo not use code fences.\nDo not add text outside the JSON object."
                )

        prep_result = prepare_context(
            history=history or [],
            retrieved_context=context,
            user_message=question or "",
            max_tokens=max_context_tokens,
            reserve_tokens=response_reserve_tokens,
            strategy=strategy,
            preserve_recent=preserve_recent,
            system_prompt=system_prompt
        )
        messages = prep_result["messages"]
        logging.info(
            "[Context Budget] Tokens: %d / %d (Reserved: %d) | Strategy Applied: %s",
            prep_result["total_tokens"],
            prep_result["budget"],
            response_reserve_tokens,
            prep_result["strategy_applied"]
        )

    logging.info("[Request] Target Model: %s (Temp: %.2f, Max Tokens: %d)", model, temperature, max_tokens)
    logging.info("[Request] User Question: %s", question)

    # List of candidate models to try in case of model error or availability fallback
    models_to_try = [model]
    for fallback in ["groq/compound-mini", "qwen/qwen3.6-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    last_error = None
    for current_model in models_to_try:
        try:
            # Attempt 1: Call API with response_format={"type": "json_object"} if supported
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                )
            except (TypeError, GroqAPIError):
                # Fallback if specific model or API version doesn't accept response_format kwarg
                response = client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            reply_text = response.choices[0].message.content

            # Parse and validate JSON structure
            try:
                return parse_and_validate_json_response(reply_text)
            except ValueError as parse_err:
                logging.warning(
                    "Initial JSON parsing failed (%s). Retrying ONCE with stronger JSON instruction...",
                    parse_err
                )
                # Retry once with a stronger instruction
                retry_messages = list(messages) + [
                    {"role": "assistant", "content": reply_text or ""},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was incomplete/truncated.\n"
                            "Return a COMPLETE valid JSON object.\n"
                            "Use exactly these fields:\n"
                            "{\"answer\":\"...\",\"source\":\"...\"}\n"
                            "Keep the answer concise.\n"
                            "Do not exceed approximately 120 words.\n"
                            "Do not use Markdown or code fences.\n"
                            "Return nothing except the JSON object."
                        )
                    }
                ]
                try:
                    retry_response = client.chat.completions.create(
                        model=current_model,
                        messages=retry_messages,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"}
                    )
                except (TypeError, GroqAPIError):
                    retry_response = client.chat.completions.create(
                        model=current_model,
                        messages=retry_messages,
                        temperature=0.0,
                        max_tokens=max_tokens
                    )
                
                retry_text = retry_response.choices[0].message.content
                try:
                    return parse_and_validate_json_response(retry_text)
                except ValueError as retry_parse_err:
                    logging.error("Retry JSON parsing failed: %s", retry_parse_err)
                    return {
                        "answer": f"Error: Unable to process response. ({retry_parse_err})",
                        "source": "CDLP System",
                        "error": str(retry_parse_err)
                    }

        except GroqAPIError as e:
            last_error = e
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

    return {
        "answer": "Error: Failed to obtain response from language model.",
        "source": "CDLP System",
        "error": "API Execution Error"
    }


if __name__ == "__main__":
    question = input("Ask your question: ")
    res = run_chat_completion(question)
    print(res)

