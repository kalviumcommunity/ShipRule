import os
import logging
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APIError

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def run_chat_completion(custom_messages=None, model_override=None, api_key_override=None):
    """
    Executes a chat completion call against an OpenAI-compatible API,
    logs the request, response, and token usage, and handles common errors cleanly.
    """
    # Task 1: Load configuration from environment variables
    load_dotenv()
    
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = api_key_override if api_key_override is not None else os.getenv("OPENAI_API_KEY", "")
    model = model_override or os.getenv("CHAT_MODEL", "gpt-4o-mini")

    logging.info("=== Initializing OpenAI-Compatible LLM Client ===")
    logging.info("[Config] Base URL: %s", base_url)
    logging.info("[Config] Model: %s", model)
    logging.info("[Config] API Key Configured: %s", "Yes" if api_key else "No (Empty)")

    if not api_key:
        logging.warning("[Warning] OPENAI_API_KEY is empty. The completion request will likely fail unless using an unauthenticated local endpoint.")

    # Initialize OpenAI client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key or "placeholder_key"
    )

    # Default messages if none provided
    messages = custom_messages or [
        {
            "role": "system",
            "content": "You are a concise customs compliance assistant for ShipRule CDLP."
        },
        {
            "role": "user",
            "content": "Say hello in one sentence and explain why source traceability matters for customs duty lookups."
        }
    ]

    # Task 3: Log outgoing request payload
    logging.info("[Request] Target Model: %s", model)
    logging.info("[Request] Messages Payload: %s", messages)

    try:
        # Task 2: Send chat completion request
        logging.info("[API Call] Sending chat completion request...")
        resp = client.chat.completions.create(
            model=model,
            messages=messages
        )

        reply_text = resp.choices[0].message.content
        
        # Task 2 & 3: Print reply and log response & token usage
        print("\n--- Model Response ---")
        print(reply_text)
        print("----------------------\n")

        logging.info("[Response] Content: %s", reply_text)

        if hasattr(resp, "usage") and resp.usage:
            logging.info(
                "[Usage] Tokens -> Prompt: %s | Completion: %s | Total: %s",
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
                resp.usage.total_tokens
            )
        else:
            logging.info("[Usage] Token usage metadata not returned by provider.")

        return resp

    # Task 4: Error Handling
    except AuthenticationError as e:
        print("\n[ERROR] Auth failed (401): Check OPENAI_API_KEY in your .env file.")
        logging.error("AuthenticationError (401): %s", e)
    except RateLimitError as e:
        print("\n[ERROR] Rate limited (429): Slow down and retry with backoff.")
        logging.error("RateLimitError (429): %s", e)
    except APIConnectionError as e:
        print(f"\n[ERROR] Connection failed: Unable to reach base URL ({base_url}).")
        logging.error("APIConnectionError: %s", e)
    except APIError as e:
        print(f"\n[ERROR] API Error: {e.message if hasattr(e, 'message') else str(e)}")
        logging.error("APIError: %s", e)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error occurred: {str(e)}")
        logging.error("Unexpected Exception: %s", e, exc_info=True)

    return None

if __name__ == "__main__":
    run_chat_completion()
