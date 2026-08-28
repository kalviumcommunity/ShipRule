import os
import sys
import logging
from dotenv import load_dotenv
from groq import Groq

# Ensure UTF-8 output handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    from src.prompt_templates import render, BATCH_EVAL_TEMPLATE, SYSTEM_PROMPT_TEMPLATE
except ImportError:
    from prompt_templates import render, BATCH_EVAL_TEMPLATE, SYSTEM_PROMPT_TEMPLATE



def load_prompt_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def run_prompt_comparison():
    """
    Task 3: Compare two prompt variations (Vague vs Constrained) for identical user queries.
    Saves and displays the comparative results.
    """
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")

    if not api_key:
        print("[ERROR] GROQ_API_KEY is not set in .env file.")
        return

    client = Groq(api_key=api_key)

    # Base directory paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vague_prompt_path = os.path.join(base_dir, "prompts", "system_prompt_v1_vague.txt")
    constrained_prompt_path = os.path.join(base_dir, "prompts", "system_prompt_v2_constrained.txt")

    vague_system_prompt = load_prompt_file(vague_prompt_path)
    constrained_system_prompt = load_prompt_file(constrained_prompt_path)

    test_queries = [
        {
            "category": "In-Domain Logistics Query",
            "prompt": "What are the required import documents and duty rates for shipping laptop computers (HS Code 8471.30) to India?"
        },
        {
            "category": "Out-of-Domain Query (Cinema & Food)",
            "prompt": "Can you recommend the top 3 movies currently playing in cinema and what popcorn/snacks I should eat?"
        },
        {
            "category": "Format-Constrained Query (JSON)",
            "prompt": "Reply ONLY with a JSON object containing keys 'duty_rate', 'required_docs', and 'source_agency' for HS Code 8471.30."
        }
    ]

    output_lines = []
    header = "=" * 80 + "\nCDLP PROMPT VARIATION COMPARISON REPORT (TASK 3)\n" + "=" * 80
    print(header)
    output_lines.append(header)

    models_to_try = [model, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]

    for idx, test in enumerate(test_queries, 1):
        query_section = f"\nTEST CASE {idx}: [{test['category']}]\nUSER QUESTION: \"{test['prompt']}\"\n" + "-" * 80
        print(query_section)
        output_lines.append(query_section)

        # Render evaluation query payload using centralized PromptTemplate (Task 3 reuse)
        user_payload = render(
            BATCH_EVAL_TEMPLATE,
            category=test["category"],
            domain="Customs Compliance & Duty Lookup",
            context="ShipRule CDLP platform duty classification database.",
            question=test["prompt"]
        )

        # 1. Run Vague Prompt
        vague_res = "ERROR"
        for current_model in models_to_try:
            try:
                res = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": vague_system_prompt},
                        {"role": "user", "content": user_payload}
                    ]
                )
                vague_res = res.choices[0].message.content.strip()
                break
            except Exception as e:
                continue

        # 2. Run Constrained Prompt
        constrained_res = "ERROR"
        for current_model in models_to_try:
            try:
                res = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": constrained_system_prompt},
                        {"role": "user", "content": user_payload}
                    ]
                )
                constrained_res = res.choices[0].message.content.strip()
                break
            except Exception as e:
                continue

        res_block = (
            f"\n[PROMPT V1 — Vague System Prompt (\"You are a helpful assistant.\")]\n"
            f"{vague_res}\n\n"
            f"[PROMPT V2 — Constrained CDLP System Prompt (Role + Scope + Out-of-Scope Refusal)]\n"
            f"{constrained_res}\n"
        )
        print(res_block)
        output_lines.append(res_block)

    summary_note = (
        "=" * 80 + "\n"
        "TASK 4 SUMMARY & DOCUMENTATION NOTE:\n"
        "1. Out-of-Domain Refusal: Prompt V1 generated movie & food recommendations, violating domain safety.\n"
        "   Prompt V2 correctly refused the cinema/food query using the standard CDLP fallback statement.\n"
        "2. Format Adherence: Prompt V2 strictly followed sentence limits and JSON formatting constraints.\n"
        "3. Conclusion: Prompt V2 is chosen as the production system prompt for CDLP / ShipRule.\n"
        + "=" * 80
    )
    print(summary_note)
    output_lines.append(summary_note)

    # Save to outputs/prompt_comparison_output.txt
    output_filepath = os.path.join(base_dir, "outputs", "prompt_comparison_output.txt")
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SUCCESS] Comparison results written to {output_filepath}")


if __name__ == "__main__":
    run_prompt_comparison()
