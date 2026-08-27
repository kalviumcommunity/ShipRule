"""
ShipRule CDLP - Token Counter & Cost Estimation Module
=====================================================
Demonstrates token counting, cost estimation for input vs output tokens,
and length-to-token relationship analysis using tiktoken (cl100k_base / o200k_base).
"""

import os
import sys
from typing import Dict, List, Any
import tiktoken

# Ensure UTF-8 output handling on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Default pricing rates per 1 Million tokens (as of standard OpenAI pricing)
PRICING_TABLE = {
    "gpt-4o": {
        "name": "GPT-4o",
        "input_per_million": 2.50,
        "output_per_million": 10.00,
    },
    "gpt-4o-mini": {
        "name": "GPT-4o-mini",
        "input_per_million": 0.15,
        "output_per_million": 0.60,
    },
    "gpt-3.5-turbo": {
        "name": "GPT-3.5-Turbo",
        "input_per_million": 0.50,
        "output_per_million": 1.50,
    }
}


_TOKENIZER_CACHE = {}

def get_tokenizer(encoding_name: str = "cl100k_base"):
    """Returns a tiktoken encoding instance, handling SSL cert issues and timeouts."""
    if encoding_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[encoding_name]

    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    orig_get = requests.get
    orig_session_get = requests.Session.get

    def unverified_get(*args, **kwargs):
        kwargs["verify"] = False
        if "timeout" not in kwargs:
            kwargs["timeout"] = 3
        return orig_get(*args, **kwargs)

    def unverified_session_get(self, *args, **kwargs):
        kwargs["verify"] = False
        if "timeout" not in kwargs:
            kwargs["timeout"] = 3
        return orig_session_get(self, *args, **kwargs)

    try:
        requests.get = unverified_get
        requests.Session.get = unverified_session_get
        enc = tiktoken.get_encoding(encoding_name)
        _TOKENIZER_CACHE[encoding_name] = enc
        return enc
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            _TOKENIZER_CACHE["cl100k_base"] = enc
            return enc
        except Exception:
            _TOKENIZER_CACHE[encoding_name] = None
            return None
    finally:
        requests.get = orig_get
        requests.Session.get = orig_session_get


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Counts the number of tokens in a string using tiktoken or fallback heuristic."""
    if not text:
        return 0
    try:
        enc = get_tokenizer(encoding_name)
        if enc:
            return len(enc.encode(text))
    except Exception:
        pass
    # Fallback token estimate: ~1.3 tokens per word, minimum 1 token for non-empty text
    words = text.split()
    if not words:
        return max(1, len(text) // 4)
    return max(1, int(len(words) * 1.3) + len(text) // 100)


def tokenize_with_chunks(text: str, encoding_name: str = "cl100k_base") -> List[str]:
    """Tokenizes text and returns individual decoded token strings."""
    enc = get_tokenizer(encoding_name)
    token_ids = enc.encode(text)
    return [enc.decode([token_id]) for token_id in token_ids]


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model_key: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Calculates cost for given input and output token counts,
    accounting for differing input vs output price rates.
    """
    pricing = PRICING_TABLE.get(model_key, PRICING_TABLE["gpt-4o-mini"])
    input_rate = pricing["input_per_million"] / 1_000_000
    output_rate = pricing["output_per_million"] / 1_000_000

    input_cost = input_tokens * input_rate
    output_cost = output_tokens * output_rate
    total_cost = input_cost + output_cost

    return {
        "model": pricing["name"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "input_rate_per_m": pricing["input_per_million"],
        "output_rate_per_m": pricing["output_per_million"],
    }


def analyze_samples() -> List[Dict[str, Any]]:
    """Analyzes token counts across different project corpus samples of varying lengths."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    constrained_prompt_path = os.path.join(base_dir, "prompts", "system_prompt_v2_constrained.txt")
    
    full_doc_content = ""
    if os.path.exists(constrained_prompt_path):
        with open(constrained_prompt_path, "r", encoding="utf-8") as f:
            full_doc_content = f.read().strip()
    else:
        full_doc_content = (
            "ShipRule CDLP is an automated customs compliance and duty classification platform.\n"
            "Source traceability ensures every duty lookup is verified against official trade regulations.\n"
            "It validates import documents, calculates tariffs based on HS Codes, and flags restricted goods."
        )

    samples = [
        {
            "name": "Sample 1: Short User Question (Input)",
            "type": "Short Question",
            "text": "What are the required import documents and duty rates for shipping laptop computers (HS Code 8471.30) to India?"
        },
        {
            "name": "Sample 2: Medium Model Response (Output Paragraph)",
            "type": "Paragraph Response",
            "text": (
                "Hello! Source traceability is critical for customs duty lookups because linking every duty rate "
                "and required document to an official government agency URL provides an auditable compliance "
                "record and eliminates risk during customs clearance."
            )
        },
        {
            "name": "Sample 3: Full System Prompt / Regulatory Document (Input Document)",
            "type": "Full Document",
            "text": full_doc_content
        },
        {
            "name": "Sample 4: Structured JSON Output (Output)",
            "type": "JSON Output",
            "text": (
                '{\n'
                '  "hs_code": "8471.30",\n'
                '  "product": "Laptop Computers",\n'
                '  "duty_rate": "7.5%",\n'
                '  "required_documents": [\n'
                '    "Commercial Invoice",\n'
                '    "Bill of Lading",\n'
                '    "BIS Registration Certificate",\n'
                '    "Certificate of Origin"\n'
                '  ],\n'
                '  "restricted_status": "Restricted (Import License Required)",\n'
                '  "source_agency": "Directorate General of Foreign Trade (DGFT), India"\n'
                '}'
            )
        }
    ]

    enc = get_tokenizer("cl100k_base")

    results = []
    for sample in samples:
        text = sample["text"]
        char_count = len(text)
        word_count = len(text.split())
        token_count = len(enc.encode(text))
        chars_per_token = (char_count / token_count) if token_count > 0 else 0
        tokens_per_word = (token_count / word_count) if word_count > 0 else 0

        results.append({
            "name": sample["name"],
            "type": sample["type"],
            "text": text,
            "char_count": char_count,
            "word_count": word_count,
            "token_count": token_count,
            "chars_per_token": chars_per_token,
            "tokens_per_word": tokens_per_word,
        })

    return results


def run_length_token_experiment() -> List[Dict[str, Any]]:
    """
    Demonstrates the relationship between text length and token count across
    different text types (Plain English, Long/Unusual Words, Code, Multilingual, Whitespace).
    """
    enc = get_tokenizer("cl100k_base")

    test_cases = [
        {
            "category": "Standard English (Customs Rule)",
            "text": "Every shipment requires a commercial invoice and bill of lading before customs clearance.",
            "note": "Standard English words tokenize efficiently (~1 to 1.3 tokens/word, ~4 chars/token)."
        },
        {
            "category": "Long / Compound Technical Words",
            "text": "Pneumonoultramicroscopicsilicovolcanoconiosis antidisestablishmentarianism electroencephalography",
            "note": "Rare or long compound words break into many subwords, increasing token density per word."
        },
        {
            "category": "Python Code Snippet",
            "text": "def calculate_duty(item_val: float, tariff_pct: float) -> float:\n    return round(item_val * (tariff_pct / 100.0), 2)",
            "note": "Code contains punctuation, indents, and underscores, which split into multiple individual tokens."
        },
        {
            "category": "Multilingual: Hindi (Devanagari Script)",
            "text": "सीमा शुल्क और आयात नियमों के लिए आधिकारिक दस्तावेजों की जांच आवश्यक है।",
            "note": "Non-Latin scripts use multiple bytes per character, resulting in high token counts per word."
        },
        {
            "category": "Multilingual: Chinese (Mandarin)",
            "text": "所有进出口货物必须遵守海关总署的相关规定并提交合规证明文件。",
            "note": "Characters represent whole concepts/morphemes, yielding ~1.5 to 2 tokens per character in cl100k."
        },
        {
            "category": "Special Symbols & Repeated Whitespace",
            "text": "   === [SHIPRULE_DUTY_LOOKUP] ===   >>> ID: #8471-30-IN <<<   $$$ 100.00% $$$   ",
            "note": "Punctuation clusters and spaces often generate isolated tokens."
        }
    ]

    results = []
    for case in test_cases:
        text = case["text"]
        char_count = len(text)
        word_count = len(text.split())
        tokens = enc.encode(text)
        token_count = len(tokens)
        chars_per_token = (char_count / token_count) if token_count > 0 else 0
        tokens_per_word = (token_count / word_count) if word_count > 0 else 0

        results.append({
            "category": case["category"],
            "text": text,
            "char_count": char_count,
            "word_count": word_count,
            "token_count": token_count,
            "chars_per_token": chars_per_token,
            "tokens_per_word": tokens_per_word,
            "note": case["note"]
        })

    return results


def run_full_token_analysis() -> str:
    """Executes all 5 tasks and returns a formatted report string matching the specified format."""
    report_lines = []

    def p(line: str = ""):
        print(line)
        report_lines.append(line)

    enc = get_tokenizer("cl100k_base")

    # Sample dataset matching CDLP RAG queries and responses of varying lengths
    samples = [
        {
            "sample_num": 1,
            "title": "SAMPLE 1: SHORT SHIPPING QUERY",
            "category": "Short Query",
            "input_text": "What are the shipping rules for this order?",
            "output_text": "I do not have verified customs regulation data for this query in CDLP."
        },
        {
            "sample_num": 2,
            "title": "SAMPLE 2: MEDIUM CUSTOMS DUTY & COMPLIANCE QUERY",
            "category": "Paragraph Query & Response",
            "input_text": "What are the required import documents and duty rates for shipping laptop computers (HS Code 8471.30) to India?",
            "output_text": "Laptops under HS Code 8471.30 entering India require a Commercial Invoice, Bill of Lading, BIS Registration Certificate, and Certificate of Origin. The standard basic customs duty rate is 7.5% subject to import licensing under DGFT guidelines."
        },
        {
            "sample_num": 3,
            "title": "SAMPLE 3: FULL REGULATORY KNOWLEDGE BASE & SYSTEM PROMPT",
            "category": "Full Document / Multi-line Prompt & Context",
            "input_text": (
                "ShipRule CDLP is an automated customs compliance and duty classification platform.\n"
                "Source traceability ensures every duty lookup is verified against official trade and customs regulations.\n"
                "You are an official AI Support Assistant for the Customs Duty & Documentation Lookup Platform (CDLP / ShipRule).\n"
                "Role & Scope: You provide answers ONLY regarding customs duties, import documentation requirements, "
                "HS code classifications, restriction status, government source traceability, and shipment/logistics regulations.\n"
                "Out-of-Scope Constraints: You MUST NOT answer questions outside logistics and customs.\n"
                "User Query: Summarize the core compliance rules and verification standards enforced by ShipRule CDLP."
            ),
            "output_text": (
                "ShipRule CDLP enforces automated customs classification by verifying duty rates, mandatory import documentation "
                "(such as Commercial Invoices and Bills of Lading), and restriction statuses directly against official regulatory sources. "
                "Every lookup maintains strict source traceability to ensure auditability and compliance across cross-border shipments."
            )
        }
    ]

    # Rates: GPT-4o-mini ($0.15 / 1M input, $0.60 / 1M output)
    rate_in_per_token = 0.15 / 1_000_000
    rate_out_per_token = 0.60 / 1_000_000

    p("=" * 70)
    p("         SHIPRULE CDLP — TOKEN COUNTING & COST ESTIMATION REPORT")
    p("=" * 70)
    p("Tokenizer: tiktoken (Encoding: cl100k_base)")
    p("Pricing Model: GPT-4o-mini (Input: $0.15 / 1M tokens | Output: $0.60 / 1M tokens)")
    p("")

    for s in samples:
        in_text = s["input_text"]
        out_text = s["output_text"]
        
        in_len = len(in_text)
        in_tokens = len(enc.encode(in_text))
        
        out_len = len(out_text)
        out_tokens = len(enc.encode(out_text))
        
        in_cost = in_tokens * rate_in_per_token
        out_cost = out_tokens * rate_out_per_token
        total_cost = in_cost + out_cost

        p("=" * 70)
        p(s["title"])
        p("=" * 70)
        p("")
        p("Input Text:")
        p(in_text)
        p("")
        p(f"Input Length: {in_len} characters")
        p(f"Input Tokens: {in_tokens}")
        p("")
        p("Model Output:")
        p(out_text)
        p("")
        p(f"Output Length: {out_len} characters")
        p(f"Output Tokens: {out_tokens}")
        p("")
        p("Cost Estimate (GPT-4o-mini):")
        p(f"Input Cost  = {in_tokens} tokens × ($0.15 / 1,000,000) = ${in_cost:.8f}")
        p(f"Output Cost = {out_tokens} tokens × ($0.60 / 1,000,000) = ${out_cost:.8f}")
        p(f"Total Cost  = Input Cost (${in_cost:.8f}) + Output Cost (${out_cost:.8f}) = ${total_cost:.8f}")
        p(f"Cost per 100,000 Calls = ${total_cost * 100_000:.4f}")
        p("=" * 70)
        p("")

    # Task 4 Demonstration
    p("=" * 70)
    p("TASK 4: LENGTH VS. TOKEN RELATIONSHIP DEMONSTRATION")
    p("=" * 70)
    p("Demonstrating that text length and token count track together but are not strictly proportional:")
    p("-" * 70)
    p(f"{'Category':<32} | {'Chars':<6} | {'Words':<6} | {'Tokens':<7} | {'Chars/Tok':<10}")
    p("-" * 70)

    experiment_cases = run_length_token_experiment()
    for case in experiment_cases:
        p(f"{case['category']:<32} | {case['char_count']:<6} | {case['word_count']:<6} | {case['token_count']:<7} | {case['chars_per_token']:<10.2f}")

    p("\nKey Takeaways:")
    p("1. Tracking: In standard English prose, token count generally scales with character length (~4 characters/token).")
    p("2. Divergence (Non-proportionality):")
    p("   - Long/Uncommon Words: Break into multiple subword tokens (e.g. 97 chars / 3 words -> 28 tokens).")
    p("   - Code & Special Characters: Syntax symbols, brackets, and whitespace create extra isolated tokens.")
    p("   - Non-Latin Scripts: Devanagari (Hindi) and Chinese require multiple byte tokens per character (e.g. 71 chars -> 75 tokens).")
    p("=" * 70)

    return "\n".join(report_lines)



def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, "outputs", "token_analysis_output.txt")
    
    # Run analysis and print to terminal
    report = run_full_token_analysis()

    # Save output to outputs/token_analysis_output.txt
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[SUCCESS] Token analysis & cost estimation report saved to:\n  {output_path}")


if __name__ == "__main__":
    main()
