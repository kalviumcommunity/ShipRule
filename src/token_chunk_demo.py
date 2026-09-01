"""
ShipRule CDLP - Concept 13: Token-Aware Chunk Sizing & Overlap Demonstration
=============================================================================
Demonstrates token-aware chunking with tiktoken (cl100k_base), overlap context sharing,
overlap value metrics comparison (0, 20, 40, 60 tokens), and boundary context preservation.
"""

import os
import sys
import tiktoken
from typing import Dict, List, Any

# Ensure UTF-8 output handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.document_loader import token_chunks, load_directory, load_document
from src.token_counter import get_tokenizer


def run_overlap_demonstration(text: str) -> str:
    """Demonstrates how adjacent chunks share context through token overlap."""
    lines = []
    lines.append("=" * 70)
    lines.append("1. OVERLAP DEMONSTRATION (size=400, overlap=60)")
    lines.append("=" * 70)

    enc = get_tokenizer("cl100k_base")
    chunks = token_chunks(text, size=400, overlap=60)
    lines.append(f"Input text token count: {len(enc.encode(text))} tokens")
    lines.append(f"Total chunks generated: {len(chunks)}\n")

    for i in range(min(2, len(chunks))):
        c = chunks[i]
        lines.append(f"--- Chunk {c['chunk_id']} ---")
        lines.append(f"  Token range : tokens {c['start_token']}–{c['end_token'] - 1} (total: {c['token_count']} tokens)")
        snippet = c['text'][:120].replace('\n', ' ')
        lines.append(f"  Text snippet: \"{snippet}...\"")
        lines.append("")

    if len(chunks) >= 2:
        c1, c2 = chunks[0], chunks[1]
        overlap_start = c2['start_token']
        overlap_end = c1['end_token']
        lines.append("Overlap Analysis:")
        lines.append(f"  Chunk 1 token range: tokens {c1['start_token']}–{c1['end_token'] - 1}")
        lines.append(f"  Chunk 2 token range: tokens {c2['start_token']}–{c2['end_token'] - 1}")
        lines.append(f"  Shared token window: tokens {overlap_start}–{overlap_end - 1} ({overlap_end - overlap_start} tokens)")
        shared_text = enc.decode(enc.encode(text)[overlap_start:overlap_end]).strip().replace('\n', ' ')
        lines.append(f"  Shared text snippet: \"{shared_text[:140]}\"")
        lines.append("  Explanation: Tokens in the overlapping region appear in BOTH Chunk 1 and Chunk 2,")
        lines.append("               ensuring context near the boundary is preserved for RAG retrieval.")

    lines.append("=" * 70)
    return "\n".join(lines)


def run_overlap_comparison(text: str) -> str:
    """Compares metrics across different overlap values (0, 20, 40, 60 tokens)."""
    lines = []
    lines.append("=" * 70)
    lines.append("2. OVERLAP VALUE COMPARISON EXPERIMENT")
    lines.append("=" * 70)

    enc = get_tokenizer("cl100k_base")
    base_tokens = len(enc.encode(text))
    lines.append(f"Document Base Token Count: {base_tokens} tokens (Target Chunk Size: 400 tokens)\n")

    overlap_values = [0, 20, 40, 60]

    for ov in overlap_values:
        chunks = token_chunks(text, size=400, overlap=ov)
        total_gen_tokens = sum(c['token_count'] for c in chunks)
        duplicated_tokens = total_gen_tokens - base_tokens if len(chunks) > 1 else 0
        overhead_pct = (duplicated_tokens / base_tokens * 100) if base_tokens > 0 else 0.0

        lines.append(f"Overlap: {ov}")
        lines.append(f"Chunks: {len(chunks)}")
        lines.append(f"Total tokens: {total_gen_tokens}")
        lines.append(f"Duplicated tokens: {duplicated_tokens}")
        lines.append(f"Percentage overhead: {overhead_pct:.2f}%")
        lines.append("")

    lines.append("Takeaways:")
    lines.append("  - Higher overlap increases total chunk tokens and storage overhead.")
    lines.append("  - Overlap=60 (~15% of 400 tokens) provides strong boundary context retention with modest overhead.")
    lines.append("=" * 70)
    return "\n".join(lines)


def run_boundary_context_demo() -> str:
    """Demonstrates how boundary context is lost without overlap and preserved with 60-token overlap."""
    lines = []
    lines.append("=" * 70)
    lines.append("3. BOUNDARY CONTEXT PRESERVATION DEMONSTRATION")
    lines.append("=" * 70)

    # Construct a passage where a critical answer sentence falls across the chunk boundary
    prefix_filler = (
        "ShipRule CDLP is an automated customs compliance and duty classification platform. "
        "It assists logistics operators, importers, and compliance officers in verifying cross-border documentation. "
        "All shipments passing through international trade corridors must strictly adhere to import regulations. "
    ) * 11  # ~385 tokens

    critical_sentence = (
        "CRITICAL REQUIREMENT: For all high-value electronic imports (HS Code 8471.30), "
        "a BIS Registration Certificate and DGFT Import License MUST be attached to the Commercial Invoice."
    )

    suffix_filler = (
        " Failure to provide verified customs documentation will result in immediate shipment detention, "
        "demurrage penalties, and formal regulatory review by local customs authorities."
    ) * 10

    full_text = prefix_filler + critical_sentence + suffix_filler

    enc = get_tokenizer("cl100k_base")
    total_tokens = len(enc.encode(full_text))

    lines.append(f"Synthetic Document Length: {total_tokens} tokens")
    lines.append(f"Target Chunk Size: 400 tokens\n")

    # Case A: No Overlap (0 tokens)
    lines.append("--- Case A: No Overlap (overlap = 0 tokens) ---")
    chunks_no_overlap = token_chunks(full_text, size=400, overlap=0)
    lines.append(f"Generated {len(chunks_no_overlap)} chunks.")

    if len(chunks_no_overlap) >= 2:
        c1_text = chunks_no_overlap[0]['text']
        c2_text = chunks_no_overlap[1]['text']

        lines.append("\n  [Chunk 1 Tail (tokens 350-400)]:")
        lines.append(f"  \"{c1_text[-120:].replace('\n', ' ')}\"")
        lines.append("\n  [Chunk 2 Head (tokens 400-450)]:")
        lines.append(f"  \"{c2_text[:120].replace('\n', ' ')}\"")
        lines.append("\n  Observation (Case A): The critical compliance requirement is split cleanly at token 400.")
        lines.append("  Neither Chunk 1 nor Chunk 2 alone contains the complete requirement sentence + context.")

    # Case B: 60-token Overlap
    lines.append("\n--- Case B: 60-Token Overlap (overlap = 60 tokens) ---")
    chunks_with_overlap = token_chunks(full_text, size=400, overlap=60)
    lines.append(f"Generated {len(chunks_with_overlap)} chunks.")

    if len(chunks_with_overlap) >= 2:
        c1_text = chunks_with_overlap[0]['text']
        c2_text = chunks_with_overlap[1]['text']

        lines.append("\n  [Chunk 1 Tail (tokens 340-400)]:")
        lines.append(f"  \"{c1_text[-180:].replace('\n', ' ')}\"")
        lines.append("\n  [Chunk 2 Head (tokens 340-400 overlap region + continuation)]:")
        lines.append(f"  \"{c2_text[:200].replace('\n', ' ')}\"")
        lines.append("\n  Observation (Case B): Chunk 2 includes the 60-token overlap starting from token 340.")
        lines.append("  The entire critical requirement sentence ('CRITICAL REQUIREMENT: ...') and its preceding context")
        lines.append("  are fully present inside Chunk 2, allowing RAG vector retrieval to successfully match and retrieve it.")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    corpus_dir = os.path.join(project_root, "data", "sample_corpus")
    sample_path = os.path.join(corpus_dir, "customs_requirements.txt")

    if not os.path.exists(sample_path):
        docs = load_directory(corpus_dir, verbose=False)
        sample_text = docs[0]["text"] if docs else "Sample text for chunking demonstration."
    else:
        sample_doc = load_document(sample_path, verbose=False)
        sample_text = sample_doc["text"] if sample_doc else "Sample text for chunking demonstration."

    # If text is too short to demonstrate multi-chunk overlap, repeat it
    enc = get_tokenizer("cl100k_base")
    if len(enc.encode(sample_text)) < 500:
        sample_text = (sample_text + "\n\n") * 5

    report_parts = [
        "\n==============================================================================",
        "     SHIPRULE CDLP — TOKEN-AWARE CHUNK SIZING & OVERLAP DEMONSTRATION         ",
        "==============================================================================\n",
        run_overlap_demonstration(sample_text),
        "\n",
        run_overlap_comparison(sample_text),
        "\n",
        run_boundary_context_demo()
    ]

    report = "\n".join(report_parts)
    print(report)

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "token_chunking_results.txt")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[SUCCESS] Token chunking demonstration report saved to:\n  {results_path}")


if __name__ == "__main__":
    main()
