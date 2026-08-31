"""
ShipRule CDLP - Document Chunking Module
=========================================
Implements reusable Document Chunking Strategies for RAG & AI Retrieval Systems:
1. Fixed-Size Chunking (with configurable size and overlap)
2. Paragraph-Based Chunking (preserves semantic paragraph boundaries)
3. Sentence-Based Chunking (preserves grammatical sentence integrity)

Generates unified chunk objects, statistics, boundary inspection, and comparison reports
ready for the embedding and ChromaDB ingestion pipeline.
"""

import os
import sys
import re
import json
import math
from typing import Dict, List, Any, Optional, Tuple

# Ensure UTF-8 output handling on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.document_loader import load_directory, load_document, load_documents

# Known abbreviations for sentence splitting
COMMON_ABBREVIATIONS = {
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "vs.", "etc.",
    "e.g.", "i.e.", "u.s.", "u.k.", "u.n.", "approx.", "dept.", "est.",
    "fig.", "inc.", "ltd.", "corp.", "no.", "vol.", "jan.", "feb.",
    "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.",
    "nov.", "dec.", "hs.", "code."
}


# ==============================================================================
# 1. CORE CHUNKING STRATEGY FUNCTIONS
# ==============================================================================

def fixed_size_chunking(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """
    Strategy A: Slices text into fixed-size character windows with configurable overlap.

    Args:
        text: Plain-text string to chunk.
        size: Window size in characters (default: 500).
        overlap: Character overlap between consecutive chunks (default: 50).

    Returns:
        List of raw chunk strings.

    Raises:
        ValueError: If size <= 0, overlap < 0, or overlap >= size.
    """
    if size <= 0:
        raise ValueError(f"Chunk size must be a positive integer > 0, got {size}")
    if overlap < 0:
        raise ValueError(f"Overlap cannot be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(f"Overlap ({overlap}) must be strictly less than chunk size ({size})")

    if not text or not text.strip():
        return []

    chunks = []
    i = 0
    text_len = len(text)
    step = size - overlap

    while i < text_len:
        chunk = text[i:i + size]
        if chunk.strip():
            chunks.append(chunk)
        if i + size >= text_len:
            break
        i += step

    return chunks


def paragraph_chunking(text: str) -> List[str]:
    """
    Strategy B: Splits text on natural paragraph boundaries (double newlines).
    Preserves complete paragraphs, removes empty chunks, and retains thematic meaning.

    Args:
        text: Plain-text string to chunk.

    Returns:
        List of paragraph chunk strings.
    """
    if not text or not text.strip():
        return []

    # Normalize line breaks and split on 2 or more newlines
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paras = re.split(r"\n\s*\n+", normalized)

    chunks = []
    for p in raw_paras:
        clean_p = p.strip()
        if clean_p:
            chunks.append(clean_p)

    return chunks


def sentence_chunking(text: str) -> List[str]:
    """
    Strategy C: Splits text on grammatical sentence boundaries without cutting sentences.
    Handles common abbreviations, decimals, and terminal punctuations.

    Args:
        text: Plain-text string to chunk.

    Returns:
        List of sentence chunk strings.
    """
    if not text or not text.strip():
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Match sentence endings followed by whitespace or line break
    pattern = re.compile(r'([.!?]+(?:\s+|\n+|$))')
    matches = list(pattern.finditer(normalized))

    if not matches:
        trimmed = normalized.strip()
        return [trimmed] if trimmed else []

    sentences = []
    start = 0

    for m in matches:
        end = m.end()
        candidate = normalized[start:end]
        
        words = candidate.strip().split()
        last_word = words[-1].lower() if words else ""

        # Avoid splitting on known abbreviations
        if last_word in COMMON_ABBREVIATIONS or re.search(r'\b[A-Za-z]\.$', last_word):
            continue

        # Avoid splitting on numbered lists (e.g. "1. ")
        if re.search(r'^\d+\.$', last_word):
            continue

        cleaned = candidate.strip()
        if cleaned:
            sentences.append(cleaned)
        start = end

    # Collect any remaining trailing characters
    if start < len(normalized):
        trailing = normalized[start:].strip()
        if trailing:
            sentences.append(trailing)

    return sentences


# ==============================================================================
# 2. UNIFIED CHUNK OBJECT & BUILDER
# ==============================================================================

def create_unified_chunk(
    source: str,
    strategy: str,
    chunk_index: int,
    chunk_text: str,
    document_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs a standardized chunk dictionary for the ShipRule RAG pipeline.

    Format:
    {
      "chunk_id": "shipping_rules_001",
      "source": "shipping_rules.txt",
      "document_type": "txt",
      "strategy": "paragraph",
      "chunk_index": 1,
      "character_count": 342,
      "chunk_text": "..."
    }
    """
    stem = os.path.splitext(source)[0]
    # Clean stem for clean chunk_id
    clean_stem = re.sub(r'[^a-zA-Z0-9_]', '_', stem)
    
    if not document_type:
        _, ext = os.path.splitext(source)
        document_type = ext.lstrip(".").lower() or "txt"

    return {
        "chunk_id": f"{clean_stem}_{strategy}_{chunk_index:03d}",
        "source": source,
        "document_type": document_type,
        "strategy": strategy,
        "chunk_index": chunk_index,
        "character_count": len(chunk_text),
        "chunk_text": chunk_text
    }


def chunk_document_by_strategy(
    doc: Dict[str, str],
    strategy: str = "paragraph",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Chunks a single document dict {"source": "...", "text": "..."} using the specified strategy.

    Supported strategies:
        - 'fixed' or 'fixed_size': Fixed-size chunking (kwargs: size=500, overlap=50)
        - 'paragraph': Paragraph-based chunking
        - 'sentence': Sentence-based chunking
    """
    if not doc or not doc.get("text"):
        return []

    source = doc.get("source", "unknown_document")
    text = doc.get("text", "")
    _, ext = os.path.splitext(source)
    doc_type = ext.lstrip(".").lower() or "txt"

    strat_key = strategy.lower().strip()

    if strat_key in ("fixed", "fixed_size"):
        size = kwargs.get("size", 500)
        overlap = kwargs.get("overlap", 50)
        raw_chunks = fixed_size_chunking(text, size=size, overlap=overlap)
        canonical_strat = "fixed"

    elif strat_key == "paragraph":
        raw_chunks = paragraph_chunking(text)
        canonical_strat = "paragraph"

    elif strat_key == "sentence":
        raw_chunks = sentence_chunking(text)
        canonical_strat = "sentence"

    else:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Supported: 'fixed', 'paragraph', 'sentence'")

    unified_chunks = []
    for idx, raw_c in enumerate(raw_chunks, start=1):
        unified_chunks.append(
            create_unified_chunk(
                source=source,
                strategy=canonical_strat,
                chunk_index=idx,
                chunk_text=raw_c,
                document_type=doc_type
            )
        )

    return unified_chunks


# ==============================================================================
# 3. STATISTICS & BOUNDARY INSPECTION
# ==============================================================================

def calculate_chunk_stats(chunks: List[Dict[str, Any]], original_char_count: int) -> Dict[str, Any]:
    """Calculates comprehensive statistics for a set of generated chunks."""
    if not chunks:
        return {
            "original_char_count": original_char_count,
            "total_chunks": 0,
            "avg_size": 0.0,
            "min_size": 0,
            "max_size": 0
        }

    sizes = [c["character_count"] for c in chunks]
    return {
        "original_char_count": original_char_count,
        "total_chunks": len(chunks),
        "avg_size": round(sum(sizes) / len(sizes), 1),
        "min_size": min(sizes),
        "max_size": max(sizes)
    }


def inspect_boundaries(chunks: List[Dict[str, Any]], max_samples: int = 2) -> List[Dict[str, Any]]:
    """
    Inspects representative boundary transitions between adjacent chunks to analyze
    whether the boundary breaks a sentence, a paragraph, or a complete idea.
    """
    inspections = []
    if len(chunks) < 2:
        return inspections

    for i in range(min(max_samples, len(chunks) - 1)):
        c1 = chunks[i]["chunk_text"]
        c2 = chunks[i + 1]["chunk_text"]

        c1_tail = c1[-70:].strip().replace("\n", " ") if len(c1) > 70 else c1.strip().replace("\n", " ")
        c2_head = c2[:70].strip().replace("\n", " ") if len(c2) > 70 else c2.strip().replace("\n", " ")

        # Analyze boundary continuity
        breaks_sentence = not bool(re.search(r'[.!?]$', c1.strip()))
        breaks_paragraph = True
        breaks_complete_idea = breaks_sentence

        analysis = "Clean boundary (preserves complete sentence/paragraph idea)."
        if breaks_sentence:
            analysis = "Breaks mid-sentence/mid-clause. Retrieval query may match partial keywords without full context."

        inspections.append({
            "chunk_pair": f"Chunk {chunks[i]['chunk_index']} ➔ Chunk {chunks[i+1]['chunk_index']}",
            "chunk_1_tail": f"...{c1_tail}",
            "chunk_2_head": f"{c2_head}...",
            "breaks_sentence": breaks_sentence,
            "boundary_assessment": analysis
        })

    return inspections


# ==============================================================================
# 4. RECOMMENDATION ENGINE FOR SHIPRULE
# ==============================================================================

def recommend_best_strategy(docs: List[Dict[str, str]], all_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates the actual ShipRule corpus to determine and recommend the optimal chunking strategy.
    """
    # Check paragraph structure of the corpus
    total_docs = len(docs)
    structured_policy_count = 0
    total_paragraphs = 0

    for d in docs:
        text = d.get("text", "")
        paras = paragraph_chunking(text)
        total_paragraphs += len(paras)
        if len(paras) >= 2:
            structured_policy_count += 1

    ratio_structured = structured_policy_count / max(1, total_docs)

    if ratio_structured >= 0.6:
        recommended = "Paragraph-Based Chunking"
        rec_key = "paragraph"
        reason = (
            "Paragraph-based chunking is recommended for the current ShipRule corpus because "
            "shipping and customs documents contain structured policy sections where preserving "
            "complete paragraphs helps maintain the context required for retrieval."
        )
    else:
        recommended = "Fixed-Size Chunking (with Overlap)"
        rec_key = "fixed"
        reason = (
            "Fixed-size chunking is recommended because the corpus is predominantly unstructured or dense text."
        )

    return {
        "recommended_strategy": recommended,
        "strategy_key": rec_key,
        "reason": reason,
        "corpus_documents_evaluated": total_docs,
        "total_paragraphs_detected": total_paragraphs
    }


# ==============================================================================
# 5. FULL PIPELINE EXECUTION & CLI
# ==============================================================================

def run_chunking_pipeline(
    corpus_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Loads all documents from data/sample_corpus/, runs all 3 chunking strategies,
    computes comparative statistics, performs boundary inspection, saves JSON outputs,
    and returns a consolidated report dictionary.
    """
    if corpus_dir is None:
        corpus_dir = os.path.join(project_root, "data", "sample_corpus")
    if output_dir is None:
        output_dir = os.path.join(project_root, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("============================================================", flush=True)
        print("SHIPRULE - DOCUMENT CHUNKING STRATEGIES", flush=True)
        print("============================================================\n", flush=True)

    # 1. Load documents using the existing Document Loader
    docs = load_directory(corpus_dir, verbose=False)
    if not docs:
        if verbose:
            print(f"[WARNING] No documents found in {corpus_dir}")
        return {}

    # Sort docs predictably
    docs = sorted(docs, key=lambda x: x.get("source", ""))

    all_chunks_by_strat: Dict[str, List[Dict[str, Any]]] = {
        "fixed": [],
        "paragraph": [],
        "sentence": []
    }

    per_doc_stats: Dict[str, Dict[str, Any]] = {}

    for doc in docs:
        doc_source = doc["source"]
        doc_chars = len(doc["text"])
        per_doc_stats[doc_source] = {}

        # Strategy A: Fixed-size
        fixed_chunks = chunk_document_by_strategy(doc, strategy="fixed", size=500, overlap=50)
        all_chunks_by_strat["fixed"].extend(fixed_chunks)
        per_doc_stats[doc_source]["fixed"] = calculate_chunk_stats(fixed_chunks, doc_chars)

        # Strategy B: Paragraph
        para_chunks = chunk_document_by_strategy(doc, strategy="paragraph")
        all_chunks_by_strat["paragraph"].extend(para_chunks)
        per_doc_stats[doc_source]["paragraph"] = calculate_chunk_stats(para_chunks, doc_chars)

        # Strategy C: Sentence
        sent_chunks = chunk_document_by_strategy(doc, strategy="sentence")
        all_chunks_by_strat["sentence"].extend(sent_chunks)
        per_doc_stats[doc_source]["sentence"] = calculate_chunk_stats(sent_chunks, doc_chars)

        if verbose:
            print(f"Processing: {doc_source}")
            print(f"Original Size: {doc_chars:,} characters\n")
            print(f"{'Strategy':<16}  {'Chunks':>6}    {'Avg Size':>8}    {'Min Size':>8}    {'Max Size':>8}")
            print("-" * 62)
            print(f"{'Fixed-Size':<16}  {per_doc_stats[doc_source]['fixed']['total_chunks']:>6}    {per_doc_stats[doc_source]['fixed']['avg_size']:>8.1f}    {per_doc_stats[doc_source]['fixed']['min_size']:>8}    {per_doc_stats[doc_source]['fixed']['max_size']:>8}")
            print(f"{'Paragraph':<16}  {per_doc_stats[doc_source]['paragraph']['total_chunks']:>6}    {per_doc_stats[doc_source]['paragraph']['avg_size']:>8.1f}    {per_doc_stats[doc_source]['paragraph']['min_size']:>8}    {per_doc_stats[doc_source]['paragraph']['max_size']:>8}")
            print(f"{'Sentence':<16}  {per_doc_stats[doc_source]['sentence']['total_chunks']:>6}    {per_doc_stats[doc_source]['sentence']['avg_size']:>8.1f}    {per_doc_stats[doc_source]['sentence']['min_size']:>8}    {per_doc_stats[doc_source]['sentence']['max_size']:>8}")
            print("-" * 62 + "\n")

    # 3. Overall Corpus Comparison
    total_doc_chars = sum(len(d["text"]) for d in docs)
    corpus_stats = {}
    for strat_key, chunk_list in all_chunks_by_strat.items():
        corpus_stats[strat_key] = calculate_chunk_stats(chunk_list, total_doc_chars)

    if verbose:
        print("============================================================")
        print("OVERALL CORPUS COMPARISON TABLE")
        print("============================================================\n")
        print(f"{'Strategy':<16} | {'Chunk Count':>11} | {'Avg Size':>8} | {'Context Preservation':<20} | {'Main Advantage':<28} | {'Main Limitation'}")
        print("-" * 125)
        print(f"{'Fixed-Size':<16} | {corpus_stats['fixed']['total_chunks']:>11} | {corpus_stats['fixed']['avg_size']:>8.1f} | {'Low - Moderate':<20} | {'Predictable chunk bounds':<28} | {'May split sentences/ideas'}")
        print(f"{'Paragraph-Based':<16} | {corpus_stats['paragraph']['total_chunks']:>11} | {corpus_stats['paragraph']['avg_size']:>8.1f} | {'High':<20} | {'Preserves complete sections':<28} | {'Uneven chunk sizes'}")
        print(f"{'Sentence-Based':<16} | {corpus_stats['sentence']['total_chunks']:>11} | {corpus_stats['sentence']['avg_size']:>8.1f} | {'Moderate - High':<20} | {'Preserves clause syntax':<28} | {'Creates many small chunks'}")
        print("-" * 125 + "\n")

    # 4. Boundary Inspection
    boundary_inspections = {
        "fixed": inspect_boundaries(all_chunks_by_strat["fixed"]),
        "paragraph": inspect_boundaries(all_chunks_by_strat["paragraph"]),
        "sentence": inspect_boundaries(all_chunks_by_strat["sentence"])
    }

    if verbose:
        print("============================================================")
        print("BOUNDARY INSPECTION & RETRIEVAL IMPACT")
        print("============================================================\n")
        for s_key, s_name in [("fixed", "Fixed-Size"), ("paragraph", "Paragraph-Based"), ("sentence", "Sentence-Based")]:
            print(f"Strategy: {s_name}")
            insps = boundary_inspections.get(s_key, [])
            if insps:
                for insp in insps:
                    print(f"  {insp['chunk_pair']}:")
                    print(f"    Tail: \"{insp['chunk_1_tail']}\"")
                    print(f"    Head: \"{insp['chunk_2_head']}\"")
                    print(f"    Analysis: {insp['boundary_assessment']}\n")
            else:
                print("  No boundary cuts detected.\n")

    # 5. Recommendation
    rec = recommend_best_strategy(docs, per_doc_stats)
    if verbose:
        print("============================================================")
        print("RECOMMENDATION FOR SHIPRULE CORPUS")
        print("============================================================")
        print(f"Recommended Strategy: {rec['recommended_strategy']}")
        print(f"Rationale: {rec['reason']}\n")
        print("============================================================\n")

    # 6. Save JSON Outputs
    fixed_path = os.path.join(output_dir, "chunks_fixed.json")
    para_path = os.path.join(output_dir, "chunks_paragraph.json")
    sent_path = os.path.join(output_dir, "chunks_sentence.json")
    report_path = os.path.join(output_dir, "chunking_report.json")

    with open(fixed_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks_by_strat["fixed"], f, indent=2, ensure_ascii=False)

    with open(para_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks_by_strat["paragraph"], f, indent=2, ensure_ascii=False)

    with open(sent_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks_by_strat["sentence"], f, indent=2, ensure_ascii=False)

    final_report = {
        "corpus_directory": corpus_dir,
        "documents_processed": [d["source"] for d in docs],
        "per_document_statistics": per_doc_stats,
        "overall_corpus_statistics": corpus_stats,
        "boundary_inspections": boundary_inspections,
        "recommendation": rec,
        "output_files": {
            "fixed_size_chunks": "outputs/chunks_fixed.json",
            "paragraph_chunks": "outputs/chunks_paragraph.json",
            "sentence_chunks": "outputs/chunks_sentence.json"
        }
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"[OUTPUT] Generated chunks and reports saved to:")
        print(f"  • {fixed_path}")
        print(f"  • {para_path}")
        print(f"  • {sent_path}")
        print(f"  • {report_path}\n")

    return final_report



if __name__ == "__main__":
    run_chunking_pipeline()
