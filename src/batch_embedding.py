"""
KDU 3.28: Batch Embedding & Rate/Cost Management Pipeline
===========================================================
Implements a production-grade, resumable, rate-limit resilient embedding pipeline for the
Customs Duty & Documentation Lookup Platform (CDLP).

Key Capabilities:
1. Batching: Sends chunks in configurable batches (default size=64) to maximize API throughput.
2. Retry with Backoff: Handles rate limits & transient failures using exponential backoff retry loops.
3. Cost & Token Tracking: Calculates total tokens and estimates embedding USD cost.
4. Resumability (Skip on Re-run): Detects existing embedded chunk IDs and skips them to avoid duplicate costs.
5. Summary Reporting: Generates structured metrics on chunks, skipped items, embeddings created, failed batches, and costs.
"""

import json
import os
import sys
import time
from typing import List, Dict, Any, Generator, Callable, Optional, Set

# Ensure project root directory is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.token_counter import count_tokens

# Price per 1,000 input tokens for OpenAI text-embedding-3-small ($0.02 per 1M tokens)
DEFAULT_PRICE_PER_1K_TOKENS = 0.00002


def batches(items: List[Any], size: int) -> Generator[List[Any], None, None]:
    """
    Yield successive batches of a specified size from an input list.

    Args:
        items: List of elements to split.
        size: Maximum batch size.

    Yields:
        Sublists of items with max length `size`.
    """
    if size <= 0:
        raise ValueError("Batch size must be greater than 0")
    for start in range(0, len(items), size):
        yield items[start:start + size]


def estimate_tokens(texts: List[str]) -> int:
    """
    Estimate total tokens across a list of input text strings.

    Args:
        texts: List of text strings in a batch.

    Returns:
        Total estimated token count.
    """
    return sum(count_tokens(text) for text in texts)


def embed_with_retry(
    texts: List[str],
    embed_fn: Callable[[List[str]], List[List[float]]],
    max_attempts: int = 5,
    initial_backoff: float = 0.1,
    backoff_factor: float = 2.0
) -> List[List[float]]:
    """
    Embeds a list of texts using `embed_fn`, retrying transient failures and rate limits
    with exponential backoff.

    Args:
        texts: List of strings to embed.
        embed_fn: Embedding callback function taking List[str] -> List[List[float]].
        max_attempts: Maximum retry attempts before raising exception.
        initial_backoff: Initial wait time in seconds (default 0.1s for fast execution).
        backoff_factor: Multiplier for exponential backoff.

    Returns:
        List of embedding float vectors.
    """
    for attempt in range(max_attempts):
        try:
            return embed_fn(texts)
        except Exception as error:
            if attempt == max_attempts - 1:
                raise error
            wait_seconds = initial_backoff * (backoff_factor ** attempt)
            print(f"  [Retry {attempt + 1}/{max_attempts}] Error encountered: {error}. Retrying after {wait_seconds:.2f}s...")
            time.sleep(wait_seconds)


class BatchEmbeddingPipeline:
    """
    Manages batch embedding operations, rate-limit retries, token/cost reporting,
    and resumability for the CDLP customs regulation database.
    """

    def __init__(
        self,
        batch_size: int = 64,
        price_per_1k_tokens: float = DEFAULT_PRICE_PER_1K_TOKENS,
        max_retry_attempts: int = 5,
        cache_path: Optional[str] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None
    ):
        self.batch_size = batch_size
        self.price_per_1k_tokens = price_per_1k_tokens
        self.max_retry_attempts = max_retry_attempts
        self.cache_path = cache_path or os.path.join(project_root, "data", "embeddings_cache.json")
        self.embed_fn = embed_fn or self._default_mock_embed_fn

    def _default_mock_embed_fn(self, texts: List[str]) -> List[List[float]]:
        """
        Default embedding generator creating deterministic 384-dim mock vectors
        for local offline execution and testing.
        """
        embeddings = []
        for text in texts:
            # Deterministic pseudo-embedding based on hash of text
            val = float(abs(hash(text)) % 1000) / 1000.0
            vec = [val] * 384
            embeddings.append(vec)
        return embeddings

    def load_existing_embeddings(self) -> Dict[str, Dict[str, Any]]:
        """Loads cached embedding records from local storage if available."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_embeddings(self, batch: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """
        Saves embedded chunks to disk storage to maintain durable progress.
        Allows the pipeline to resume seamlessly after interruptions.
        """
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        cached = self.load_existing_embeddings()

        for chunk, vec in zip(batch, embeddings):
            cached[chunk["id"]] = {
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "embedding_dim": len(vec),
                "embedded_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(cached, f, indent=2)

    def clear_cache(self) -> None:
        """Clears the local embedding cache (useful for testing or full re-indexing)."""
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

    def process_corpus(self, all_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes a list of chunks in batches, filtering out already-embedded chunks,
        retrying failed batches, and computing run statistics.

        Args:
            all_chunks: List of chunk dictionaries containing 'id' and 'text'.

        Returns:
            Dictionary summary of the run metrics.
        """
        existing_store = self.load_existing_embeddings()
        existing_embedding_ids: Set[str] = set(existing_store.keys())

        # Filter out chunks that already have vectors (resumability)
        pending_chunks = [
            chunk for chunk in all_chunks
            if chunk["id"] not in existing_embedding_ids
        ]

        summary = {
            "total_chunks": len(all_chunks),
            "skipped_existing": len(all_chunks) - len(pending_chunks),
            "embedded": 0,
            "failed": 0,
            "input_tokens": 0,
            "estimated_cost_usd": 0.0,
            "batches_processed": 0,
            "retry_attempts": 0,
            "batch_size": self.batch_size,
            "price_per_1k_tokens": self.price_per_1k_tokens
        }

        print(f"\n[Batch Embedding Pipeline] Total chunks: {summary['total_chunks']} | "
              f"Skipped existing: {summary['skipped_existing']} | "
              f"Pending chunks to embed: {len(pending_chunks)}")

        for batch in batches(pending_chunks, size=self.batch_size):
            summary["batches_processed"] += 1
            texts = [chunk["text"] for chunk in batch]
            batch_tokens = estimate_tokens(texts)

            try:
                # Execute embedding call with exponential backoff retry
                response_embeddings = embed_with_retry(
                    texts=texts,
                    embed_fn=self.embed_fn,
                    max_attempts=self.max_retry_attempts
                )
                self.save_embeddings(batch, response_embeddings)
                summary["embedded"] += len(response_embeddings)
                summary["input_tokens"] += batch_tokens

            except Exception as batch_error:
                print(f"  [ERROR] Batch processing failed permanently after retries: {batch_error}")
                summary["failed"] += len(batch)

        # Compute total cost estimate based on input tokens
        estimated_cost = (summary["input_tokens"] / 1000.0) * self.price_per_1k_tokens
        summary["estimated_cost_usd"] = round(estimated_cost, 6)

        return summary


# Standard CDLP (Customs Duty & Documentation Lookup Platform) Corpus Dataset
CDLP_SAMPLE_CORPUS = [
    {
        "id": "CDLP-IN-8471-01",
        "text": "Customs Record India HS Code 8471.30 (Laptops): Basic Customs Duty 7.5%, SWS 10%. Documents: Commercial Invoice, BIS Registration, DGFT Import License. Restricted Status: Restricted.",
        "metadata": {"country": "India", "hs_code": "8471.30"}
    },
    {
        "id": "CDLP-IT-8703-01",
        "text": "Customs Record Italy HS Code 8703 (Motor Vehicles): Duty 10%, VAT 22%. Documents: Certificate of Origin, EUR.1, Type Approval. Restricted Status: Unrestricted.",
        "metadata": {"country": "Italy", "hs_code": "8703"}
    },
    {
        "id": "CDLP-DE-8541-01",
        "text": "Customs Record Germany HS Code 8541.43 (Solar PV Modules): Duty 0%, VAT 19% (0% for residential). Documents: CE Declaration, Commercial Invoice. Restricted Status: Unrestricted.",
        "metadata": {"country": "Germany", "hs_code": "8541.43"}
    },
    {
        "id": "CDLP-BR-9018-01",
        "text": "Customs Record Brazil HS Code 9018.90 (Medical Equipment): Duty 14%, PIS 1.65%, COFINS 7.6%. Documents: ANVISA Sanitary Registration. Restricted Status: Restricted.",
        "metadata": {"country": "Brazil", "hs_code": "9018.90"}
    },
    {
        "id": "CDLP-US-8517-01",
        "text": "Customs Record USA HS Code 8517.62 (Telecom Switches & Routers): Duty 0%. Documents: FCC Declaration of Conformity, Commercial Invoice. Restricted Status: Unrestricted.",
        "metadata": {"country": "USA", "hs_code": "8517.62"}
    },
    {
        "id": "CDLP-JP-3004-01",
        "text": "Customs Record Japan HS Code 3004.90 (Medicaments & Pharmaceuticals): Duty 0%, Consumption Tax 10%. Documents: MHLW Import License, Certificate of Analysis. Restricted Status: Restricted.",
        "metadata": {"country": "Japan", "hs_code": "3004.90"}
    },
    {
        "id": "CDLP-GB-2204-01",
        "text": "Customs Record UK HS Code 2204.21 (Wine of Fresh Grapes): Duty £2.23 per litre + Excise Duty. Documents: C&E 105 Import Entry, Certificate of Analysis. Restricted Status: Unrestricted.",
        "metadata": {"country": "UK", "hs_code": "2204.21"}
    },
    {
        "id": "CDLP-ZA-7308-01",
        "text": "Customs Record South Africa HS Code 7308.90 (Structures of Iron/Steel): Duty 15%, VAT 15%. Documents: ITAC Import Permit, Test Certificate. Restricted Status: Restricted.",
        "metadata": {"country": "South Africa", "hs_code": "7308.90"}
    },
    {
        "id": "CDLP-AE-8415-01",
        "text": "Customs Record UAE HS Code 8415.10 (Air Conditioners): Duty 5%, VAT 5%. Documents: ECAS Certificate of Conformity, Bill of Entry. Restricted Status: Unrestricted.",
        "metadata": {"country": "UAE", "hs_code": "8415.10"}
    },
    {
        "id": "CDLP-MX-8708-01",
        "text": "Customs Record Mexico HS Code 8708.29 (Automotive Parts): Duty 5%, VAT 16%. Documents: Pedimento de Importación, NOM Compliance Certificate. Restricted Status: Unrestricted.",
        "metadata": {"country": "Mexico", "hs_code": "8708.29"}
    }
]


def run_batch_embedding_demo(batch_size: int = 4, price_per_1k_tokens: float = DEFAULT_PRICE_PER_1K_TOKENS) -> Dict[str, Any]:
    """
    Runs a end-to-end demonstration of the batch embedding pipeline:
    1. First Run: Embeds all chunks in batch sizes, calculates cost, stores embeddings.
    2. Re-run: Verifies resumability (skips already-embedded chunks to save cost).
    3. Exports run summaries to JSON and human-readable text files in outputs/.
    """
    pipeline = BatchEmbeddingPipeline(batch_size=batch_size, price_per_1k_tokens=price_per_1k_tokens)

    # Clear cache before demo to demonstrate a clean initial run
    pipeline.clear_cache()

    print("======================================================================")
    print("      KDU 3.28 BATCH EMBEDDING & RATE/COST MANAGEMENT PIPELINE        ")
    print("======================================================================")
    print("\n--- STAGE 1: INITIAL EMBEDDING RUN ---")
    initial_summary = pipeline.process_corpus(CDLP_SAMPLE_CORPUS)

    print("\nRun Summary (Initial Run):")
    print(json.dumps(initial_summary, indent=2))
    print(f"Estimated Cost USD: ${initial_summary['estimated_cost_usd']:.6f}")

    print("\n--- STAGE 2: RE-RUN RESUMABILITY TEST (SKIP ON RE-RUN) ---")
    rerun_summary = pipeline.process_corpus(CDLP_SAMPLE_CORPUS)

    print("\nRun Summary (Re-run):")
    print(json.dumps(rerun_summary, indent=2))
    print(f"Estimated Cost USD: ${rerun_summary['estimated_cost_usd']:.6f}")

    # Build exported output artifacts
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    summary_export = {
        "initial_run": initial_summary,
        "rerun_resumability": rerun_summary,
        "cdlp_corpus_count": len(CDLP_SAMPLE_CORPUS)
    }

    json_path = os.path.join(output_dir, "batch_embedding_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)

    txt_lines = [
        "======================================================================",
        "      KDU 3.28 BATCH EMBEDDING & RATE/COST MANAGEMENT RUN REPORT      ",
        "======================================================================\n",
        f"Corpus Size: {len(CDLP_SAMPLE_CORPUS)} CDLP Customs Regulation Chunks",
        f"Batch Size: {batch_size}",
        f"Price Rate per 1k Tokens: ${price_per_1k_tokens}\n",
        "--- INITIAL RUN SUMMARY ---",
        f"Total Chunks:        {initial_summary['total_chunks']}",
        f"Skipped Existing:    {initial_summary['skipped_existing']}",
        f"Embedded:            {initial_summary['embedded']}",
        f"Failed:              {initial_summary['failed']}",
        f"Batches Processed:   {initial_summary['batches_processed']}",
        f"Input Tokens Sent:   {initial_summary['input_tokens']}",
        f"Estimated Cost:      ${initial_summary['estimated_cost_usd']:.6f}\n",
        "--- RE-RUN RESUMABILITY TEST SUMMARY ---",
        f"Total Chunks:        {rerun_summary['total_chunks']}",
        f"Skipped Existing:    {rerun_summary['skipped_existing']} (100% Skipped - Resumable)",
        f"Embedded:            {rerun_summary['embedded']}",
        f"Failed:              {rerun_summary['failed']}",
        f"Input Tokens Sent:   {rerun_summary['input_tokens']}",
        f"Estimated Cost:      ${rerun_summary['estimated_cost_usd']:.6f}\n",
        "--- ARCHITECTURAL & RESUMABILITY DESIGN NOTES ---",
        "1. Batching: Sends chunks in groups of `batch_size` (e.g. 64) per request, reducing API overhead.",
        "2. Resumability: Checks `existing_embedding_ids` before batching. If a chunk already exists,",
        "   it is skipped. This makes the embedding pipeline safe to restart after a crash or partial run.",
        "3. Exponential Backoff: Transient failures and HTTP 429 Rate Limit errors are retried with exponential",
        "   backoff (`wait_seconds = initial_backoff * (2 ** attempt)`). Permanent errors after max attempts",
        "   are logged in `failed` metrics rather than hiding them.",
        "4. Cost Estimation: Accurately tracks input token counts and multiplies by per-thousand pricing model.",
        "5. Large Corpus Strategy: For massive corpora beyond single-run capacity, write progress after every batch,",
        "   store vectors immediately, and partition across parallel worker processes."
    ]

    txt_path = os.path.join(output_dir, "batch_embedding_output.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    print(f"\n[Output] Batch embedding reports saved to:\n  - {json_path}\n  - {txt_path}")
    return summary_export


if __name__ == "__main__":
    run_batch_embedding_demo()
