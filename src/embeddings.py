"""
ShipRule CDLP - Generating Embeddings via API
==============================================
Reads validated document chunks from the ingestion pipeline output,
generates numerical vector embeddings using an OpenAI-compatible API
or local embedding engine (when configured with Groq credentials), validates
vector dimensions, and saves full and sample embedding artifacts to outputs/.
"""

import os
import sys
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

# Ensure UTF-8 output handling on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
from openai import OpenAI


# ==============================================================================
# 1. LOCAL EMBEDDING ADAPTER (FOR GROQ / OFFLINE ENVIRONMENTS)
# ==============================================================================

class LocalEmbeddingClient:
    """
    OpenAI-compatible embedding adapter using ChromaDB's ONNX embedding engine.
    Used when Groq keys are configured (since Groq only hosts LLMs and not embedding models)
    or for local zero-cost embedding execution.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        import chromadb.utils.embedding_functions as ef
        self._fn = ef.DefaultEmbeddingFunction()

    class _EmbeddingsResource:
        def __init__(self, parent):
            self.parent = parent

        def create(self, input: Any, model: str):
            texts = [input] if isinstance(input, str) else list(input)
            raw_vectors = self.parent._fn(texts)

            class _DataItem:
                def __init__(self, embedding_list):
                    self.embedding = [float(x) for x in embedding_list]

            class _Response:
                def __init__(self, items):
                    self.data = items

            data_items = [_DataItem(v) for v in raw_vectors]
            return _Response(data_items)

    @property
    def embeddings(self):
        return LocalEmbeddingClient._EmbeddingsResource(self)


# ==============================================================================
# 2. LOAD PREPARED / VALIDATED CHUNKS
# ==============================================================================

def load_validated_chunks(
    chunks_file: Optional[str] = None,
    max_chunks: Optional[int] = 5
) -> List[Dict[str, Any]]:
    """
    Reads prepared chunks from outputs/processed_chunks.json.
    If the file does not exist, triggers the ingestion pipeline to generate it.
    Limits the number of chunks to max_chunks (if specified) to avoid unnecessary cost.
    """
    if not chunks_file:
        chunks_file = os.path.join(project_root, "outputs", "processed_chunks.json")

    # If processed chunks don't exist yet, run ingestion pipeline automatically
    if not os.path.exists(chunks_file):
        from src.ingestion import run_ingestion_pipeline
        run_ingestion_pipeline(verbose=False)

    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"Processed chunks file not found at '{chunks_file}'")

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError(f"Invalid format in '{chunks_file}': expected list of chunk objects.")

    if max_chunks is not None and max_chunks > 0:
        return chunks[:max_chunks]

    return chunks


# ==============================================================================
# 3. PROVIDER CONFIGURATION & CLIENT INITIALIZATION
# ==============================================================================

def validate_embedding_provider_config(
    base_url: str,
    model: str,
    api_key: Optional[str] = None
) -> str:
    """
    Validates provider configuration and prints a safe configuration summary.
    Fails early if proprietary OpenAI embedding models are configured against Groq endpoints.
    """
    base_url_clean = (base_url or "").strip().lower()
    model_clean = (model or "").strip().lower()

    # Guard: Groq OpenAI-compatible endpoint does NOT support proprietary OpenAI embedding models
    if "groq.com" in base_url_clean and ("text-embedding-" in model_clean or model_clean == "text-embedding-3-small"):
        raise ValueError(
            f"Configuration Error:\n"
            f"{model} is configured with the Groq endpoint ({base_url}).\n"
            f"Groq provides LLM chat completion but does not host OpenAI proprietary embedding models.\n"
            f"Please configure a provider that supports this embedding model or update EMBEDDING_BASE_URL and EMBEDDING_API_KEY in your .env file."
        )

    has_key = bool(api_key and api_key.strip())
    is_groq_key = has_key and api_key.strip().startswith("gsk_")
    provider_type = "Local ONNX Engine (Groq compatibility mode)" if is_groq_key else "OpenAI-compatible API"

    return (
        f"Embedding Provider Configuration:\n"
        f"  Provider: {provider_type}\n"
        f"  Base URL: {base_url}\n"
        f"  Model   : {model}\n"
        f"  API Key : {'configured' if has_key else 'missing'}"
    )


def create_embedding_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> Any:
    """
    Initializes an embedding client.
    - Uses EMBEDDING_API_KEY and EMBEDDING_BASE_URL (does not implicitly use GROQ_API_KEY).
    - If a valid OpenAI API key (sk-...) is provided, connects to the OpenAI API endpoint.
    - If EMBEDDING_API_KEY is configured with a Groq key (gsk_...), uses the local ChromaDB ONNX
      embedding engine to avoid 401/404 errors since Groq does not host embedding models.
    """
    load_dotenv()

    raw_api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    raw_base_url = base_url or os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    resolved_api_key = raw_api_key.strip().strip("\"'") if raw_api_key else ""
    resolved_base_url = raw_base_url.strip().strip("\"'") if raw_base_url else "https://api.openai.com/v1"

    if not resolved_api_key:
        raise ValueError(
            "Embedding configuration error:\n"
            "EMBEDDING_API_KEY is missing.\n\n"
            "Please configure a separate embedding provider in your .env file:\n"
            "  EMBEDDING_API_KEY=your_key_here\n"
            "  EMBEDDING_BASE_URL=https://api.openai.com/v1\n"
            "  EMBED_MODEL=text-embedding-3-small"
        )

    # If key is a Groq key (explicitly configured in EMBEDDING_API_KEY), use local ONNX embedding client
    if resolved_api_key.startswith("gsk_"):
        return LocalEmbeddingClient(model_name="text-embedding-3-small")

    return OpenAI(
        api_key=resolved_api_key,
        base_url=resolved_base_url
    )



def generate_embedding(
    client: Any,
    text: str,
    model: str
) -> List[float]:
    """
    Calls the embeddings endpoint or local adapter for a given text string.
    Returns a list of float values representing the embedding vector.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty or whitespace-only text.")

    response = client.embeddings.create(
        input=text,
        model=model
    )

    if not response or not hasattr(response, "data") or not response.data:
        raise ValueError("Empty or malformed response returned from embeddings API.")

    embedding = response.data[0].embedding
    if not isinstance(embedding, (list, tuple)) and not hasattr(embedding, "__iter__"):
        raise ValueError("Embeddings endpoint returned an invalid vector format.")

    # Convert to list of floats
    vector = [float(x) for x in embedding]
    if len(vector) == 0:
        raise ValueError("Embeddings endpoint returned an empty vector.")

    return vector


# ==============================================================================
# 4. CHUNK EMBEDDING PROCESSOR WITH FAILURE ISOLATION
# ==============================================================================

def embed_chunks(
    chunks: List[Dict[str, Any]],
    client: Any,
    model: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Iterates through chunk objects, generates an embedding for each chunk's text,
    attaches vector and metadata, and isolates failures without crashing the pipeline.
    """
    embedded_chunks: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", f"chunk_{idx:03d}")
        source = chunk.get("source", "unknown")
        doc_type = chunk.get("document_type", "txt")
        strategy = chunk.get("strategy", "paragraph")
        chunk_index = chunk.get("chunk_index", idx)
        text = chunk.get("chunk_text", "")
        char_count = chunk.get("character_count", len(text))

        if not text or not str(text).strip():
            failures.append({
                "chunk_id": chunk_id,
                "source": source,
                "status": "FAILED",
                "error_message": "Empty chunk text"
            })
            continue

        try:
            vector = generate_embedding(client, text, model=model)
            vector_dim = len(vector)

            embedded_item = {
                "embedding_id": f"embedding_{idx:03d}",
                "chunk_id": chunk_id,
                "source": source,
                "document_type": doc_type,
                "strategy": strategy,
                "chunk_index": chunk_index,
                "character_count": char_count,
                "chunk_text": text,
                "embedding_model": model,
                "vector_dimension": vector_dim,
                "embedding": vector
            }
            embedded_chunks.append(embedded_item)

        except Exception as e:
            failures.append({
                "chunk_id": chunk_id,
                "source": source,
                "status": "FAILED",
                "error_message": str(e)
            })

    return embedded_chunks, failures


# ==============================================================================
# 5. EMBEDDING VALIDATION & PREVIEW FORMATTERS
# ==============================================================================

REQUIRED_EMBEDDING_FIELDS = {
    "embedding_id",
    "chunk_id",
    "source",
    "document_type",
    "strategy",
    "chunk_index",
    "character_count",
    "chunk_text",
    "embedding_model",
    "vector_dimension",
    "embedding"
}


def validate_embeddings(
    embedded_chunks: List[Dict[str, Any]]
) -> Tuple[bool, Optional[int], List[str]]:
    """
    Validates that:
    1. Every embedded chunk has complete metadata and valid vector.
    2. Every vector is non-empty and contains only numeric values.
    3. All embeddings in the collection have consistent dimensions.
    4. Reported vector_dimension matches actual len(embedding).
    """
    if not embedded_chunks:
        return False, None, ["No embedded chunks to validate."]

    errors: List[str] = []
    expected_dim: Optional[int] = None

    for i, item in enumerate(embedded_chunks):
        c_id = item.get("chunk_id", f"index_{i}")

        # Check required fields
        missing_keys = REQUIRED_EMBEDDING_FIELDS - set(item.keys())
        if missing_keys:
            errors.append(f"Chunk '{c_id}' is missing required fields: {missing_keys}")
            continue

        # Check vector validity
        vector = item.get("embedding")
        if not isinstance(vector, list) or len(vector) == 0:
            errors.append(f"Chunk '{c_id}' contains empty or non-list embedding.")
            continue

        if not all(isinstance(x, (int, float)) for x in vector):
            errors.append(f"Chunk '{c_id}' contains non-numerical values in embedding.")
            continue

        v_dim = len(vector)
        reported_dim = item.get("vector_dimension")

        if reported_dim != v_dim:
            errors.append(
                f"Chunk '{c_id}' reported dimension {reported_dim} does not match vector length {v_dim}."
            )

        if expected_dim is None:
            expected_dim = v_dim
        elif v_dim != expected_dim:
            errors.append(
                f"Inconsistent dimension on chunk '{c_id}': expected {expected_dim}, got {v_dim}."
            )

        # Check text integrity
        text = item.get("chunk_text", "")
        if not text or not str(text).strip():
            errors.append(f"Chunk '{c_id}' contains empty chunk_text.")

    is_valid = len(errors) == 0
    return is_valid, expected_dim, errors


def format_vector_preview(vector: List[float], max_elements: int = 3) -> str:
    """Formats a vector for human-readable console preview, e.g. [0.0123, -0.0456, 0.0789, ...]"""
    if not vector:
        return "[]"
    elements = [f"{x:.4f}" for x in vector[:max_elements]]
    if len(vector) > max_elements:
        return f"[{', '.join(elements)}, ...]"
    return f"[{', '.join(elements)}]"


def create_sample_embedding_output(
    embedded_chunks: List[Dict[str, Any]],
    sample_size: int = 2
) -> List[Dict[str, Any]]:
    """
    Creates a lightweight sample representation suitable for git committing
    containing metadata, chunk text, vector dimension, and trimmed vector preview.
    """
    samples = []
    for item in embedded_chunks[:sample_size]:
        vector = item.get("embedding", [])
        trimmed_preview = [round(float(x), 4) for x in vector[:3]]
        samples.append({
            "embedding_id": item.get("embedding_id"),
            "chunk_id": item.get("chunk_id"),
            "source": item.get("source"),
            "document_type": item.get("document_type"),
            "strategy": item.get("strategy"),
            "chunk_index": item.get("chunk_index"),
            "character_count": item.get("character_count"),
            "chunk_text": item.get("chunk_text"),
            "embedding_model": item.get("embedding_model"),
            "vector_length": len(vector),
            "vector_preview": trimmed_preview
        })
    return samples


# ==============================================================================
# 6. ARTIFACT PERSISTENCE & REPORTING
# ==============================================================================

def save_embedding_results(
    embedded_chunks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    model: str,
    selected_count: int,
    output_dir: Optional[str] = None,
    sample_size: int = 2
) -> Dict[str, Any]:
    """
    Persists embedded chunks, analytical report, and trimmed sample output in outputs/.
    """
    out_dir = output_dir or os.path.join(project_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    is_valid, detected_dim, validation_errors = validate_embeddings(embedded_chunks)

    report_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model,
        "statistics": {
            "chunks_selected": selected_count,
            "chunks_successfully_embedded": len(embedded_chunks),
            "failed_embeddings": len(failures),
            "vector_dimension": detected_dim
        },
        "validation": {
            "status": "PASSED" if is_valid else "FAILED",
            "is_dimension_consistent": is_valid and len(validation_errors) == 0,
            "validation_errors": validation_errors
        },
        "output_files": {
            "embedded_chunks": "outputs/embedded_chunks.json",
            "embedding_report": "outputs/embedding_report.json",
            "sample_embedding_output": "outputs/sample_embedding_output.json"
        }
    }

    # 1. outputs/embedded_chunks.json (Full vectors, text, metadata)
    with open(os.path.join(out_dir, "embedded_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)

    # 2. outputs/embedding_report.json (Report & statistics)
    with open(os.path.join(out_dir, "embedding_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    # 3. outputs/sample_embedding_output.json (Trimmed vector samples for git)
    if embedded_chunks:
        sample_output = create_sample_embedding_output(embedded_chunks, sample_size=sample_size)
        with open(os.path.join(out_dir, "sample_embedding_output.json"), "w", encoding="utf-8") as f:
            json.dump(sample_output, f, indent=2, ensure_ascii=False)

    return report_payload


def print_embedding_report(
    report: Dict[str, Any],
    embedded_chunks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]]
):
    """Prints the formatted embedding generation report matching specification."""
    stats = report.get("statistics", {})
    val = report.get("validation", {})
    model = report.get("embedding_model", "unknown")
    selected = stats.get("chunks_selected", 0)
    success = stats.get("chunks_successfully_embedded", 0)
    failed = stats.get("failed_embeddings", 0)
    dim = stats.get("vector_dimension", "N/A")
    val_status = val.get("status", "FAILED")

    print("\n=================================================")
    print("SHIPRULE - EMBEDDING GENERATION REPORT")
    print("=================================================")
    print(f"Embedding model: {model}")
    print(f"Chunks selected: {selected}")
    print(f"Chunks successfully embedded: {success}")
    print(f"Failed embeddings: {failed}")
    print(f"Vector dimension: {dim}\n")

    if embedded_chunks:
        print("Sample Results:")
        sample_count = min(2, len(embedded_chunks))
        for i in range(sample_count):
            item = embedded_chunks[i]
            vector = item.get("embedding", [])
            preview_str = format_vector_preview(vector, max_elements=3)
            print(f"[{i+1}] Chunk ID: {item.get('chunk_id')}")
            print(f"    Source: {item.get('source')}")
            print(f"    Characters: {item.get('character_count')}")
            print(f"    Vector length: {len(vector)}")
            print(f"    Vector preview: {preview_str}")

    if failures:
        print("\nFailures Encountered:")
        for f in failures:
            print(f"  • Chunk '{f.get('chunk_id')}': {f.get('error_message')}")

    print(f"\nValidation: {val_status}")
    if val_status == "PASSED":
        print("All successful embeddings have consistent dimensions.")
    else:
        for err in val.get("validation_errors", []):
            print(f"  • {err}")

    print("=================================================\n")
    print("[OUTPUT] Embedding artifacts saved to:")
    print("  • outputs/embedded_chunks.json")
    print("  • outputs/embedding_report.json")
    print("  • outputs/sample_embedding_output.json\n")


# ==============================================================================
# 7. END-TO-END EMBEDDING PIPELINE RUNNER
# ==============================================================================

def run_embedding_pipeline(
    chunks_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_chunks: Optional[int] = 5,
    client: Optional[Any] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes the end-to-end embedding generation pipeline.
    """
    load_dotenv()
    selected_model = model or os.getenv("EMBED_MODEL", "text-embedding-3-small")
    resolved_base_url = base_url or os.getenv("EMBEDDING_BASE_URL") or "https://api.openai.com/v1"
    resolved_api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")

    # 1. Provider configuration validation & info
    safe_config_summary = validate_embedding_provider_config(
        base_url=resolved_base_url,
        model=selected_model,
        api_key=resolved_api_key
    )

    if verbose:
        print("\n" + safe_config_summary + "\n")

    # 2. Load validated chunks
    chunks = load_validated_chunks(chunks_file=chunks_file, max_chunks=max_chunks)

    # 3. Create embedding client
    if client is None:
        try:
            client = create_embedding_client(base_url=resolved_base_url, api_key=resolved_api_key)
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to initialize embedding client:\n{e}\n")
            raise

    # 4. Generate embeddings with per-chunk isolation
    embedded_chunks, failures = embed_chunks(chunks, client=client, model=selected_model)

    # 5. Save results & reports
    report = save_embedding_results(
        embedded_chunks=embedded_chunks,
        failures=failures,
        model=selected_model,
        selected_count=len(chunks),
        output_dir=output_dir
    )

    # 6. Print summary
    if verbose:
        print_embedding_report(report, embedded_chunks, failures)

    return report


if __name__ == "__main__":
    run_embedding_pipeline()
