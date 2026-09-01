"""
ShipRule CDLP - Embedding Generation CLI Entry Point
=====================================================
Generates numerical vector embeddings for validated document chunks via
a dedicated OpenAI-compatible embeddings API endpoint, verifies vector dimensions,
and saves full and sample embedding artifacts to outputs/.
"""

import sys
import os
import argparse

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.embeddings import run_embedding_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="ShipRule Embedding Generation Pipeline"
    )
    parser.add_argument(
        "--chunks-file",
        type=str,
        default=os.path.join(project_root, "outputs", "processed_chunks.json"),
        help="Path to the validated chunks file (default: outputs/processed_chunks.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(project_root, "outputs"),
        help="Directory to save generated embedding artifacts (default: outputs)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model name (defaults to EMBED_MODEL in .env)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom embedding provider base URL (defaults to EMBEDDING_BASE_URL in .env)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Custom embedding API key (defaults to EMBEDDING_API_KEY in .env)"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=5,
        help="Maximum number of chunks to process to control cost (default: 5, use 0 for all)"
    )

    args = parser.parse_args()
    max_c = args.max_chunks if args.max_chunks > 0 else None

    try:
        run_embedding_pipeline(
            chunks_file=args.chunks_file,
            output_dir=args.output_dir,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_chunks=max_c,
            verbose=True
        )
    except Exception as e:
        print(f"\n[EMBEDDING PIPELINE HALTED]\n{e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
