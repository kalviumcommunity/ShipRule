"""
ShipRule CDLP - Embedding Quality Checks & Sanity Tests CLI
============================================================
CLI entry point to execute lightweight sanity tests over embedding and retrieval.
Verifies vector dimensions, model consistency, metadata alignment, cosine similarity,
and expected retrieval source ranking.
"""

import sys
import os
import argparse

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sanity_checker import run_sanity_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="ShipRule Embedding Quality Checks & Sanity Tests CLI"
    )
    parser.add_argument(
        "--chunks-file",
        type=str,
        default=os.path.join(project_root, "outputs", "processed_chunks.json"),
        help="Path to the validated chunks file (default: outputs/processed_chunks.json)"
    )
    parser.add_argument(
        "--test-cases-file",
        type=str,
        default=None,
        help="Path to custom JSON test cases file (optional)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Configurable top-k threshold for source evaluation (default: 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(project_root, "outputs"),
        help="Directory to save generated sanity report artifact (default: outputs)"
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

    args = parser.parse_args()

    try:
        run_sanity_pipeline(
            chunks_file=args.chunks_file,
            test_cases_file=args.test_cases_file,
            top_k=args.top_k,
            output_dir=args.output_dir,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            verbose=True
        )
    except Exception as e:
        print(f"\n[SANITY TEST PIPELINE HALTED]\n{e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
