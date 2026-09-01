"""
ShipRule CDLP - Corpus Ingestion CLI Entry Point
=================================================
Runs the end-to-end corpus ingestion and validation pipeline over the knowledge base,
tracks every file in the manifest, verifies reconciliation, and saves outputs.
"""

import sys
import os
import argparse

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion import run_ingestion_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="ShipRule Corpus Preparation & Ingestion Validation Pipeline"
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=os.path.join(project_root, "data", "sample_corpus"),
        help="Path to the document corpus directory (default: data/sample_corpus)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(project_root, "outputs"),
        help="Directory to store manifests, reports, and processed chunks (default: outputs)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="paragraph",
        choices=["paragraph", "fixed", "sentence"],
        help="Chunking strategy to apply during ingestion (default: paragraph)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Enable resumable ingestion to skip unchanged, previously ingested files"
    )

    args = parser.parse_args()
    run_ingestion_pipeline(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        strategy=args.strategy,
        resumable=args.resume,
        verbose=True
    )


if __name__ == "__main__":
    main()
