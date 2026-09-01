"""
ShipRule CDLP - Document Chunking CLI Entry Point
==================================================
Runs the chunking pipeline over the sample corpus, displays comparative statistics,
inspects chunk boundaries, and writes reports to outputs/.
"""

import sys
import os

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chunker import run_chunking_pipeline

if __name__ == "__main__":
    run_chunking_pipeline()
