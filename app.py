"""
ShipRule CDLP - Application Entry Point
========================================
Runs the interactive ShipRule Customs Duty & Documentation Lookup Platform (CDLP) RAG session.
"""

import os
import sys

# Ensure root directory is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.main import main

if __name__ == "__main__":
    main()
