"""
ShipRule CDLP - Backend API Server Launcher
=============================================
Launches the FastAPI RAG service using Uvicorn on http://127.0.0.1:8000.
"""

import os
import sys
import uvicorn

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    print("[INFO] Starting ShipRule RAG Backend API Server on http://127.0.0.1:8000...")
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
