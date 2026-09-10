"""
ShipRule CDLP - Backend API Terminal Demo
==========================================
Demonstrates live requests to the ShipRule RAG FastAPI Backend API
and prints formatted JSON outputs directly to the terminal.
"""

import os
import sys
import json

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# UTF-8 stdout configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from app.api import app


def run_terminal_demo():
    client = TestClient(app)

    print("================================================================================")
    print("                SHIPRULE RAG BACKEND API TERMINAL DEMO                          ")
    print("================================================================================")
    print("")

    # 1. GET / Root Endpoint
    print(">>> 1. GET / (Root Service Metadata Endpoint)")
    print("--------------------------------------------------------------------------------")
    r1 = client.get("/")
    print(f"HTTP Status Code: {r1.status_code}")
    print(json.dumps(r1.json(), indent=2))
    print("")

    # 2. GET /health Check Endpoint
    print(">>> 2. GET /health (Health Check Endpoint)")
    print("--------------------------------------------------------------------------------")
    r2 = client.get("/health")
    print(f"HTTP Status Code: {r2.status_code}")
    print(json.dumps(r2.json(), indent=2))
    print("")

    # 3. POST /query Valid Query
    question = "What shipping documentation is required for international customs clearance?"
    print(f">>> 3. POST /query (Valid User Question: '{question}')")
    print("--------------------------------------------------------------------------------")
    payload = {
        "question": question,
        "use_reranking": True,
        "final_k": 3
    }
    r3 = client.post("/query", json=payload)
    print(f"HTTP Status Code: {r3.status_code}")
    res_data = r3.json()
    print(json.dumps(res_data, indent=2))
    print("")

    # 4. POST /query Validation Error (< 3 chars)
    print(">>> 4. POST /query (Input Validation Error: question='hi')")
    print("--------------------------------------------------------------------------------")
    r4 = client.post("/query", json={"question": "hi"})
    print(f"HTTP Status Code: {r4.status_code}")
    print(json.dumps(r4.json(), indent=2))
    print("")

    print("================================================================================")
    print("                DEMO COMPLETE - ALL ENDPOINTS OPERATIONAL                       ")
    print("================================================================================")


if __name__ == "__main__":
    run_terminal_demo()
