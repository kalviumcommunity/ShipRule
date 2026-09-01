"""
Export Sample Chunks & Traceability Demonstrator (KDU 3.22)
============================================================
Generates sample metadata-tagged chunks and outputs source traceback evidence to
outputs/sample_chunk_metadata.json and outputs/sample_chunks_output.txt.
"""

import json
import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chunk_metadata import (
    chunk_document_with_metadata,
    trace_chunk_source,
    validate_chunk_metadata_schema
)


def export_sample_data():
    sample_prd_text = (
        "CDLP Functional Requirements: FR-01 to FR-07 (Data Integration: store duty rates, "
        "import docs, restricted status, source agency, source URL, last confirmed date). "
        "FR-08 to FR-12 (Validation: two-person validation required before publishing, validation "
        "history, flag re-verification). FR-13 to FR-19 (Lookup: search by destination country "
        "+ HS code, display duty/docs/restrictions/source, explicit Not Mapped state)."
    )

    sample_regulation_text = (
        "Customs Record for Laptop Computers in India: Destination Country: India. Commodity: "
        "Laptop Computers / Portable Automatic Data Processing Machines. HS Code: 8471.30. "
        "Duty Rate: 7.5% Basic Customs Duty + 10% Social Welfare Surcharge (SWS). Required "
        "Documents: Commercial Invoice, Bill of Lading, BIS Registration Certificate, DGFT Import License. "
        "Restricted Status: Restricted (Import License Required). Source Agency: Directorate General of "
        "Foreign Trade (DGFT) & CBIC, India. Source URL: https://www.cbic.gov.in. Last Confirmed Date: 2026-02-10."
    )

    # Chunk PRD document with metadata
    prd_chunks = chunk_document_with_metadata(
        text=sample_prd_text,
        source="cdlp_prd_v1.0.md",
        section="8. Functional Requirements",
        doc_type="PRD",
        chunk_size=160,
        chunk_overlap=30,
        page=5
    )

    # Chunk Customs Regulation document with metadata
    reg_chunks = chunk_document_with_metadata(
        text=sample_regulation_text,
        source="customs_reg_india.json",
        section="Laptop Import Requirements (India)",
        doc_type="RegulationData",
        chunk_size=200,
        chunk_overlap=40,
        page=1,
        extra_metadata={
            "country": "India",
            "hs_code": "8471.30",
            "source_agency": "DGFT & CBIC, India",
            "source_url": "https://www.cbic.gov.in",
            "last_confirmed_date": "2026-02-10"
        }
    )

    all_sample_chunks = prd_chunks + reg_chunks

    # Perform traceback demonstrations
    tracebacks = [trace_chunk_source(c) for c in all_sample_chunks]

    export_payload = {
        "task_name": "KDU 3.22 Chunk Metadata & Source Tracking",
        "description": "Consistent chunk tagging, metadata schema enforcement, and source traceability demonstration",
        "consistent_metadata_schema_keys": [
            "source", "section", "chunk_index", "char_start", "char_end", "doc_type",
            "page", "country", "hs_code", "last_confirmed_date", "source_agency", "source_url"
        ],
        "total_sample_chunks": len(all_sample_chunks),
        "schema_validation_passed": all(validate_chunk_metadata_schema(c) for c in all_sample_chunks),
        "sample_chunks": all_sample_chunks,
        "sample_tracebacks": tracebacks
    }

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "sample_chunk_metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    txt_path = os.path.join(output_dir, "sample_chunks_output.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== KDU 3.22 CHUNK METADATA & SOURCE TRACKING OUTPUT ===\n\n")
        f.write(f"Schema Validation Status: {'ALL PASSED' if export_payload['schema_validation_passed'] else 'FAILED'}\n")
        f.write(f"Total Sample Chunks: {len(all_sample_chunks)}\n\n")
        f.write("--- SAMPLE CHUNKS (TEXT + METADATA) ---\n\n")
        for idx, chunk in enumerate(all_sample_chunks):
            f.write(f"Chunk #{idx + 1}:\n")
            f.write(f"Text: \"{chunk['text']}\"\n")
            f.write("Metadata:\n")
            for k, v in chunk["metadata"].items():
                f.write(f"  - {k}: {v}\n")
            f.write("\n")

        f.write("--- SOURCE TRACEBACK DEMONSTRATIONS ---\n\n")
        for idx, trace in enumerate(tracebacks):
            f.write(f"Traceback #{idx + 1}:\n")
            f.write(f"  Formatted Citation: {trace['formatted_citation']}\n")
            f.write(f"  Source Document: {trace['source']}\n")
            f.write(f"  Section Heading: {trace['section']}\n")
            f.write(f"  Position Span: {trace['char_start']}..{trace['char_end']}\n")
            f.write(f"  Source Agency: {trace['source_agency']}\n")
            f.write(f"  Source URL: {trace['source_url']}\n\n")

    print(f"[Export] Successfully generated {json_path}")
    print(f"[Export] Successfully generated {txt_path}")


if __name__ == "__main__":
    export_sample_data()
