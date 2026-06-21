from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Literal
import json
import os

from src.detect import process_entry

app = FastAPI(
    title="SNA Analysis API",
    description="API for Entity Extraction and Relationship Classification via LLMs.",
    version="1.0.0"
)

# Severity hierarchy (most severe first)
SEVERITY_ORDER = ["EXTREMIST", "RADICALISM", "VIOLATED", "THREAT", "HATE", "EMOTIONAL", "SENTIMENTAL", "NEUTRAL"]

class TextPayload(BaseModel):
    id: int = 0
    text: str

class BatchEntry(BaseModel):
    id: int = 0
    text: str

class BatchPayload(BaseModel):
    entries: list[BatchEntry]

def remap_output(raw: dict) -> dict:
    """Maps process_entry output to the required schema structure."""
    analysis      = raw.get("analysis", {})
    relationships = analysis.get("relationships", [])

    labels_found      = [r.get("taxonomy_classification", "NEUTRAL") for r in relationships]
    taxonomy_category = "NEUTRAL"
    for level in SEVERITY_ORDER:
        if level in labels_found:
            taxonomy_category = level
            break

    return {
        "taxonomy_category":  taxonomy_category,
        "extracted_entities": analysis.get("entities", []),
        "detected_relations": relationships,
        "id":                 raw.get("id"),
        "original_text":      raw.get("original_text"),
    }

def _load_results() -> list:
    """Loads stored extraction results from disk."""
    file_path = "results/extraction_results.json"
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Results file not found. Run extraction pipeline first.",
        )
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.post(
    "/analyze",
    summary="Analyze text in real-time",
    tags=["Real-time Analysis"],
)
def analyze_text(payload: TextPayload):
    try:
        raw = process_entry(payload.id, payload.text)
        return remap_output(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error processing text: {str(e)}")

@app.post(
    "/analyze/batch",
    summary="Analyze multiple texts in real-time",
    tags=["Real-time Analysis"],
)
def analyze_batch(payload: BatchPayload):
    results, errors = [], []
    for entry in payload.entries:
        try:
            raw = process_entry(entry.id, entry.text)
            results.append(remap_output(raw))
        except Exception as e:
            errors.append({"id": entry.id, "error": str(e)})
    return {"results": results, "errors": errors}

@app.get("/analyze/health", tags=["Real-time Analysis"])
def health_check():
    return {"status": "ok", "pipeline": "ready"}

@app.get("/results", tags=["Batch Results"])
def get_batch_results():
    return _load_results()

@app.get("/results/{entry_id}", tags=["Batch Results"])
def get_result_by_id(entry_id: int):
    data    = _load_results()
    matches = [r for r in data if r.get("id") == entry_id]
    if not matches:
        raise HTTPException(status_code=404, detail=f"No result found for id={entry_id}")
    return matches[0]

@app.get("/results/search", tags=["Batch Results"])
def search_results(keyword: str = Query(...)):
    data          = _load_results()
    keyword_lower = keyword.lower()
    matches       = [r for r in data if keyword_lower in json.dumps(r).lower()]
    return {"keyword": keyword, "count": len(matches), "results": matches}

@app.delete("/results", tags=["Batch Results"])
def clear_results():
    file_path = "results/extraction_results.json"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Results file not found.")
    os.remove(file_path)
    return {"message": "Results file deleted successfully."}

@app.get("/stats", tags=["Stats"])
def get_stats():
    data            = _load_results()
    total_entries   = len(data)
    total_entities  = sum(len(r.get("analysis", {}).get("entities", [])) for r in data)
    total_relations = sum(len(r.get("analysis", {}).get("relationships", [])) for r in data)

    category_counts: dict[str, int] = {label: 0 for label in SEVERITY_ORDER}
    for r in data:
        for rel in r.get("analysis", {}).get("relationships", []):
            label = rel.get("taxonomy_classification", "NEUTRAL")
            if label in category_counts:
                category_counts[label] += 1

    return {
        "total_entries":               total_entries,
        "total_entities":              total_entities,
        "total_relationships":         total_relations,
        "classifications_breakdown":   category_counts,
    }
# To run: uvicorn src.api:app --reload
# Documentation: http://127.0.0.1:8000/docs