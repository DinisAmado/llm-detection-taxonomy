# Technical Documentation: detect.py (Unified Extractor)

## Overview
The `detect.py` script serves as the core of the Unified Generative Extraction Layer. It replaces the previous multi-script architecture by consolidating all 7 taxonomy classifications into a single LLM pass.

## Technical Specifications
* **Model**: `gpt-4o-mini`.
* **Input**: Raw text strings from `examples.txt`.
* **Output**: Validated JSON objects stored in `results/extraction_results.json`.

## Key Functions

### `extract_intelligence(text)`
This function handles the primary communication with the OpenAI API.
* **System Prompt**: Implements the "Master Prompt" logic, including NER and Relation Extraction rules.
* **Format Constraint**: Uses `response_format={"type": "json_object"}` to ensure structural integrity.
* **Error Handling**: Includes a `safe_json_load` fallback to handle any potential parsing issues.

### `run_batch_test()`
Manages the batch processing workflow.
* **File Reading**: Iterates through `examples.txt`, filtering out invalid or short lines.
* **Rate Limiting**: Implements a `time.sleep(0.4)` delay to respect API limits.
* **Persistence**: Aggregates results into a single list for final JSON export.

## Data Schema
The script strictly follows the Graph-ready schema defined in the project requirements:

| Field | Description |
| :--- | :--- |
| `entities` | List of identified Persons, Groups, Institutions, or Locations. |
| `relationships` | Mapped interactions between entities (Source -> Target). |
| `taxonomy_classification` | One of the 7 project categories (Extremist, Hate, etc.). |
| `confidence_reasoning` | Brief justification for the assigned classification. |

## Security
* **Environment Variables**: API keys are loaded via `python-dotenv` from a protected `.env` file.
* **Git Safety**: The `.env` file must be listed in `.gitignore` to prevent credential exposure.