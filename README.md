# SNA Analysis API — Entity Extraction and Relationship Classification

This project is a **Social Network Analysis (SNA)** tool powered by LLMs and NLP models. It extracts entities (Persons, Groups, Institutions, Locations) and classifies the interactions between them through a **3-tier cascade pipeline**, detecting everything from basic sentiments to extremist rhetoric or direct threats.

---

## Features

- **Social Graph Extraction:** Identifies entities and interaction verbs using Llama-3.1-8B.
- **3-Tier Cascade Classification:**
  - **Tier 1:** Hate Speech filter (RoBERTa).
  - **Tier 2:** Sentiment baseline (Multilingual RoBERTa).
  - **Tier 3:** Severity escalation via LLM (Llama-3) for categories such as `EXTREMIST`, `RADICALISM`, `VIOLATED`, and `THREAT`.
- **Automatic Ingestion:** Script to import Hugging Face datasets (EN, ES, PT, FR) directly into the system.
- **Dual Persistence:** Results saved to JSON files for quick lookup and to a SQLite database for structured analysis.

---

## Prerequisites

Before you begin, make sure you have the following installed:

- **Python 3.9+**
- **Pip** (Python package manager)
- **Hugging Face Token (`HF_TOKEN`):** Required to access models via the Inference API. Get yours at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Installation

**1. Clone the repository** (or organize the files in a folder):

```bash
mkdir sna-analysis && cd sna-analysis
# Place api.py, detect.py, ingest.py, and requirements.txt inside the src/ folder
```

**2. Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Set up environment variables:**

Create a `.env` file in the project root:

```env
HF_TOKEN=your_token_here
```

**5. Create the required directories:**

```bash
mkdir logs results data
```

---

## How to Run

### 1. Start the API (Backend)

The API must be running for real-time processing or ingestion to work.

```bash
uvicorn src.api:app --reload
```

> **Interactive Docs (Swagger):** Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test the endpoints manually.

### 2. Run Local Batch Processing

If you have a `data/examples.txt` file with one sentence per line, you can run the detection script directly:

```bash
python src/detect.py
```

Results will be saved to `results/extraction_results.json`.

### 3. Ingest Real Datasets

To populate the database with data from Hugging Face (e.g. Measuring Hate Speech, HatEval):

```bash
# Ingest 100 Portuguese texts
python src/ingest.py --lang PT --limit 100 --workers 2

# Ingest 50 English texts
python src/ingest.py --lang EN --limit 50
```

Data will be processed by the API and stored in `results/sna.db`.

---

## Classification Structure (Hierarchy)

The system classifies relationships following a descending severity order:

| Level | Category | Description |
|---|---|---|
| 1 | `EXTREMIST` | Calls for terrorism or mass violence. |
| 2 | `RADICALISM` | Systematic dehumanization of groups. |
| 3 | `VIOLATED` | Description of physical violence that occurred. |
| 4 | `THREAT` | Explicit threats directed at a person or group. |
| 5 | `HATE` | Hate speech detected by Tier 1. |
| 6 | `EMOTIONAL` | Negative sentiment (Tier 2). |
| 7 | `SENTIMENTAL` | Positive sentiment (Tier 2). |
| 8 | `NEUTRAL` | None of the above. |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Analyzes a single text and returns the graph + category. |
| `POST` | `/analyze/batch` | Processes multiple texts at once. |
| `GET` | `/results` | Lists all results saved in the JSON file. |
| `GET` | `/stats` | Returns global statistics (avg entities, category counts). |
| `DELETE` | `/results` | Clears the results history. |

---

## Sample Output (JSON)

```json
{
  "taxonomy_category": "THREAT",
  "extracted_entities": [
    {"id": "Author", "type": "Person"},
    {"id": "Target", "type": "Person"}
  ],
  "detected_relations": [
    {
      "source": "Author",
      "target": "Target",
      "interaction_type": "threatens",
      "taxonomy_classification": "THREAT",
      "confidence_score": 95.0,
      "confidence_reasoning": "Explicit threat to cause physical harm to the target's property."
    }
  ]
}
```
