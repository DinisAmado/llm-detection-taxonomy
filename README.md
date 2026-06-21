# SNA Analysis API — Entity Extraction and Relationship Classification

This project is a **Social Network Analysis (SNA)** tool powered by LLMs and NLP models. It extracts entities (Persons, Groups, Institutions, Locations) and classifies the interactions between them through a **3-tier cascade pipeline**, detecting everything from basic sentiments to extremist rhetoric or direct threats.

---

## Features

- **Social Graph Extraction:** Identifies entities and interaction verbs using Llama-3.1-8B.
- **3-Tier Cascade Classification:**
  - **Tier 1:** Hate Speech filter (RoBERTa).
  - **Tier 2:** Sentiment baseline (Multilingual RoBERTa / DistilBERT).
  - **Tier 3:** Severity escalation via LLM (Llama-3) for categories such as `EXTREMIST`, `RADICALISM`, `VIOLATED`, and `THREAT`.
- **Text Normalisation:** Automatic language detection, emoji demojization, URL/mention scrubbing, slang expansion (PT/ES), and character deduplication before any model call.
- **Graph Guardrail:** Post-extraction cleaning step that normalises pronouns to `AUTHOR`/`TARGET`, removes duplicate nodes, and filters relationships to only valid entity pairs.
- **Knowledge Graph Visualisation:** Generates interactive HTML (pyvis/vis.js) and static PNG graphs colour-coded by entity type and severity, plus optional per-category subgraphs.
- **Graph Analytics:** Computes in-degree, betweenness centrality, and PageRank (influence) for every node and exports a `metrics_report.json` for the dashboard.
- **Decision Support Dashboard:** Streamlit app with critical alerts, network influencer rankings, and an embedded interactive knowledge graph.
- **Automatic Ingestion:** Script to import Hugging Face datasets (EN, ES, PT, FR) directly into the system.
- **Dual Persistence:** Results saved to JSON files for quick lookup and to a SQLite database for structured analysis.
- **Validation & Ablation Study:** Standalone script that generates a confusion matrix, class distribution chart, and comparative ROC curve (hybrid pipeline vs. keyword baseline), plus a full metrics report (Accuracy, Precision, Recall, F1-Score).

---

## Project Structure

```
sna-analysis/
├── src/
│   ├── detect.py      # Core pipeline: text normalisation, graph extraction, 3-tier classification
│   ├── api.py         # FastAPI server exposing /analyze, /results, /stats endpoints
│   ├── ingest.py      # Batch ingestion from Hugging Face datasets into SQLite
│   ├── kg.py          # Knowledge graph builder: HTML + PNG exports, subgraphs, CLI
│   ├── analytics.py   # Graph metrics: centrality, PageRank, community detection
│   ├── app.py         # Streamlit decision support dashboard
│   └── validate.py    # Validation script: confusion matrix, ROC curve, metrics report
├── data/
│   └── examples.txt   # One sentence per line — input for local batch processing
├── results/
│   ├── extraction_results.json
│   └── sna.db
├── graphs/
│   ├── knowledge_graph.html
│   ├── knowledge_graph.png
│   ├── metrics_report.json
│   └── validation/
│       ├── confusion_matrix.png
│       ├── class_distribution.png
│       ├── roc_curve_comparison.png
│       └── metrics_report.txt
├── logs/
└── .env
```

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
# Place all .py files inside the src/ folder
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
mkdir logs results data graphs
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

### 4. Generate the Knowledge Graph

After the database is populated, run `kg.py` to build the graphs and compute analytics:

```bash
# Full graph (HTML + PNG)
python src/kg.py

# Filter by language and severity category
python src/kg.py --lang PT EN --category HATE THREAT

# Limit to the 50 most-connected entities and export per-category subgraphs
python src/kg.py --top-entities 50 --subgraphs

# Skip PNG, generate only the interactive HTML
python src/kg.py --no-png
```

All outputs (HTML, PNG, `metrics_report.json`) are written to the `graphs/` directory.

### 5. Launch the Dashboard

```bash
streamlit run src/app.py
```

> Requires the database (`results/sna.db`) and graph files (`graphs/`) to exist. Run `ingest.py` and `kg.py` first.

### 6. Run the Validation & Ablation Study

After ingestion and graph generation, run the validation script to evaluate the pipeline's classification performance:

```bash
python src/validate.py
```

This generates four outputs inside `graphs/validation/`:

| Output | Description |
|---|---|
| `confusion_matrix.png` | Confusion matrix of the hybrid pipeline (TP, FP, TN, FN). |
| `class_distribution.png` | Bar chart showing the 2:1 class imbalance (Severe vs. No Risk). |
| `roc_curve_comparison.png` | Comparative ROC curves: hybrid pipeline vs. keyword-based baseline. |
| `metrics_report.txt` | Text report with Accuracy, Precision, Recall, and F1-Score. |

The script uses a fixed random seed (`numpy.random.seed(42)`) to ensure reproducibility of the simulated ground truth.

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
| `GET` | `/analyze/health` | Returns the current status of the API pipeline. |
| `GET` | `/results` | Lists all results saved in the JSON file. |
| `GET` | `/results/{id}` | Returns a single saved result by numeric ID. |
| `GET` | `/results/search` | Filters saved results by keyword (case-insensitive). |
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

---

## Supported Datasets (Ingestion)

| Language | Dataset |
|---|---|
| EN | `ucberkeley-dlab/measuring-hate-speech` |
| ES | `valeriobasile/HatEval` |
| PT | `Paul/hatecheck-portuguese` |
| FR | `AxelDlv00/ToxiFrench` |

---

## Models Used

| Role | Model |
|---|---|
| Graph extraction (Stage 1) | `meta-llama/Llama-3.1-8B-Instruct` |
| Hate speech detection (Tier 1) | `facebook/roberta-hate-speech-dynabench-r4-target` |
| Sentiment — English (Tier 2) | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Sentiment — Multilingual (Tier 2) | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` |
| Severity escalation (Tier 3) | `meta-llama/Meta-Llama-3-8B-Instruct` |