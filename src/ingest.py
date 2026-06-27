import os
import argparse
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

# --- Config ---
API_URL  = "http://127.0.0.1:8000/analyze"
DB_PATH  = "results/sna.db"
LOG_PATH = "logs/ingest.log"

DATASET_CONFIG = {
    "EN": ("ucberkeley-dlab/measuring-hate-speech", "default", "train", "text"),
    "ES": ("valeriobasile/HatEval",                 "default", "train", "text"),
    "PT": ("Paul/hatecheck-portuguese",             "default", "test", "test_case"),
    "FR": ("AxelDlv00/ToxiFrench",                  "default", "train", "content"), 
}

# --- Logging ---
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Hugging Face login
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

# --- SQLite Setup ---
def init_db(db_path: str) -> sqlite3.Connection:
    """Initializes the database schema for the SNA framework."""
    Path(db_path).parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS results (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text     TEXT    NOT NULL,
            lang              TEXT    NOT NULL,
            taxonomy_category TEXT    NOT NULL,
            entities_json     TEXT    NOT NULL,
            relations_json    TEXT    NOT NULL,
            created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS entities (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id   INTEGER NOT NULL REFERENCES results(id),
            entity_id   TEXT    NOT NULL,
            entity_type TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS relations (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id               INTEGER NOT NULL REFERENCES results(id),
            source                  TEXT    NOT NULL,
            target                  TEXT    NOT NULL,
            interaction_type        TEXT    NOT NULL,
            taxonomy_classification TEXT    NOT NULL,
            confidence_score        REAL,
            confidence_reasoning    TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_results_category ON results(taxonomy_category);
        CREATE INDEX IF NOT EXISTS idx_results_lang     ON results(lang);
        CREATE INDEX IF NOT EXISTS idx_relations_class  ON relations(taxonomy_classification);
    """)
    conn.commit()
    return conn

def save_result(conn: sqlite3.Connection, lang: str, api_response: dict):
    """Saves API processing output to SQLite storage."""
    original_text     = api_response.get("original_text", "")
    taxonomy_category = api_response.get("taxonomy_category", "NEUTRAL")
    entities          = api_response.get("extracted_entities", [])
    relations         = api_response.get("detected_relations", [])

    cur = conn.execute(
        """INSERT INTO results (original_text, lang, taxonomy_category, entities_json, relations_json)
           VALUES (?, ?, ?, ?, ?)""",
        (original_text, lang, taxonomy_category,
         json.dumps(entities, ensure_ascii=False),
         json.dumps(relations, ensure_ascii=False)),
    )
    result_id = cur.lastrowid

    conn.executemany(
        "INSERT INTO entities (result_id, entity_id, entity_type) VALUES (?, ?, ?)",
        [(result_id, e.get("id", ""), e.get("type", "")) for e in entities],
    )

    conn.executemany(
        """INSERT INTO relations
           (result_id, source, target, interaction_type, taxonomy_classification,
            confidence_score, confidence_reasoning)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [(result_id,
          r.get("source", ""),
          r.get("target", ""),
          r.get("interaction_type", ""),
          r.get("taxonomy_classification", "NEUTRAL"),
          r.get("confidence_score"),
          r.get("confidence_reasoning", ""))
         for r in relations],
    )
    conn.commit()

# --- API Integration ---

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def _post_api_with_retry(payload: dict) -> dict:
    """Executes POST request to the analysis endpoint with exponential backoff."""
    r = requests.post(API_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def call_api(entry_id: int, text: str) -> dict | None:
    """Sends text payload to the FastAPI endpoint with fault tolerance."""
    payload = {"id": entry_id, "text": text}
    
    try:
        return _post_api_with_retry(payload)
    except Exception as e:
        logger.error(f"All retries exhausted for text: {text[:60]!r} — Error: {e}")
        return None

# --- Dataset Loader ---
def load_texts(lang: str, limit: int) -> list[str]:
    """Pulls specified limit of rows from HuggingFace dataset."""
    if lang not in DATASET_CONFIG:
        raise ValueError(f"Unsupported lang: {lang}. Choose from {list(DATASET_CONFIG)}")

    dataset_name, config, split, text_col = DATASET_CONFIG[lang]
    logger.info(f"Loading dataset '{dataset_name}' (lang={lang}, limit={limit})")

    try:
        ds = load_dataset(dataset_name, config, split=split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    texts = [str(row[text_col]) for row in ds if row.get(text_col)]
    texts = [t for t in texts if len(t.strip()) > 10][:limit]
    logger.info(f"Loaded {len(texts)} texts")
    return texts

# --- Main Ingestion Logic ---
def run_ingestion(lang: str, limit: int, workers: int):
    logger.info(f"=== Ingestion started — lang={lang}, limit={limit}, workers={workers} ===")

    texts = load_texts(lang, limit)
    conn  = init_db(DB_PATH)

    success = 0
    failed  = 0

    def process(idx_text):
        idx, text = idx_text
        result = call_api(idx, text)
        return idx, text, result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, (i, t)): i for i, t in enumerate(texts, 1)}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                i, text, result = future.result()
                if result:
                    save_result(conn, lang, result)
                    logger.info(f"[{i}/{len(texts)}] ✓ {result.get('taxonomy_category')} — {text[:50]!r}")
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"[{idx}] Unexpected error: {e}", exc_info=True)
                failed += 1

    conn.close()
    logger.info(f"=== Ingestion finished — {success} saved, {failed} failed ===")
    logger.info(f"Database: {DB_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest dataset into SNA pipeline")
    parser.add_argument("--lang",    default="EN",  choices=["EN", "ES", "PT", "FR"])
    parser.add_argument("--limit",   default=100,   type=int, help="Max texts to process")
    parser.add_argument("--workers", default=2,     type=int, help="Concurrent API calls")
    args = parser.parse_args()

    run_ingestion(args.lang, args.limit, args.workers)

# Example usage:
# python src/ingest.py --lang EN --limit 100
# python src/ingest.py --lang ES --limit 100    