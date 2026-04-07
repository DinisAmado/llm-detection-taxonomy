# llm-detection-taxonomy
Python project to classify different types of speech
# Two-Stage Open-Source Ensemble: Intention Detection for SNA

This project implements a robust, two-stage Natural Language Processing (NLP) pipeline for Knowledge Graph extraction and intention classification in social media contexts. The solution utilizes an **Open-Source Ensemble** hosted on the Hugging Face Hub, removing dependencies on proprietary APIs.

## Architecture Overview

The system utilizes a **Segmented Model Architecture** to maximize accuracy across different linguistic taxonomies:

### Stage 1: Base Graph Extractor (NER & RE)
A large-scale generative model (**Llama 3.1 8B Instruct**) performs Named Entity Recognition (NER) and Relation Extraction (RE).
* **Input:** Raw social media posts or text lines.
* **Output:** A base JSON schema containing socially relevant entities and their core interactions (Source -> Target -> Interaction).

### Stage 2: Taxonomy Routing
Interactions identified in Stage 1 are routed through specialized model groups based on their context:
* **Group A (Sentiment/Emotion):** Powered by `twitter-roberta-base-sentiment-latest` for sentiment analysis.
* **Group B (Hate/Threat):** Powered by `roberta-hate-speech-dynabench-r4-target` to detect toxicity and direct threats.
* **Group C (Reasoners):** Powered by `Meta-Llama-3-8B-Instruct` for complex ideological analysis, such as Radicalism or Extremism.

## Tech Stack

* **Language:** Python 3.10+
* **Inference:** Hugging Face Inference API (Serverless)
* **Parallelization:** `ThreadPoolExecutor` for batch processing optimization
* **Key Libraries:** `huggingface_hub`, `python-dotenv`, `concurrent.futures`

## Setup and Execution

### 1. Prerequisites
* Install dependencies: `pip install huggingface_hub python-dotenv`
* Create a `.env` file in the root directory with your access token:
    ```env
    HF_TOKEN=your_huggingface_token_here
    ```

### 2. Running the Pipeline
The script `detect.py` automatically processes the `examples.txt` file and handles API rate limits using exponential backoff.

```bash
python detect.py

```

## Output Schema (JSON)
The final output is stored in results/extraction_results.json, featuring a rich schema designed for Neo4j Knowledge Graph injection:

{
  "id": 13,
  "original_text": "You swore not to lay your hands on Charlie...",
  "analysis": {
    "entities": [...],
    "relationships": [
      {
        "source": "Author",
        "target": "Target",
        "interaction_type": "hates",
        "taxonomy_classification": "THREAT",
        "confidence_reasoning": "Group B Encoder detected hate speech.",
        "accuracy": "99.99%"
      }
    ]
  }
}

## Taxonomy Compliance

The system strictly adheres to the 7 official categories required for the project:

EMOTIONAL

EXTREMIST

HATE

RADICALISM

SENTIMENTAL

THREAT

VIOLATED

Note: Neutral interactions are dynamically mapped to SENTIMENTAL to ensure data integrity and 100% compatibility with the final taxonomy.