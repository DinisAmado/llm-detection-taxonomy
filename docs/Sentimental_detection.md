# sentimental.py — Sentiment Classification System

## Overview
A sentiment analysis system using Twitter-RoBERTa to classify Portuguese/English text for emotional polarity. Leverages pre-trained transformer model for fast, accurate sentiment detection distinguishing between emotionally-charged and neutral content.

## System Architecture

### Dependencies
```python
from huggingface_hub import InferenceClient  # LLM/ML API client
import re                                     # Text cleaning
import time                                   # Rate limiting
```

### Model Configuration
```python
HF_TOKEN = "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"
model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
```

**Why Twitter-RoBERTa?**
- Specialized sentiment model trained on Twitter data
- Handles informal language, slang, and emojis well
- Fast inference (token classification vs. chat completion)
- Three-class output: positive, negative, neutral
- Lower latency and cost compared to chat models

### Inference Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model Type | `text_classification` | Direct sentiment prediction (not chat) |
| API Method | `.text_classification()` | Efficient token-level classification |
| Rate Limit | 0.5 seconds | Fast processing without quota exhaustion |

---

## Function Reference

### `classify_sentimental(text: str) → str`

**Purpose:** Classify a single text for sentiment polarity  
**Input:** Text string (Portuguese or English)  
**Output:** `"SENTIMENTAL"`, `"OTHER"`, or `"ERRO_API"`

**Process:**
1. Send text to Twitter-RoBERTa API via `text_classification()`
2. Receive list of results with labels and confidence scores
3. Extract result with highest confidence score
4. Get sentiment label (lowercase): 'positive', 'negative', or 'neutral'
5. Check if label matches sentimental triggers
6. Return classification or error code

**API Response Structure:**
```python
results = [
    {'label': 'POSITIVE', 'score': 0.95},
    {'label': 'NEGATIVE', 'score': 0.03},
    {'label': 'NEUTRAL', 'score': 0.02}
]
```

**Label Extraction Logic:**
```python
# Find highest confidence prediction
best_result = max(results, key=lambda x: x['score'])

# Normalize label (RoBERTa returns uppercase)
label = best_result['label'].lower()  # e.g., 'positive'

# Check classification criteria
if label in ['positive', 'negative']:
    return "SENTIMENTAL"  # Has sentiment polarity
return "OTHER"  # Neutral (no strong sentiment)
```

**Output Logic:**
- `'positive'` → SENTIMENTAL (positive emotion/stance)
- `'negative'` → SENTIMENTAL (negative emotion/stance)
- `'neutral'` → OTHER (no emotional charge)

**Error Handling:**
```python
except Exception as e:
    print(f"\n[DEBUG] Erro na API: {e}")
    return "ERRO_API"
```
- Catches API timeouts, malformed responses, network errors
- Prints debug message for troubleshooting
- Returns error code but continues batch processing

---

### `run_test() → None`

**Purpose:** Execute full sentiment classification pipeline end-to-end  
**Input:** `examples.txt` file (required)  
**Output:** Formatted console report + aggregate statistics

**Execution Flow:**
1. Initialize stats dictionary: `{"SENTIMENTAL": 0, "OTHER": 0, "ERRO_API": 0}`
2. Open `examples.txt` with UTF-8 encoding
3. Print formatted table header
4. For each line in file:
   - Strip whitespace
   - Skip if empty or < 5 characters
   - Remove backslash escape sequences
   - Call `classify_sentimental()`
   - Increment corresponding stat counter
   - Print formatted result (classification + text preview)
   - Sleep 0.5 seconds (rate limiting)
5. Print summary statistics table
6. Handle FileNotFoundError gracefully

**Rate Limiting Strategy:**
```python
time.sleep(0.5)  # 0.5 second delay between calls
```
- Prevents API throttling/rate limit errors
- ~120 texts per minute processing speed
- Faster than chat-based models (direct classification)
- Trade-off: Balance speed with reliability

**Output Format:**

Per-line output:
```
{RESULT:<15} | {TEXT[:75]}...
```
- Result: Left-aligned in 15-character field
- Separator: Pipe character ("|")
- Text: First 75 characters (truncated with ellipsis if longer)

Summary statistics:
```
==============================
RESUMO (SENTIMENTAL):
Sentimentais: {stats['SENTIMENTAL']}
Outros: {stats['OTHER']}
==============================
```

**Statistics Dictionary:**
```python
stats = {"SENTIMENTAL": 0, "OTHER": 0, "ERRO_API": 0}
stats[result] = stats.get(result, 0) + 1  # Safe increment with fallback
```

---

## Input/Output Specification

### Input Requirements
- **Source File:** `examples.txt` (must exist in working directory)
- **Encoding:** UTF-8 (handles Portuguese accents: ç, ã, é, etc.)
- **Format:** One text per line (newline-separated)
- **Language:** Portuguese/English optimized
- **Minimum Length:** 5 characters per line
- **No Maximum Length:** Supports arbitrary text length

### Input Validation
```python
if not line or len(line) < 5:
    continue  # Skip empty lines and very short text
```

### Input Preprocessing
```python
clean_text = re.sub(r'\\', '', line).strip()
```
- Removes backslash escape characters
- Strips leading/trailing whitespace
- Validates non-empty after cleaning

### Output Format Specification

**Per-Line Output:**
```
{RESULT:<15} | {TEXT[:75]}...
```
- Result: Left-aligned field (15 characters wide)
- Separator: Pipe character ("|") with spaces
- Text: First 75 characters only
- Ellipsis: Always appended (even for short text)

**Summary Statistics:**
- Header: "=" * 30 separator
- Label: "RESUMO (SENTIMENTAL):"
- Counters: "Category: {count}" format
- Footer: "=" * 30 separator

### Example Output Mapping
```
Input:  "Estou muito feliz e satisfeito com meu trabalho"
Output: "SENTIMENTAL     | Estou muito feliz e satisfeito com meu trabalho..."
Label:  POSITIVE (RoBERTa) → SENTIMENTAL (classification)

Input:  "O café é quente"
Output: "OTHER           | O café é quente..."
Label:  NEUTRAL (RoBERTa) → OTHER (classification)

Input:  "Que pessoa horrorosa e desagradável!"
Output: "SENTIMENTAL     | Que pessoa horrorosa e desagradável!..."
Label:  NEGATIVE (RoBERTa) → SENTIMENTAL (classification)
```

---

## Error Handling & Recovery

| Error Type | Trigger | Handling | Impact |
|------------|---------|----------|--------|
| **FileNotFoundError** | `examples.txt` missing | Print error, exit gracefully | No classifications run |
| **API Connection Error** | Network timeout/service down | Return `"ERRO_API"`, continue | Single line skipped; batch continues |
| **Malformed JSON Response** | Invalid API response structure | Caught by try-except | Classification fails for that text |
| **Empty Results List** | API returns no predictions | `max()` fails with ValueError | Caught; returns `"ERRO_API"` |
| **Authentication Failure** | Invalid HF_TOKEN | API raises 401 error | All classifications fail |
| **Non-UTF-8 File** | Invalid encoding in examples.txt | UnicodeDecodeError on read | Script crashes (no fallback encoding) |
| **KeyError on Results** | Missing 'label' or 'score' key | Caught by exception handler | Returns `"ERRO_API"` |

**Resilience:** Script continues processing after individual line failures; one error doesn't halt entire batch.

---

## Workflow Diagram

```
START
  │
  ├─→ Load examples.txt
  │    (UTF-8 encoding)
  │
  ├─→ Initialize: stats = {SENTIMENTAL: 0, OTHER: 0, ERRO_API: 0}
  │
  ├─→ Print table header: "RESULTADO | TEXTO"
  │
  ├─→ FOR EACH line in file:
  │    │
  │    ├─→ line.strip()
  │    │
  │    ├─→ Skip if empty OR len < 5
  │    │
  │    ├─→ Remove backslashes (regex)
  │    │
  │    ├─→ Call classify_sentimental(clean_text)
  │    │    │
  │    │    ├─→ Send to Twitter-RoBERTa API
  │    │    │    ├─ API: text_classification()
  │    │    │    └─ Return: [{label, score}, ...]
  │    │    │
  │    │    ├─→ Parse response
  │    │    │    ├─ max() by score
  │    │    │    ├─ Extract label
  │    │    │    └─ Lowercase normalize
  │    │    │
  │    │    ├─→ Classification logic
  │    │    │    ├─ label in ['positive', 'negative']?
  │    │    │    ├─ YES → return "SENTIMENTAL"
  │    │    │    └─ NO → return "OTHER"
  │    │    │
  │    │    ├─→ [Exception] → return "ERRO_API"
  │    │    │
  │    │    └─→ Return result
  │    │
  │    ├─→ stats[result] += 1
  │    │
  │    ├─→ Print: "{result:<15} | {text[:75]}..."
  │    │
  │    └─→ time.sleep(0.5)
  │
  ├─→ Print separator: "=" * 30
  │
  ├─→ Print "RESUMO (SENTIMENTAL):"
  │    ├─ Sentimentais: {stats['SENTIMENTAL']}
  │    ├─ Outros: {stats['OTHER']}
  │    └─ Separator: "=" * 30
  │
  └─→ END / [FileNotFoundError] → Print error, exit
```

---

## Performance Characteristics

### Computational Cost
- **Per-text latency:** 0.2-0.7 seconds (API call + parsing + sleep)
- **Throughput:** ~120 texts/minute (with 0.5s rate limit)
- **100 samples:** ~50 seconds
- **Bottleneck:** API rate limiting (0.5s mandatory delay) + network latency

### Token Usage
| Phase | Tokens |
|-------|--------|
| Input text | Varies (up to 512 tokens for RoBERTa) |
| Special tokens | ~2-3 tokens (CLS, SEP) |
| **Per call** | ~5-515 tokens (text-dependent) |

### Network Requirements
- Continuous internet connection (API-dependent)
- Hugging Face Inference API accessibility
- Typical API latency: 100-400ms (faster than LLM chat models)

### Storage Requirements
- Code: ~1.5 KB
- Does not cache results
- Output: Console only (not logged)

---

## Notable Implementation Details

### Why `text_classification` vs. Chat Models?

This approach differs from previous chat-based classifiers:
- **Direct classification:** RoBERTa fine-tuned specifically for sentiment
- **Faster inference:** ~100-200ms vs. 1-2s for chat models
- **Lower cost:** Token-efficient classification
- **Simpler output:** Direct label + score (no parsing needed)
- **Better for Twitter:** Trained on social media language

### Why `max()` on Results?

```python
best_result = max(results, key=lambda x: x['score'])
```
- RoBERTa returns all three classes with probabilities
- Taking `max()` selects highest confidence prediction
- Assumes multi-class prediction (always one dominant label)
- Lambda function extracts 'score' field for comparison

### Label Normalization
```python
label = best_result['label'].lower()
```
- RoBERTa API returns uppercase labels: 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
- Conversion to lowercase simplifies string matching
- Prevents case-sensitivity bugs in comparison

### Binary Decision Logic
```python
if label in ['positive', 'negative']:
    return "SENTIMENTAL"
return "OTHER"
```
- Simple, deterministic classification
- Treats positive and negative equally (both are "sentimental")
- Neutral texts → "OTHER" (no emotional charge)
- Conservative: Only flags clear sentiments

---

## Security Considerations ⚠️

### Critical Issues

1. **Hardcoded API Token in Source Code**
   ```python
   HF_TOKEN = "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"
   ```
   - **Risk Level:** 🔴 CRITICAL
   - **Exposure:** Anyone with repo access has full API billing authority
   - **Impact:** Account compromise, financial loss
   - **Recommended Fix:**
   ```python
   import os
   HF_TOKEN = os.getenv("HF_TOKEN")
   if not HF_TOKEN:
       raise ValueError("HF_TOKEN environment variable not set")
   ```

2. **No Input Validation Before API Call**
   - **Risk:** Potential for text injection or model exploitation
   - **Impact:** Misclassification or resource exhaustion
   - **Fix:** Sanitize inputs or validate length

3. **No Rate Limiting Per User**
   - **Risk:** Single runaway batch job blocks production traffic
   - **Fix:** Implement configurable delays or queue system

### Recommended Security Hardening

```python
import os
from dotenv import load_dotenv

# Load token from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Validate token exists
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not found in environment")

# Sanitize input length
MAX_TEXT_LENGTH = 512
def classify_sentimental(text: str) -> str:
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    # ... rest of function
```

---

## Configuration & Customization

### Adjusting Sentiment Triggers

Current logic (binary classification):
```python
if label in ['positive', 'negative']:
    return "SENTIMENTAL"
return "OTHER"
```

**Option 1: Separate Positive/Negative (Enhanced)**
```python
if label == 'positive':
    return "POSITIVE"
elif label == 'negative':
    return "NEGATIVE"
return "NEUTRAL"
```

**Option 2: Only Negative Sentiment**
```python
if label == 'negative':
    return "SENTIMENTAL"
return "OTHER"
```

### Changing Model

```python
# Use older version
model_id = "cardiffnlp/twitter-roberta-base-sentiment"

# Use multilingual RoBERTa
model_id = "xlm-roberta-base"

# Use DistilBERT (lighter)
model_id = "distilbert-base-uncased"
```

### Adjusting Rate Limiting

```python
time.sleep(0.2)  # Faster (risky; may hit rate limits)
time.sleep(1.0)  # Slower (safer; more conservative)
```

### Input File Customization

```python
# Change input filename
with open("tweets.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
```

### Output File Logging (Enhancement)

```python
import csv
from datetime import datetime

with open("sentiment_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Text", "Classification", "Timestamp"])
    writer.writerow([clean_text, result, datetime.now()])
```

---

## Usage Examples

### Single Text Classification
```python
result = classify_sentimental("Estou muito feliz hoje!")
print(result)  # Output: SENTIMENTAL (POSITIVE detected)
```

### Batch Processing (Recommended)
```python
if __name__ == "__main__":
    run_test()  # Processes all examples.txt lines
```

### Custom Text List
```python
texts = [
    "Amo este produto!",
    "O café é quente",
    "Que experiência terrível!"
]

for text in texts:
    result = classify_sentimental(text)
    print(f"{text} → {result}")
```

### With Error Handling
```python
try:
    result = classify_sentimental(user_input)
    if result == "SENTIMENTAL":
        print("✓ Emotional/opinionated content")
    elif result == "ERRO_API":
        print("❌ Classification failed")
    else:
        print("○ Neutral content")
except Exception as e:
    print(f"Error: {e}")
```

---

## Dependencies & Installation

### Required Packages
```bash
pip install huggingface-hub
```

### Version Requirements
- Python 3.7 or higher
- `huggingface-hub >= 0.16.0`

### Environment Setup

**Option 1: Environment Variable (Recommended)**
```bash
export HF_TOKEN="your_token_here"
```

**Option 2: .env File**
Create `.env` in project root:
```
HF_TOKEN=your_token_here
```
Then load in code:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Known Limitations & Issues

| Issue | Severity | Cause | Workaround |
|-------|----------|-------|-----------|
| No retry on API failure | Medium | Single request, no backoff | Add retry decorator |
| Fixed 0.5-second rate limit | Low | Hardcoded sleep | Make configurable parameter |
| Token hardcoded in source | **HIGH** | Security oversight | Use environment variables |
| No confidence scores returned | Low | Binary classification only | Request API confidence endpoint |
| File-dependent input | Medium | Hard-coded file path | Accept stdin or CLI arguments |
| No result persistence | Low | Console-only output | Log to CSV/database |
| Language-specific training | Low | Trained on Twitter English/Portuguese | May work poorly on other languages |
| No spam/bot detection | Low | RoBERTa focuses on sentiment only | Use different model for spam detection |

---

## Future Enhancements

1. **Confidence Scores** — Return probability (0-1) alongside classification
2. **Async Processing** — Use `asyncio` for parallel API calls
3. **Multi-Model Ensemble** — Combine RoBERTa + other sentiment models
4. **Emotion Detection** — Extend beyond sentiment (joy, anger, fear)
5. **Database Logging** — Store results in SQLite/PostgreSQL with metadata
6. **Batch API Endpoint** — Process multiple texts per request
7. **Language Detection** — Auto-detect language before classification
8. **Caching Layer** — Cache identical text classifications
9. **Web API Wrapper** — Expose as Flask/FastAPI endpoint
10. **Visualization Dashboard** — Chart sentiment trends over time

---

## Testing Recommendations

### Test Cases
```python
# Positive sentiment
classify_sentimental("Estou muito feliz hoje!")      # → SENTIMENTAL
classify_sentimental("Adorei este produto")          # → SENTIMENTAL

# Negative sentiment
classify_sentimental("Que dia horrível")             # → SENTIMENTAL
classify_sentimental("Odeio esperar em filas")       # → SENTIMENTAL

# Neutral (no sentiment)
classify_sentimental("O café é quente")              # → OTHER
classify_sentimental("A reunião é amanhã")           # → OTHER

# Edge cases
classify_sentimental("OK")                           # Skipped (<5 chars)
classify_sentimental("Não tenho opinião")            # → TEST
```

### Best Practices
- Test with balanced dataset (positive/negative/neutral samples)
- Manually review borderline cases
- Monitor API error rates
- Track classification accuracy against manual reviews
- Validate against domain knowledge (Twitter vs. formal text)

---

## References & Resources

- [Hugging Face Inference API Docs](https://huggingface.co/docs/hub/inference-api)
- [Twitter-RoBERTa Model Card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [Sentiment Analysis Benchmarks](https://huggingface.co/tasks/text-classification)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)

---

**Last Updated:** [Current Date]  
**Status:** Production-Ready (security hardening recommended)  
**Maintenance Notes:**
- Rotate API tokens every 90 days
- Monitor Twitter-RoBERTa model version updates
- Track classification accuracy against manual reviews
- Be aware of model bias toward social media language