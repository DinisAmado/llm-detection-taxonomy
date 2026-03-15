## Sentiment Classification Module

- File: emotional.py

## Overview
A Portuguese text classification system using Hugging Face's Inference API and Meta's Llama 3 8B model to categorize text as emotionally-charged or neutral.

## System Architecture

### Dependencies
- `huggingface_hub.InferenceClient` — LLM API client
- `re` — Regular expression for text cleaning
- `time` — Rate limiting between requests

### Configuration
```python
HF_TOKEN = "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"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
```

### API Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `temperature` | 0.1 | Deterministic output |
| `max_tokens` | 5 | Response token limit |
| `role` | system/user | Chat message structure |

## Function Reference

### `classify_text(text: str) → str`
**Purpose:** Classify a single text snippet  
**Input:** Portuguese text string  
**Output:** One of `"EMOTIONAL"`, `"OTHER"`, or `"ERRO_API"`  
**Fallback:** Returns `"OTHER"` on ambiguous responses

**Prompt Engineering:**
- Bilingual sentiment analyzer (Portuguese/English)
- Clear category definitions
- Constrained output (single word only)

### `run_test() → None`
**Purpose:** Execute full classification pipeline  
**Input:** Reads from `examples.txt`  
**Output:** Console report + statistics

**Process:**
1. Load examples.txt with UTF-8 encoding
2. Skip empty or too-short lines
3. Clean backslashes from text
4. Call API for each line
5. Print formatted results
6. Output summary statistics

**Rate Limiting:** 1-second delay between API calls

## Input/Output Specification

### Input Requirements
- **File:** `examples.txt` (must exist in working directory)
- **Format:** One text per line (UTF-8 encoded)
- **Language:** Portuguese
- **Minimum length:** 5 characters

### Output Format
```
RESULTADO       | TEXTO
-----------------------------------
EMOTIONAL       | Sample text...
OTHER           | Another sample...

==============================
RESUMO DO TESTE:
Emocionais: X
Outros: Y
==============================
```

## Error Handling

| Error | Handling |
|-------|----------|
| File Not Found | Prints error message, exits gracefully |
| API Failure | Returns `"ERRO_API"`, continues processing |
| Ambiguous Response | Defaults to `"OTHER"` classification |

## Workflow Diagram

```
Start
  ↓
Load examples.txt
  ↓
For each line:
  ├─ Skip if empty/too short
  ├─ Clean text (remove \)
  ├─ Call LLM API
  ├─ Parse response
  ├─ Increment stats counter
  └─ Sleep 1 second
  ↓
Print all results
  ↓
Display summary stats
  ↓
End
```

## Performance Characteristics

- **Per-request latency:** ~1-2 seconds (including 1s mandatory delay)
- **100 samples:** ~100-200 seconds (~2-3 minutes)
- **API calls:** 1 per text (no caching)
- **Token usage:** ~17 prompt tokens + ~1 completion token per call

## Notable Implementation Details

### Gotchas ⚠️
1. **Hardcoded token in source code** — Security risk; should use environment variables
2. **No retry logic** — Failed API calls immediately return `"ERRO_API"`
3. **Rate limiting is fixed** — Cannot be adjusted per run
4. **File must exist** — No automatic creation or fallback
5. **String matching is greedy** — Any substring match counts (e.g., "EMOTIONAL" found anywhere)

### Improvements Suggested
```python
# Better token handling
import os
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Configurable rate limiting
def run_test(delay: float = 1.0):
    time.sleep(delay)

# Stricter output validation
if prediction.strip().upper() in ["EMOTIONAL", "OTHER"]:
    return prediction.strip().upper()
```

## Usage Example

```python
# Classify a single text
result = classify_text("Que dia lindo!")
print(result)  # Output: EMOTIONAL or OTHER

# Run full test
run_test()
```

## Dependencies Installation

```bash
pip install huggingface-hub
```

## Configuration Requirements

1. Valid Hugging Face API token
2. `examples.txt` file in working directory
3. Network access to Hugging Face API
4. ~1-2 minutes per 100 samples

---

**Last Updated:** [Current Date]  
**Status:** Production-Ready (with security considerations)