# radicalism.py — Political Radicalism Detection System

## Overview
A political sentiment analysis system using DeepSeek's R1-Distill-Llama-8B model to classify Portuguese text for radical political content. Leverages reasoning-based LLM inference to distinguish between mainstream political discourse and extremist appeals.

## System Architecture

### Dependencies
```python
from huggingface_hub import InferenceClient  # LLM API client
import re                                     # Text cleaning & parsing
import time                                   # Rate limiting
```

### Model Configuration
```python
HF_TOKEN = "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"
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
```

**Why DeepSeek-R1?**
- Reasoning-capable model with `<think>` tags for internal deliberation
- Superior accuracy for nuanced political analysis
- Handles multilingual content (Portuguese + English)
- Balanced model size (8B parameters) for fast inference
- Strong performance on policy/ideology classification

### Inference Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.1 | Deterministic outputs; minimal hallucination |
| `max_tokens` | 400 | Room for reasoning blocks + comprehensive response |
| `role` | system/user | Structured chat message format |

---

## Function Reference

### `classify_radicalism(text: str) → str`

**Purpose:** Classify a single text for political radicalism  
**Input:** Portuguese political statement (string)  
**Output:** `"RADICALISM"`, `"OTHER"`, or `"ERRO_API"`

**Process:**
1. Format system prompt with analysis instructions (Portuguese)
2. Embed user's text in message payload
3. Send to DeepSeek API with reasoning enabled
4. Receive response with potential `<think>` reasoning blocks
5. Parse response, removing reasoning blocks
6. Search for "RADICALISM" keyword in cleaned response
7. Return classification or error code

**System Prompt Definition (Portuguese):**
```
Role: "analista de segurança" (security analyst)

RADICALISM Classification Triggers:
- Apelos à derrubada de instituições
  (Appeals to overthrow institutions)
- Rejeição total de valores democráticos
  (Total rejection of democratic values)
- Extremismo político
  (Political extremism)

OTHER Classification:
- Opinião política comum (Common political opinion)
- Desabafo (Personal venting)
- Facto (Factual statements)

Output Format:
"Responde APENAS com a palavra 'RADICALISM' ou 'OTHER'"
(ONLY respond with the word 'RADICALISM' or 'OTHER')
```

**Response Cleaning & Parsing:**
```python
# Remove reasoning blocks (DeepSeek R1 feature)
answer = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL)

# Normalize case and whitespace
answer = answer.strip().upper()

# Match classification keyword
if "RADICALISM" in answer:
    return "RADICALISM"
return "OTHER"  # Conservative fallback
```

**Key Implementation Details:**
- `re.DOTALL` flag allows `.` to match newlines (spans entire thinking block)
- Uppercase conversion prevents case-sensitivity issues
- Substring matching (simple but effective)
- Default fallback to "OTHER" (conservative approach)

**Error Handling:**
```python
except Exception as e:
    print(f"\n[DEBUG] Erro: {e}")
    return "ERRO_API"
```
- Catches API failures, timeouts, malformed responses
- Prints debug message for troubleshooting
- Returns error code but continues batch processing

---

### `run_test() → None`

**Purpose:** Execute full radicalism detection pipeline end-to-end  
**Input:** `examples.txt` file (required)  
**Output:** Formatted console report + aggregate statistics

**Execution Flow:**
1. Initialize stats dictionary: `{"RADICALISM": 0, "OTHER": 0, "ERRO_API": 0}`
2. Open `examples.txt` with UTF-8 encoding
3. Print formatted table header
4. For each line in file:
   - Strip leading/trailing whitespace
   - Skip if empty or < 5 characters
   - Remove backslash escape sequences
   - Call `classify_radicalism()`
   - Increment corresponding stat counter
   - Print formatted result (classification + text preview)
   - Sleep 1 second (rate limiting)
5. Print summary statistics table
6. Handle FileNotFoundError gracefully

**Rate Limiting Strategy:**
```python
time.sleep(1.0)  # 1 second mandatory delay between calls
```
- Prevents API throttling/rate limit errors
- ~60 texts per minute processing speed
- Trade-off: Slow but reliable, no quota exhaustion

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
RESUMO (RADICALISMO):
Radicalismo: {stats['RADICALISM']}
Outros: {stats['OTHER']}
==============================
```

**Statistics Dictionary:**
```python
stats = {"RADICALISM": 0, "OTHER": 0, "ERRO_API": 0}
stats[result] = stats.get(result, 0) + 1  # Safe increment with fallback
```

---

## Input/Output Specification

### Input Requirements
- **Source File:** `examples.txt` (must exist in working directory)
- **Encoding:** UTF-8 (handles Portuguese accents: ç, ã, é, etc.)
- **Format:** One text per line (newline-separated)
- **Language:** Portuguese (optimized); English may work
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
- Removes backslash escape characters (handles CSV/JSON imports)
- Strips whitespace before and after
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
- Label: "RESUMO (RADICALISMO):"
- Counters: "Category: {count}" format
- Footer: "=" * 30 separator

### Example Output Mapping
```
Input:  "Devemos derrubar as instituições democráticas"
Output: "RADICALISM      | Devemos derrubar as instituições democráticas..."

Input:  "Não concordo com a política do governo"
Output: "OTHER           | Não concordo com a política do governo..."

Input:  "O presidente é incompetente"
Output: "OTHER           | O presidente é incompetente..."
```

---

## Error Handling & Recovery

| Error Type | Trigger | Handling | Impact |
|------------|---------|----------|--------|
| **FileNotFoundError** | `examples.txt` missing | Print error, exit gracefully | No classifications run |
| **API Connection Error** | Network timeout/service down | Return `"ERRO_API"`, continue | Single line skipped; batch continues |
| **Malformed JSON Response** | Invalid API response structure | Caught by try-except | Classification fails for that text |
| **Empty Response** | Blank API content | Defaults to `"OTHER"` | False negative (miss radicalism) |
| **Authentication Failure** | Invalid HF_TOKEN | API raises 401 error | All classifications fail |
| **Regex Parsing Failure** | <think> tag structure mismatch | Substring still matches | May include thinking text in analysis |
| **Non-UTF-8 File** | Invalid encoding in examples.txt | UnicodeDecodeError on read | Script crashes (no fallback encoding) |

**Resilience:** Script continues processing after individual line failures; one error doesn't halt entire batch.

---

## Workflow Diagram

```
START
  │
  ├─→ Load examples.txt
  │    (UTF-8 encoding)
  │
  ├─→ Initialize: stats = {RADICALISM: 0, OTHER: 0, ERRO_API: 0}
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
  │    ├─→ Call classify_radicalism(clean_text)
  │    │    │
  │    │    ├─→ Format system prompt (Portuguese)
  │    │    │
  │    │    ├─→ Call DeepSeek API
  │    │    │    ├─ temperature=0.1
  │    │    │    ├─ max_tokens=400
  │    │    │    └─ Return response
  │    │    │
  │    │    ├─→ Parse response
  │    │    │    ├─ Strip <think>...</think> blocks
  │    │    │    ├─ Uppercase text
  │    │    │    └─ Match "RADICALISM" substring
  │    │    │
  │    │    ├─→ Classification logic
  │    │    │    ├─ "RADICALISM" found → return "RADICALISM"
  │    │    │    └─ Otherwise → return "OTHER"
  │    │    │
  │    │    ├─→ [Exception] → return "ERRO_API"
  │    │    │
  │    │    └─→ Return result
  │    │
  │    ├─→ stats[result] += 1
  │    │
  │    ├─→ Print: "{result:<15} | {text[:75]}..."
  │    │
  │    └─→ time.sleep(1.0)
  │
  ├─→ Print separator: "=" * 30
  │
  ├─→ Print "RESUMO (RADICALISMO):"
  │    ├─ Radicalismo: {stats['RADICALISM']}
  │    ├─ Outros: {stats['OTHER']}
  │    └─ Separator: "=" * 30
  │
  └─→ END / [FileNotFoundError] → Print error, exit
```

---

## Performance Characteristics

### Computational Cost
- **Per-text latency:** 1-3 seconds (API call + parsing + sleep)
- **Throughput:** ~60 texts/minute (with 1s mandatory rate limit)
- **100 samples:** ~100-120 seconds (~2 minutes)
- **Bottleneck:** API rate limiting (1s mandatory delay)

### Token Usage
| Phase | Tokens |
|-------|--------|
| System prompt | ~50 tokens |
| User message | ~10-20 tokens (text-dependent) |
| Thinking block | ~100-300 tokens (reasoning) |
| Response | ~2-5 tokens ("RADICALISM" or "OTHER") |
| **Total per call** | ~160-375 tokens |

### Network Requirements
- Persistent internet connection (API-dependent)
- Hugging Face Inference API accessibility
- Typical API latency: 500ms - 2s per call
- Token consumption: ~160-375 tokens/call

---

## Notable Implementation Details

### Why `max_tokens=400`?

DeepSeek R1 outputs reasoning in `<think>` tags before classification. Setting `max_tokens=400`:
- Allows full reasoning block capture (helpful for debugging)
- Room for response word ("RADICALISM" or "OTHER")
- Prevents token overflow for longer texts
- Balances cost vs. transparency

### Why `temperature=0.1`?

Low temperature ensures:
- Consistent, reproducible classifications
- Reduced hallucination risk
- Focused, confident predictions
- Deterministic behavior (identical input → identical output)

### The <think> Stripping
```python
answer = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL)
```
- **Why:** DeepSeek R1 shows internal reasoning; must be removed for clean analysis
- **Regex pattern:** Matches opening/closing think tags with any content between
- **DOTALL flag:** Allows `.` to match newlines (spans entire reasoning block)
- **Result:** Only final classification remains for keyword matching

### Conservative Fallback
```python
if "RADICALISM" in answer:
    return "RADICALISM"
return "OTHER"  # Default to safer classification
```
- Any unmatched response defaults to "OTHER"
- Reduces false positives (safer approach)
- Risk: False negatives if model doesn't output exact keyword

---

## Security Considerations ⚠️

### Critical Issues

1. **Hardcoded API Token Exposed in Source Code**
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
   - **Risk:** Prompt injection if text contains hidden instructions
   - **Impact:** Potential model manipulation or misclassification
   - **Fix:** Sanitize inputs or use safer message formatting

3. **No Rate Limiting Per User**
   - **Risk:** Single runaway batch job blocks all other processing
   - **Fix:** Implement configurable delays or queue system

4. **No Audit Logging**
   - **Risk:** Cannot track what texts were classified or when
   - **Fix:** Add CSV/database logging with timestamps

### Recommended Security Hardening

```python
# Better token management
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not configured")

# Input validation
def sanitize_text(text: str) -> str:
    """Remove potentially harmful characters"""
    return text.replace("'", "").replace('"', '')

# Audit logging
import csv
from datetime import datetime

def log_classification(text: str, result: str):
    with open("audit.log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), text, result])
```

---

## Configuration & Customization

### Adjusting Classification Criteria

Edit the system prompt in `classify_radicalism()`:
```python
"content": (
    "Classificar como:\n"
    "RADICALISM: [your custom criteria]\n"
    "OTHER: [your custom criteria]"
)
```

**Example: Stricter Detection**
```python
"RADICALISM: Qualquer apelo para mudança violenta de governo"
"OTHER: Crítica política comum"
```

### Changing Model

```python
# Switch to lighter model
model_id = "meta-llama/Llama-2-7b-chat"

# Switch to larger model
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
```

### Adjusting Rate Limiting

```python
time.sleep(2.0)  # Increase to 2 seconds
time.sleep(0.5)  # Decrease (risky; may hit rate limits)
```

### Input File Customization

```python
# Change input filename
with open("political_texts.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Change encoding
with open("examples.txt", "r", encoding="iso-8859-1") as f:
    lines = f.readlines()
```

### Output File Logging (Enhancement)

```python
# Save results to CSV
import csv

with open("radicalism_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Text", "Classification", "Timestamp"])
    for text in texts:
        result = classify_radicalism(text)
        writer.writerow([text, result, datetime.now()])
```

---

## Usage Examples

### Single Text Classification
```python
result = classify_radicalism("Devemos derrubar todas as instituições")
print(result)  # Output: RADICALISM
```

### Batch Processing (Recommended)
```python
if __name__ == "__main__":
    run_test()  # Processes all examples.txt lines
```

### Custom Text List
```python
texts = [
    "Rejeitar completamente a democracia",
    "Discordo com a política atual",
    "O governo deve ser derrubado violentamente"
]

for text in texts:
    result = classify_radicalism(text)
    print(f"{text} → {result}")
```

### With Error Handling
```python
try:
    result = classify_radicalism(user_input)
    if result == "RADICALISM":
        print("⚠️ Political radicalism detected")
    elif result == "ERRO_API":
        print("❌ Classification failed")
    else:
        print("✓ Non-radical political content")
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

**Option 3: Hugging Face CLI**
```bash
huggingface-cli login
# Automatically configured in ~/.cache/huggingface
```

---

## Known Limitations & Issues

| Issue | Severity | Cause | Workaround |
|-------|----------|-------|-----------|
| No retry on API failure | Medium | Single request, no backoff | Add retry decorator with exponential backoff |
| Fixed 1-second rate limit | Low | Hardcoded sleep | Make configurable `sleep_delay` parameter |
| Token hardcoded in source | **HIGH** | Security oversight | Use environment variables |
| No confidence scores | Medium | Binary classification only | Request API to return probability |
| File-dependent input | Medium | Hard-coded file path | Accept stdin or CLI arguments |
| No result persistence | Low | Console-only output | Log to CSV or database |
| Greedy string matching | Low | Simple substring search | Use exact word boundary matching |
| Single model mode | Low | No model switching | Add model selection parameter |

---

## Future Enhancements

1. **Confidence Scores** — Return probability (0-1) alongside classification
2. **Async Processing** — Use `asyncio` for parallel API calls
3. **Severity Levels** — Distinguish light vs. extreme radicalism
4. **Explanation Generation** — Include reasoning from <think> blocks
5. **Database Logging** — Store results in SQLite/PostgreSQL with metadata
6. **Batch API Endpoint** — Process multiple texts per request
7. **Language Detection** — Auto-detect language before classification
8. **Caching Layer** — Cache identical text classifications
9. **Configurable Thresholds** — Adjust sensitivity via parameters
10. **Web API Wrapper** — Expose as FastAPI/Flask endpoint for integration

---

## Testing Recommendations

### Test Cases
```python
# Positive (RADICALISM)
classify_radicalism("Devemos derrubar todas as instituições democráticas")
classify_radicalism("Rejeitar completamente os valores ocidentais")

# Negative (OTHER)
classify_radicalism("Não concordo com a política do governo")
classify_radicalism("O presidente é incompetente")

# Edge Cases
classify_radicalism("Gosto de mudança política")  # Ambiguous?
classify_radicalism("Revolução pacífica é necessária")  # Peaceful revolution?
```

### Best Practices
- Test with balanced dataset (radical + non-radical samples)
- Manually review borderline cases
- Monitor API error rates and token usage
- Track false positive/negative rates periodically
- Validate against domain experts (political analysts)

---

## References & Resources

- [Hugging Face Inference API Docs](https://huggingface.co/docs/hub/inference-api)
- [DeepSeek-R1 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- [Political Extremism Detection Research](https://arxiv.org/abs/2101.08110)
- [LLM Safety & Content Moderation](https://openai.com/research/techniques-for-improving-safety)

---

**Last Updated:** [Current Date]  
**Status:** Production-Ready (security hardening recommended)  
**Maintenance Notes:**
- Rotate API tokens every 90 days
- Monitor DeepSeek model version updates
- Track classification accuracy against manual reviews
- Review false positive/negative patterns monthly