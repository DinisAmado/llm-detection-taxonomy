# extremist.py — Content Moderation System

## Overview
An advanced content moderation system using DeepSeek-R1-Distill-Llama-70B to classify Portuguese/English text for extremist content. Leverages reasoning-based LLM inference for high-accuracy content filtering.

## System Architecture

### Dependencies
```python
from huggingface_hub import InferenceClient  # LLM API wrapper
import re                                     # Regex parsing
import time                                   # Rate limiting
```

### Model Configuration
```python
HF_TOKEN = "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"
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
```

**Why DeepSeek-R1?**
- Reasoning-capable model with `<think>` tags for internal reasoning
- Superior accuracy for nuanced content moderation
- Handles multilingual content (Portuguese + English)

### Inference Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.1 | Deterministic outputs; reduce variation |
| `max_tokens` | 20 | Account for reasoning blocks + response |
| `role` | system/user | Structured chat messages |

---

## Function Reference

### `classify_extremist(text: str) → str`

**Purpose:** Classify a single text for extremist content  
**Input:** String (Portuguese or English)  
**Output:** `"EXTREMIST"`, `"OTHER"`, or `"ERRO_API"`

**Process:**
1. Format message with system prompt (Portuguese instructions)
2. Send to DeepSeek API with reasoning enabled
3. Parse response, removing `<think>` reasoning blocks
4. Search for "EXTREMIST" keyword in cleaned response
5. Return classification or error code

**System Prompt Keywords (Extremist Triggers):**
- Ameaças de violência física (threats of physical violence)
- Incendiar propriedades (property destruction)
- Retaliação violenta (violent retaliation)
- Discurso de ódio (hate speech)
- Justiça pelas próprias mãos (vigilante justice)
- Explicit examples: "Set fire to the house", "No mercy"

**Response Cleaning:**
```regex
<think>.*?</think>  → removed via re.DOTALL
Result             → uppercased for matching
```

**Error Handling:**
- API timeout/failure → returns `"ERRO_API"`
- Ambiguous response → defaults to `"OTHER"`
- Malformed JSON → caught and logged

---

### `run_test() → None`

**Purpose:** Execute full moderation pipeline end-to-end  
**Input:** `examples.txt` file (required)  
**Output:** Formatted console report + statistics

**Execution Steps:**
1. Open `examples.txt` with UTF-8 encoding
2. Initialize stats dictionary
3. For each line:
   - Skip if empty or < 5 characters
   - Remove backslash escape characters
   - Call `classify_extremist()`
   - Record result in stats
   - Print formatted result
   - Sleep 2 seconds (rate limiting)
4. Display summary statistics
5. Handle FileNotFoundError gracefully

**Rate Limiting:**
- 2-second mandatory delay between API calls
- Prevents API throttling
- ~30 seconds per 10 samples

**Output Format:**
```
RESULTADO       | TEXTO
-----------     | ------
EXTREMIST       | [first 75 chars of text]...
OTHER           | [first 75 chars of text]...

==============================
RESUMO (EXTREMISMO):
Extremistas: X
Outros: Y
==============================
```

---

## Input/Output Specification

### Input Requirements
- **Source File:** `examples.txt` (must exist in working directory)
- **Encoding:** UTF-8
- **Format:** One text per line (newline-separated)
- **Language:** Portuguese, English, or mixed
- **Minimum Length:** 5 characters per line
- **Maximum Length:** No limit

### Input Validation
```python
if not line or len(line) < 5:
    continue  # Skip empty/minimal lines
```

### Output Format

**Per-Line Output:**
```
{RESULT:<15} | {TEXT[:75]}...
```
- Left-aligned result (15 chars)
- Text preview (75 character truncation)
- Visual separator for readability

**Summary Statistics:**
- Count of `EXTREMIST` classifications
- Count of `OTHER` classifications
- Implicit: `ERRO_API` failures (tracked but not displayed)

---

## Error Handling & Recovery

| Error Type | Trigger | Handling | Impact |
|------------|---------|----------|--------|
| **FileNotFoundError** | `examples.txt` missing | Print error, exit gracefully | No classifications run |
| **API Timeout** | Network/service issue | Return `"ERRO_API"`, continue | Single line classified as error |
| **Malformed Response** | Invalid JSON response | Catch exception, return `"ERRO_API"` | Counted in stats |
| **Empty Response** | Blank API response | Defaults to `"OTHER"` | False negative risk |
| **Parsing Failure** | Regex mismatch | Defaults to `"OTHER"` | Conservative fallback |

---

## Workflow Diagram

```
START
  │
  ├─→ Load examples.txt
  │    (UTF-8 encoded)
  │
  ├─→ FOR EACH line in file:
  │    │
  │    ├─→ Strip whitespace
  │    │
  │    ├─→ Skip if < 5 chars
  │    │
  │    ├─→ Remove backslashes
  │    │
  │    ├─→ Call DeepSeek API
  │    │    ├─ Send with system prompt
  │    │    └─ Receive reasoning + response
  │    │
  │    ├─→ Parse response
  │    │    ├─ Strip <think> blocks
  │    │    ├─ Uppercase text
  │    │    └─ Match keywords
  │    │
  │    ├─→ Classify & Update stats
  │    │    ├─ EXTREMIST or OTHER
  │    │    └─ Increment counter
  │    │
  │    ├─→ Print result
  │    │
  │    └─→ Sleep 2 seconds
  │
  ├─→ Print summary
  │    ├─ Extremist count
  │    ├─ Other count
  │    └─ Format table
  │
  └─→ END
```

---

## Performance Characteristics

### Computational Cost
- **Per-text latency:** 1-3 seconds (API + regex + sleep)
- **Throughput:** ~20 texts/minute (with 2s rate limit)
- **100 samples:** ~100-200 seconds (~2-3 minutes)
- **Bottleneck:** API rate limiting (2s mandatory delay)

### Token Usage
| Phase | Tokens |
|-------|--------|
| System prompt | ~60 tokens |
| User message | ~10 tokens (varies) |
| Thinking block | ~50-200 tokens (reasoning) |
| Response | ~5-10 tokens |
| **Total per call** | ~130-280 tokens |

### Network Requirements
- Persistent internet connection required
- Hugging Face Inference API accessibility
- Typical latency: 500ms - 2s per API call

---

## Notable Implementation Details

### Why `max_tokens=20`?
DeepSeek R1 outputs reasoning in `<think>` tags before the final classification. Setting `max_tokens=20` allows:
- Room for "EXTREMIST" or "OTHER" response (~2 tokens)
- Partial reasoning capture (helpful for debugging)
- Prevention of excessive token consumption

### Why `temperature=0.1`?
Low temperature ensures:
- Consistent, reproducible classifications
- Reduced hallucination risk
- Focused responses (no creative variations)

### The <think> Stripping
```python
answer = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip()
```
- **Why:** DeepSeek R1 shows internal reasoning; must be removed
- **Regex flags:** `re.DOTALL` allows `.` to match newlines
- **Result:** Only final classification remains

### Greedy String Matching
```python
if "EXTREMIST" in answer:
    return "EXTREMIST"
return "OTHER"
```
- Simple, fast substring search
- No regex complexity needed
- Risk: False positives if model mentions "EXTREMIST" in context

---

## Security Considerations ⚠️

### Critical Issues
1. **Hardcoded API Token Exposed** in source code
   - **Risk:** Anyone with repo access has full API billing authority
   - **Fix:** Use environment variables
   ```python
   import os
   HF_TOKEN = os.getenv("HF_TOKEN")
   ```

2. **No Input Validation** before API call
   - **Risk:** Prompt injection if text contains malicious payloads
   - **Fix:** Sanitize user input or use chat message formatting safely

3. **No Rate Limiting per User** (only global 2s)
   - **Risk:** One large batch job blocks other operations
   - **Fix:** Implement configurable delays + queue system

---

## Configuration & Customization

### Adjusting Classification Criteria
Edit the system prompt in `classify_extremist()`:
```python
"content": (
    "É um moderador de conteúdo. Classifica como:\n"
    "EXTREMIST: [your criteria]\n"
    "OTHER: [your criteria]\n"
)
```

### Changing Rate Limiting
```python
time.sleep(X)  # Modify sleep duration (seconds)
```

### Switching Models
```python
model_id = "meta-llama/Llama-2-70b-chat-hf"  # Alternative model
```

### Input File Customization
```python
with open("custom_file.txt", "r", encoding="utf-8") as f:
    # Change filename or encoding
```

---

## Usage Examples

### Single Text Classification
```python
result = classify_extremist("Set fire to the house")
print(result)  # Output: EXTREMIST
```

### Batch Processing
```python
if __name__ == "__main__":
    run_test()  # Processes all lines in examples.txt
```

### With Error Handling
```python
try:
    result = classify_extremist(user_input)
    if result == "EXTREMIST":
        print("Content flagged for review")
except Exception as e:
    print(f"Moderation failed: {e}")
```

---

## Dependencies & Installation

### Required Packages
```bash
pip install huggingface-hub
```

### Version Requirements
- Python 3.7+
- huggingface-hub >= 0.16.0

### Environment Setup
```bash
# Set Hugging Face token (recommended over hardcoding)
export HF_TOKEN="your_token_here"

# Or create .env file
# HF_TOKEN=your_token_here
```

---

## Limitations & Known Issues

| Issue | Severity | Workaround |
|-------|----------|-----------|
| No retry logic on API failures | Medium | Add try-except with retry decorator |
| Fixed 2-second rate limit | Low | Make configurable parameter |
| Token exposed in source | **HIGH** | Move to environment variables |
| No concurrent processing | Low | Use asyncio for parallel calls |
| File-dependent input | Medium | Add stdin/argument support |
| <think> tag regex assumes structure | Low | Add validation before stripping |

---

## Future Enhancements

1. **Async API Calls** — Parallel processing to reduce total runtime
2. **Confidence Scores** — Return moderation confidence (0-1)
3. **Explanation Generation** — Include why text was flagged
4. **Custom Thresholds** — Adjustable sensitivity levels
5. **Database Logging** — Store results in SQLite/PostgreSQL
6. **API Key Rotation** — Support multiple tokens with failover
7. **Batch API Endpoint** — Process multiple texts per request

---

## References

- [Hugging Face Inference API](https://huggingface.co/docs/hub/inference-api)
- [DeepSeek-R1 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
- [Content Moderation Best Practices](https://openai.com/research/techniques-for-improving-safety)

---

**Last Updated:** [Current Date]  
**Status:** Production-Ready (with security hardening recommended)  
**Maintenance:** Requires token rotation every 90 days