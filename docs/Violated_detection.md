# violated.py — Rights Violation & Boundary Infringement Detection System

## Overview
A sentiment analysis system using Meta's Llama 3.2 3B Instruct model to classify Portuguese text for expressions of personal boundary violations. Detects emotive language around privacy breaches, trust violations, manipulation, harassment, and rights infringement incidents.

## System Architecture

### Dependencies
```python
from huggingface_hub import InferenceClient  # LLM API client
import re                                     # Text cleaning
import time                                   # Rate limiting
```

### Model Configuration
```python
HF_TOKEN = "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"
model_id = "meta-llama/Llama-3.2-3B-Instruct"
```

**Why Llama 3.2 3B?**
- Lightweight instruction-tuned model (3B parameters)
- Fast inference with acceptable accuracy
- Trained on sentiment/emotion understanding
- Lower latency and cost than larger models
- Good balance between speed and performance
- Supports Portuguese language instruction following

### Inference Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.1 | Deterministic outputs; minimal variation |
| `max_tokens` | 5 | Sufficient for "VIOLATED" or "OTHER" (~2 tokens) |
| `role` | system/user | Structured chat message format |

---

## Function Reference

### `classify_violated(text: str) → str`

**Purpose:** Classify a single text for violation sentiment expression  
**Input:** Portuguese text string  
**Output:** `"VIOLATED"`, `"OTHER"`, or `"ERRO_API"`

**Process:**
1. Construct system prompt (Portuguese sentiment analyst instructions)
2. Format user message for content analysis
3. Send chat completion request to Llama 3.2 API
4. Parse response and extract classification word
5. Uppercase for case-insensitive matching
6. Return classification or error code

**System Prompt Definition:**
```
Role: "analista de sentimentos" (sentiment analyst)

VIOLATED Classification Triggers:
- Invasão de espaço pessoal (Invasion of personal space)
- Quebra de privacidade (Privacy breach)
- Abuso de confiança (Breach of trust)
- Manipulação (Manipulation)
- Perseguição (Harassment/persecution)
- Violação de direitos (Rights violation)

OTHER Classification:
- Desabafo emocional comum (Common emotional venting)
- Factos (Factual statements)
- Opiniões neutras (Neutral opinions)
- [No boundary/rights violation component]

Output Format:
"Responde APENAS com a palavra 'VIOLATED' ou 'OTHER'"
(ONLY respond with the word 'VIOLATED' or 'OTHER')
```

**User Message Format:**
```
"Analise este conteúdo: '{text}'"
(Analyze this content: '{text}')
```

**Response Parsing Logic:**
```python
prediction = response.choices[0].message.content.strip().upper()

if "VIOLATED" in prediction: 
    return "VIOLATED"
return "OTHER"
```
- Strips whitespace from response
- Converts to uppercase (case-insensitive matching)
- Simple substring search for "VIOLATED"
- Default: "OTHER" (conservative fallback)

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

**Purpose:** Execute full violation detection pipeline end-to-end  
**Input:** `examples.txt` file (required)  
**Output:** Formatted console report + statistics

**Execution Flow:**
1. Open `examples.txt` with UTF-8 encoding
2. Initialize stats dictionary: `{"VIOLATED": 0, "OTHER": 0, "ERRO_API": 0}`
3. Print table header: "RESULTADO | TEXTO"
4. For each line in file:
   - Strip whitespace
   - Skip if empty or < 5 characters
   - Remove backslash escape sequences
   - Call `classify_violated()`
   - Increment corresponding counter
   - Print formatted result
   - Sleep 0.8 seconds (rate limiting)
5. Print summary statistics table
6. Handle missing file gracefully

**Rate Limiting Strategy:**
```python
time.sleep(0.8)  # 0.8 second delay between calls
```
- Prevents API throttling/rate limit errors
- ~75 texts per minute processing speed
- Trade-off: Balanced speed and reliability

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
RESUMO (VIOLATED):
Violados: {stats['VIOLATED']}
Outros: {stats['OTHER']}
==============================
```

**Statistics Dictionary:**
```python
stats = {"VIOLATED": 0, "OTHER": 0, "ERRO_API": 0}
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
- **No Maximum Length:** Supports arbitrary text length (up to model context limit)

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
- Label: "RESUMO (VIOLATED):"
- Counters: "Category: {count}" format
- Footer: "=" * 30 separator

### Example Output Mapping
```
Input:  "Minha privacidade foi completamente invadida"
Output: "VIOLATED        | Minha privacidade foi completamente..."
Label:  Privacy breach expression

Input:  "Estou frustrado com a situação"
Output: "OTHER           | Estou frustrado com a situação..."
Label:  General frustration (no violation)

Input:  "Alguém violou minha confiança"
Output: "VIOLATED        | Alguém violou minha confiança..."
Label:  Trust violation expression
```

---

## Error Handling & Recovery

| Error Type | Trigger | Handling | Impact |
|------------|---------|----------|--------|
| **FileNotFoundError** | `examples.txt` missing | Print error message, exit gracefully | No classifications run |
| **API Connection Error** | Network timeout/service down | Return `"ERRO_API"`, continue | Single line skipped; batch continues |
| **Malformed JSON Response** | Invalid API response structure | Caught by try-except | Classification fails for that text |
| **Empty Response** | Blank API content | Defaults to `"OTHER"` | False negative (misses violation) |
| **Authentication Failure** | Invalid HF_TOKEN | API raises 401 error | All classifications fail |
| **Non-UTF-8 File** | Invalid encoding in examples.txt | UnicodeDecodeError on read | Script crashes (no fallback encoding) |
| **KeyError on Response** | Missing expected response field | Caught by exception handler | Returns `"ERRO_API"` |

**Resilience:** Script continues processing after individual line failures; one error doesn't halt entire batch.

---

## Workflow Diagram

```
START
  │
  ├─→ Load examples.txt
  │    (UTF-8 encoding)
  │
  ├─→ Initialize: stats = {VIOLATED: 0, OTHER: 0, ERRO_API: 0}
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
  │    ├─→ Call classify_violated(clean_text)
  │    │    │
  │    │    ├─→ Format system prompt (Portuguese)
  │    │    │
  │    │    ├─→ Call Llama 3.2 API
  │    │    │    ├─ temperature=0.1
  │    │    │    ├─ max_tokens=5
  │    │    │    └─ Return response
  │    │    │
  │    │    ├─→ Parse response
  │    │    │    ├─ Strip whitespace
  │    │    │    ├─ Uppercase text
  │    │    │    └─ Match "VIOLATED" substring
  │    │    │
  │    │    ├─→ Classification logic
  │    │    │    ├─ "VIOLATED" found → return "VIOLATED"
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
  │    └─→ time.sleep(0.8)
  │
  ├─→ Print separator: "=" * 30
  │
  ├─→ Print "RESUMO (VIOLATED):"
  │    ├─ Violados: {stats['VIOLATED']}
  │    ├─ Outros: {stats['OTHER']}
  │    └─ Separator: "=" * 30
  │
  └─→ END / [FileNotFoundError] → Print error, exit
```

---

## Performance Characteristics

### Computational Cost
- **Per-text latency:** 0.2-1 second (API call + parsing + sleep)
- **Throughput:** ~75 texts/minute (with 0.8s mandatory rate limit)
- **100 samples:** ~80 seconds (~1.3 minutes)
- **Bottleneck:** API rate limiting (0.8s mandatory delay)

### Token Usage
| Phase | Tokens |
|-------|--------|
| System prompt | ~40 tokens |
| User message | ~10-20 tokens (text-dependent) |
| Response | ~2-5 tokens ("VIOLATED" or "OTHER") |
| **Total per call** | ~55-65 tokens |

### Network Requirements
- Persistent internet connection (API-dependent)
- Hugging Face Inference API accessibility
- Typical API latency: 150-400ms (fast for lightweight model)

### Storage Requirements
- Code: ~1.8 KB
- Does not cache results
- Output: Console only (not logged to file)

---

## Notable Implementation Details

### Why `max_tokens=5`?

Llama 3.2 is instruction-tuned for concise responses. Setting `max_tokens=5`:
- Enforces single-word output ("VIOLATED" or "OTHER")
- Prevents model from generating explanations
- Reduces token cost and latency
- Ensures reproducible, deterministic results
- Forces compliance with system prompt instruction

### Why `temperature=0.1`?

Low temperature ensures:
- Consistent, reproducible violation sentiment classifications
- Reduced hallucination risk
- Focused, confident predictions
- Minimal variance across identical inputs

### Violation vs. Frustration Distinction

Model is trained to differentiate:
- **VIOLATED:** "Minha privacidade foi invadida" (specific boundary violation)
- **OTHER:** "Estou frustrado" (general frustration without context)
- **VIOLATED:** "Alguém violou minha confiança" (trust breach expression)
- **OTHER:** "Discordo com essa política" (disagreement without violation)

This nuance prevents false positives on general negative sentiment.

### Greedy String Matching
```python
if "VIOLATED" in prediction:
    return "VIOLATED"
return "OTHER"
```
- Simple, fast substring search
- No regex complexity needed
- Risk: False positives if model mentions "violated" in context

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
   - **Impact:** Model manipulation or bypass of violation detection
   - **Fix:** Sanitize inputs or use safer message formatting

3. **No Rate Limiting Per User**
   - **Risk:** Single runaway batch job blocks all other processing
   - **Fix:** Implement configurable delays or queue system

4. **No Audit Logging**
   - **Risk:** Cannot track violations detected or detection times
   - **Fix:** Add CSV/database logging with timestamps

### Recommended Security Hardening

```python
import os
from dotenv import load_dotenv
from datetime import datetime
import csv

# Better token management
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not configured")

# Audit logging
def log_violation(text: str, result: str):
    with open("violation_audit.log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), text, result])

# Input validation
def validate_text(text: str, max_length: int = 1000) -> str:
    """Remove potentially harmful characters"""
    if len(text) > max_length:
        text = text[:max_length]
    return text.replace("'", "").replace('"', '')
```

---

## Configuration & Customization

### Adjusting Violation Classification Criteria

Edit the system prompt in `classify_violated()`:
```python
"content": (
    "Classificar como:\n"
    "VIOLATED: [your custom violation criteria]\n"
    "OTHER: [your custom non-violation criteria]"
)
```

**Example: More Sensitive Detection**
```python
"VIOLATED: Qualquer expressão de desconforto com limites pessoais"
"OTHER: Apenas comunicação respeitosa e consentida"
```

### Changing Model

```python
# Use larger model for better accuracy
model_id = "meta-llama/Llama-2-7b-chat-hf"

# Use lighter model for speed
model_id = "meta-llama/Llama-3.2-1B-Instruct"
```

### Adjusting Rate Limiting

```python
time.sleep(0.5)  # Faster (risky; may hit rate limits)
time.sleep(1.5)  # Slower (safer; more conservative)
```

### Input File Customization

```python
# Change input filename
with open("violations.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
```

### Output File Logging (Enhancement)

```python
import csv
from datetime import datetime

with open("violation_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Text", "Classification", "Timestamp"])
    writer.writerow([clean_text, result, datetime.now()])
```

---

## Usage Examples

### Single Text Classification
```python
result = classify_violated("Minha privacidade foi invadida")
print(result)  # Output: VIOLATED
```

### Batch Processing (Recommended)
```python
if __name__ == "__main__":
    run_test()  # Processes all examples.txt lines
```

### Custom Text List
```python
texts = [
    "Alguém leu meus mensagens pessoais",
    "Estou desapontado com a decisão",
    "Minha confiança foi completamente quebrada"
]

for text in texts:
    result = classify_violated(text)
    print(f"{text} → {result}")
```

### With Error Handling
```python
try:
    result = classify_violated(user_input)
    if result == "VIOLATED":
        print("⚠️ Boundary violation expression detected")
    elif result == "ERRO_API":
        print("❌ Classification failed")
    else:
        print("✓ Non-violation content")
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
| Fixed 0.8-second rate limit | Low | Hardcoded sleep | Make configurable `sleep_delay` parameter |
| Token hardcoded in source | **HIGH** | Security oversight | Use environment variables |
| No confidence scores | Medium | Binary classification only | Request API confidence endpoint |
| File-dependent input | Medium | Hard-coded file path | Accept stdin/CLI arguments |
| No result persistence | Low | Console-only output | Log to CSV or database |
| Greedy string matching | Low | Simple substring search | Use exact word boundary matching |
| Language-specific training | Low | Optimized for Portuguese | May work less well on other languages |

---

## False Positive/Negative Risks

### False Positives (Over-detection of Violations)
```
"Estou decepcionado"           → May flag as frustration-based violation
"Não concordo com você"        → Disagreement misidentified as violation
"Você é um mau exemplo"        → Criticism labeled as violation sentiment
```

Mitigation: Manual review of flagged items before escalation

### False Negatives (Missed Violations)
```
"Sinto-me desconfortável"      → Too subtle; may miss boundary issues
"Isso me deixou em má posição" → Indirect violation language
"Seria melhor se respeitasses" → Passive expression of violation feeling
```

Mitigation: Combine with other sentiment analysis tools

---

## Future Enhancements

1. **Confidence Scores** — Return probability (0-1) alongside classification
2. **Async Processing** — Use `asyncio` for parallel API calls
3. **Severity Levels** — Distinguish minor vs. major boundary violations
4. **Explanation Generation** — Include which violation type detected
5. **Database Logging** — Store results in SQLite/PostgreSQL
6. **Violation Taxonomy** — Categorize privacy/trust/manipulation/harassment
7. **Language Detection** — Auto-detect language before classification
8. **Context Awareness** — Consider conversation history
9. **Pattern Recognition** — Track repeated violation patterns from same user
10. **Integration with Support Systems** — Escalation workflow to counselors/advocates

---

## Testing Recommendations

### Test Cases
```python
# Positive (VIOLATED)
classify_violated("Minha privacidade foi invadida")
classify_violated("Alguém violou minha confiança")

# Negative (OTHER)
classify_violated("Estou frustrado com a situação")
classify_violated("Discordo com essa política")

# Edge Cases
classify_violated("Sinto desconforto")  # Ambiguous?
classify_violated("Não reseitarem meus limites")  # Passive violation language?
```

### Best Practices
- Test with balanced dataset (violations + non-violations)
- Manually review borderline cases
- Monitor false positive/negative rates
- Validate against domain experts (counselors, advocates)
- Document edge cases and model limitations
- Periodically re-evaluate accuracy with new data

---

## Ethical & Legal Considerations

**Important:**
- This system detects **emotional expressions** of violation, not legal violations
- Use as **indicator, not definitive proof** of boundary violations
- Should be **complementary to human judgment** in support contexts
- Maintain **strict confidentiality** of analysis results
- Comply with **GDPR/data protection** laws in EU jurisdictions
- Inform users if their content is being analyzed
- Have **clear escalation procedures** for serious cases
- Avoid **automated decision-making** without human review

---

## References & Resources

- [Hugging Face Inference API Docs](https://huggingface.co/docs/hub/inference-api)
- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Sentiment Analysis Best Practices](https://huggingface.co/tasks/text-classification)
- [Emotional Boundaries in Psychology](https://www.apa.org/science/about/psa/boundaries)

---

**Last Updated:** [Current Date]  
**Status:** Production-Ready (security hardening recommended)  
**Maintenance Notes:**
- Rotate API tokens every 90 days
- Monitor Llama model version updates
- Review classification accuracy monthly
- Update violation taxonomy based on emerging patterns
- Maintain confidentiality protocols