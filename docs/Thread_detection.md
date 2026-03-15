# threat.py — Threat Detection & Security Moderation System

## Overview
A security-focused threat detection system using Meta's Llama Guard 3 8B model to classify Portuguese text for violent threats. Identifies physical threat language including violence promises, attack plans, and property destruction intentions.

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
model_id = "meta-llama/Llama-Guard-3-8B"
```

**Why Llama Guard 3?**
- Purpose-built threat/safety detection model
- Trained on security benchmarks and threat taxonomy
- Specialized in distinguishing expressions of anger from credible threats
- Lower false positive rate vs. general LLMs
- Minimal hallucination risk
- 8B parameters = fast inference with acceptable latency

### Inference Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.1 | Deterministic outputs; minimal variation |
| `max_tokens` | 5 | Sufficient for "THREAT" or "OTHER" (~2 tokens) |
| `role` | system/user | Structured chat message format |

---

## Function Reference

### `classify_threat(text: str) → str`

**Purpose:** Classify a single text for credible physical threats  
**Input:** Portuguese text string  
**Output:** `"THREAT"`, `"OTHER"`, or `"ERRO_API"`

**Process:**
1. Construct system prompt (Portuguese security analyst instructions)
2. Format user message asking for threat verification
3. Send chat completion request to Llama Guard API
4. Parse response and extract classification word
5. Uppercase for case-insensitive matching
6. Return classification or error code

**System Prompt Definition:**
```
Role: "especialista em segurança e moderação"
(Security and moderation specialist)

THREAT Classification Triggers:
- Ameaças de violência física
  (Threats of physical violence)
- Planos de ataque
  (Attack plans)
- Intenção de causar danos a propriedades
  (Intention to cause property damage)
- Intenção de causar danos a pessoas
  (Intention to cause harm to people)

OTHER Classification:
- Desabafo (Venting/emotional expression)
- Crítica (Criticism)
- Facto (Factual statements)
- [No physical threat component]

Output Format:
"Responde APENAS com a palavra 'THREAT' ou 'OTHER'"
(ONLY respond with the word 'THREAT' or 'OTHER')
```

**User Message Format:**
```
"Verifica se este texto é uma ameaça: '{text}'"
(Verify if this text is a threat: '{text}')
```

**Response Parsing Logic:**
```python
prediction = response.choices[0].message.content.strip().upper()

if "THREAT" in prediction: 
    return "THREAT"
return "OTHER"
```
- Strips whitespace from response
- Converts to uppercase (case-insensitive matching)
- Simple substring search for "THREAT"
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

**Purpose:** Execute full threat detection pipeline end-to-end  
**Input:** `examples.txt` file (required)  
**Output:** Formatted console report + statistics

**Execution Flow:**
1. Open `examples.txt` with UTF-8 encoding
2. Initialize stats dictionary: `{"THREAT": 0, "OTHER": 0, "ERRO_API": 0}`
3. Print table header: "RESULTADO | TEXTO"
4. For each line in file:
   - Strip whitespace
   - Skip if empty or < 5 characters
   - Remove backslash escape sequences
   - Call `classify_threat()`
   - Increment corresponding counter
   - Print formatted result
   - Sleep 1 second (rate limiting)
5. Print summary statistics table
6. Handle missing file gracefully

**Rate Limiting Strategy:**
```python
time.sleep(1.0)  # 1 second mandatory delay
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
RESUMO DO TESTE (THREAT):
Ameaças: {stats['THREAT']}
Outros: {stats['OTHER']}
==============================
```

**Statistics Dictionary:**
```python
stats = {"THREAT": 0, "OTHER": 0, "ERRO_API": 0}
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
- Label: "RESUMO DO TESTE (THREAT):"
- Counters: "Category: {count}" format
- Footer: "=" * 30 separator

### Example Output Mapping
```
Input:  "Vou quebrar seu carro amanhã"
Output: "THREAT          | Vou quebrar seu carro amanhã..."
Label:  Specific threat to property

Input:  "Estou muito zangado com você"
Output: "OTHER           | Estou muito zangado com você..."
Label:  Expression of anger (no threat)

Input:  "Espero que você mude de atitude"
Output: "OTHER           | Espero que você mude de atitude..."
Label:  Factual statement/hope (no threat)
```

---

## Error Handling & Recovery

| Error Type | Trigger | Handling | Impact |
|------------|---------|----------|--------|
| **FileNotFoundError** | `examples.txt` missing | Print error message, exit gracefully | No classifications run |
| **API Connection Error** | Network timeout/service down | Return `"ERRO_API"`, continue | Single line skipped; batch continues |
| **Malformed JSON Response** | Invalid API response structure | Caught by try-except | Classification fails for that text |
| **Empty Response** | Blank API content | Defaults to `"OTHER"` | False negative (misses threats) |
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
  ├─→ Initialize: stats = {THREAT: 0, OTHER: 0, ERRO_API: 0}
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
  │    ├─→ Call classify_threat(clean_text)
  │    │    │
  │    │    ├─→ Format system prompt (Portuguese)
  │    │    │
  │    │    ├─→ Call Llama Guard API
  │    │    │    ├─ temperature=0.1
  │    │    │    ├─ max_tokens=5
  │    │    │    └─ Return response
  │    │    │
  │    │    ├─→ Parse response
  │    │    │    ├─ Strip whitespace
  │    │    │    ├─ Uppercase text
  │    │    │    └─ Match "THREAT" substring
  │    │    │
  │    │    ├─→ Classification logic
  │    │    │    ├─ "THREAT" found → return "THREAT"
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
  ├─→ Print "RESUMO DO TESTE (THREAT):"
  │    ├─ Ameaças: {stats['THREAT']}
  │    ├─ Outros: {stats['OTHER']}
  │    └─ Separator: "=" * 30
  │
  └─→ END / [FileNotFoundError] → Print error, exit
```

---

## Performance Characteristics

### Computational Cost
- **Per-text latency:** 0.5-2 seconds (API call + parsing + sleep)
- **Throughput:** ~60 texts/minute (with 1s mandatory rate limit)
- **100 samples:** ~100-120 seconds (~2 minutes)
- **Bottleneck:** API rate limiting (1s mandatory delay)

### Token Usage
| Phase | Tokens |
|-------|--------|
| System prompt | ~45 tokens |
| User message | ~15-25 tokens (text-dependent) |
| Response | ~2-5 tokens ("THREAT" or "OTHER") |
| **Total per call** | ~65-75 tokens |

### Network Requirements
- Persistent internet connection (API-dependent)
- Hugging Face Inference API accessibility
- Typical API latency: 300-800ms

### Storage Requirements
- Code: ~2 KB
- Does not cache results (no persistent storage)
- Output: Console only (not logged to file)

---

## Notable Implementation Details

### Why `max_tokens=5`?

Llama Guard is designed for concise threat assessments. Setting `max_tokens=5`:
- Enforces single-word output ("THREAT" or "OTHER")
- Prevents model from generating explanations
- Reduces token cost and latency
- Ensures reproducible, deterministic results

### Why `temperature=0.1`?

Low temperature ensures:
- Consistent, reproducible threat classifications
- Reduced hallucination risk
- Focused, confident predictions
- Minimal false positives/negatives

### Threat vs. Criticism Distinction

Llama Guard 3 is trained to differentiate:
- **THREAT:** "Vou quebrar seu carro" (specific, actionable intent)
- **OTHER:** "Você é um idiota" (insult, not threat)
- **THREAT:** "Espero incendiar o prédio" (property destruction plan)
- **OTHER:** "Estou muito zangado" (anger expression without threat)

This nuance prevents false positives on heated language.

### Greedy String Matching
```python
if "THREAT" in prediction:
    return "THREAT"
return "OTHER"
```
- Simple, fast substring search
- No regex complexity needed
- Risk: False positives if model mentions "threat" in context

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
   - **Impact:** Model manipulation or bypass of threat detection
   - **Fix:** Sanitize inputs or use safer message formatting

3. **No Rate Limiting Per User**
   - **Risk:** Single runaway batch job blocks all other processing
   - **Fix:** Implement configurable delays or queue system

4. **No Audit Logging**
   - **Risk:** Cannot track threats detected or detection times
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
def log_threat(text: str, result: str):
    with open("threat_audit.log.csv", "a", newline="") as f:
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

### Adjusting Threat Classification Criteria

Edit the system prompt in `classify_threat()`:
```python
"content": (
    "Classificar texto como:\n"
    "THREAT: [your custom threat criteria]\n"
    "OTHER: [your custom non-threat criteria]"
)
```

**Example: More Sensitive Detection**
```python
"THREAT: Qualquer linguagem agressiva ou ameaçadora"
"OTHER: Apenas comunicação respeitosa"
```

### Changing Model

```python
# Use older version
model_id = "meta-llama/Meta-Llama-Guard-2-8B"

# Use lighter variant
model_id = "meta-llama/Llama-Guard-3-1B"
```

### Adjusting Rate Limiting

```python
time.sleep(0.5)  # Faster (risky; may hit rate limits)
time.sleep(2.0)  # Slower (safer; more conservative)
```

### Input File Customization

```python
# Change input filename
with open("threats.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
```

### Output File Logging (Enhancement)

```python
import csv
from datetime import datetime

with open("threat_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Text", "Classification", "Timestamp"])
    writer.writerow([clean_text, result, datetime.now()])
```

---

## Usage Examples

### Single Text Classification
```python
result = classify_threat("Vou quebrar seu carro")
print(result)  # Output: THREAT
```

### Batch Processing (Recommended)
```python
if __name__ == "__main__":
    run_test()  # Processes all examples.txt lines
```

### Custom Text List
```python
texts = [
    "Vou incendiar a casa",
    "Estou muito zangado",
    "Vou atacar você amanhã"
]

for text in texts:
    result = classify_threat(text)
    print(f"{text} → {result}")
```

### With Error Handling
```python
try:
    result = classify_threat(user_input)
    if result == "THREAT":
        print("⚠️ THREAT DETECTED — Escalate to security team")
    elif result == "ERRO_API":
        print("❌ Classification failed")
    else:
        print("✓ Non-threatening content")
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
| No confidence scores | Medium | Binary classification only | Request API confidence endpoint |
| File-dependent input | Medium | Hard-coded file path | Accept stdin/CLI arguments |
| No result persistence | Low | Console-only output | Log to CSV or database |
| Greedy string matching | Low | Simple substring search | Use exact word boundary matching |
| Cultural/linguistic bias | Medium | Model training data | Validate with multilingual dataset |

---

## False Positive/Negative Risks

### False Positives (Misclassified Threats)
```
"Vou matá-lo no videojogo"     → Model might classify as THREAT
"Você merecia ser atingido"     → Hyperbole detected as threat
"Vou destruir meu próprio PC"   → Self-directed action flagged as threat
```

Mitigation: Manual review of high-confidence threats before escalation

### False Negatives (Missed Threats)
```
"Você não deveria continuar vivo"  → Indirect threat
"Saiba que temos seus endereços"   → Implicit threat/doxxing
"Virei te procurar"                → Ambiguous intent
```

Mitigation: Combine with other security tools; periodic accuracy audits

---

## Future Enhancements

1. **Confidence Scores** — Return probability (0-1) alongside classification
2. **Async Processing** — Use `asyncio` for parallel API calls
3. **Severity Levels** — Distinguish credible vs. casual threats
4. **Explanation Generation** — Include why text was flagged
5. **Database Logging** — Store results in SQLite/PostgreSQL
6. **Batch API Endpoint** — Process multiple texts per request (lower latency)
7. **Language Detection** — Auto-detect language before classification
8. **Context Awareness** — Consider conversation history
9. **Threat Actor Profiling** — Track repeated offenders
10. **Integration with Law Enforcement** — Escalation workflow

---

## Testing Recommendations

### Test Cases
```python
# Positive (THREAT)
classify_threat("Vou quebrar seu carro amanhã")
classify_threat("Espero incendiar o prédio")

# Negative (OTHER)
classify_threat("Estou muito zangado com você")
classify_threat("Espero que você mude de atitude")

# Edge Cases
classify_threat("Vou matá-lo no jogo")  # Gaming context?
classify_threat("Você deveria desaparecer")  # Ambiguous?
```

### Best Practices
- Test with balanced dataset (threats + non-threats)
- Manually review borderline cases
- Monitor false positive/negative rates
- Validate against security domain experts
- Document edge cases and model limitations

---

## Legal & Ethical Considerations

**Important:**
- This system **supplements** human judgment; not a substitute
- **False negatives are worse than false positives** (safety-critical context)
- Always escalate detected threats to human review before action
- Maintain audit logs for legal/compliance purposes
- Comply with local laws regarding threat surveillance
- Inform users if threat detection is occurring
- Have clear escalation and response procedures

---

## References & Resources

- [Hugging Face Inference API Docs](https://huggingface.co/docs/hub/inference-api)
- [Llama Guard 3 Model Card](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
- [Content Safety & Threat Detection](https://openai.com/research/techniques-for-improving-safety)
- [Threat Assessment Best Practices](https://www.dhs.gov/threat-assessment)

---

**Last Updated:** [Current Date]  
**Status:** Production-Ready (security hardening recommended)  
**Maintenance Notes:**
- Rotate API tokens every 90 days
- Monitor Llama Guard model version updates
- Review classification accuracy monthly
- Update threat taxonomy based on emerging threat patterns
- Maintain incident response procedures