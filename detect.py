import os
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# 1. ENVIRONMENT SETUP
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

STAGE1_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_GROUP_A = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_GROUP_B = "facebook/roberta-hate-speech-dynabench-r4-target"
MODEL_GROUP_C = "meta-llama/Meta-Llama-3-8B-Instruct"

# 2. STAGE 1 PROMPT (Base Graph Extraction)
STAGE_1_PROMPT = """
You are an expert Intelligence Analyst specialized in Social Network Analysis (SNA).
Your task is to extract a base graph (Entities and Relationships) from the text.

RULES:
- Extract ONLY socially relevant entities (Person, Group, Institution, Location).
- Replace pronouns with "Author" or "Target".
- Extract relationships ONLY if there is meaningful interaction or implied action.
- interaction_type must be a specific verb (e.g., "criticizes", "supports", "threatens").

CRITICAL: OUTPUT ONLY VALID JSON. START DIRECTLY WITH {:
{
  "entities": [{"id": "Name", "type": "Type"}],
  "relationships": [{"source": "Name", "target": "Name", "interaction_type": "Verb"}]
}
"""

# 3. JSON Parser 
def safe_json_load(content):
    content_str = str(content).strip()
    try:
        return json.loads(content_str)
    except:
        pass
    # Regex fallback to find the first JSON block
    match = re.search(r'(\{.*\}|\[.*\])', content_str, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return {"error": "Invalid JSON", "raw_output": content_str[:200]}

def retry_call(fn, retries=3):
    for i in range(retries):
        try:
            return fn()
        except Exception:
            if i == retries - 1: raise
            time.sleep(2.5 * (i + 1)) 

# 4. STAGE 2: ROUTING & CLASSIFICATION
def classify_relation(original_text, rel):
    source = rel.get("source", "Unknown")
    target = rel.get("target", "Unknown")
    interaction = rel.get("interaction_type", "interaction")
    interaction_text = f"{source} {interaction} {target}".lower()

    try:
        # GROUP A: Sentimental/Emotional 
        if any(w in interaction_text for w in [
            "feel", "think", "love", "opinion", "sad", "cry", "admira", "cuida", "gracias", 
            "llora", "emociona", "empolga", "gosta", "ama", "sente", "deseja", "espera"
        ]):
            def call():
                return client.text_classification(text=original_text, model=MODEL_GROUP_A)[0]
            
            res = retry_call(call)
            label = "EMOTIONAL" if res.label == "negative" else "SENTIMENTAL"
            return {
                "taxonomy_classification": label,
                "confidence_reasoning": f"Group A Encoder detected {res.label} sentiment.",
                "accuracy": f"{round(res.score * 100, 2)}%"
            }

        # GROUP B: Hate/Threat 
        elif any(w in interaction_text for w in [
            "insult", "threat", "kill", "attack", "hate", "slur", "violencia", 
            "insulta", "ataca", "mata", "odio", "amenaza", "agride", "ofende"
        ]):
            def call():
                return client.text_classification(text=original_text, model=MODEL_GROUP_B)[0]
            
            res = retry_call(call)
            label = "HATE" if res.label == "hate" else "THREAT"
            return {
                "taxonomy_classification": label,
                "confidence_reasoning": f"Group B Encoder detected {res.label} speech.",
                "accuracy": f"{round(res.score * 100, 2)}%"
            }

        # GROUP C: Reasoners (Extremist/Radicalism/Violated)
        else:
            prompt = f"""Analyze: '{source}' {interaction} '{target}' in text: '{original_text}'. 
            Classify strictly as: EXTREMIST, RADICALISM, VIOLATED, or NEUTRAL.
            
            OUTPUT ONLY VALID JSON:
            {{
              "taxonomy_classification": "CATEGORY",
              "confidence_reasoning": "Reason",
              "accuracy": "Self-estimated percentage (e.g. 85%)"
            }}"""
            
            def call():
                response = client.chat_completion(
                    model=MODEL_GROUP_C,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150, temperature=0.1
                )
                return response.choices[0].message.content

            res_text = retry_call(call)
            data = safe_json_load(res_text)

            # Map NEUTRAL to SENTIMENTAL for 7-category taxonomy compliance
            if data.get("taxonomy_classification") == "NEUTRAL":
                data["taxonomy_classification"] = "SENTIMENTAL"
                data["confidence_reasoning"] = "Neutral interaction mapped to Sentimental."
            
            return data

    except Exception as e:
        return {
            "taxonomy_classification": "SENTIMENTAL",
            "confidence_reasoning": f"Fallback due to API error: {str(e)}",
            "accuracy": "N/A"
        }

# 5. MAIN BATCH PROCESSOR
def process_entry(idx, text):
    print(f"[{idx}] Stage 1: Extraction...")
    
    def call_s1():
        response = client.chat_completion(
            model=STAGE1_MODEL,
            messages=[{"role": "system", "content": STAGE_1_PROMPT}, {"role": "user", "content": text}],
            max_tokens=500, temperature=0.1
        )
        return response.choices[0].message.content

    s1_res = retry_call(call_s1)
    graph = safe_json_load(s1_res)

    if "relationships" in graph and isinstance(graph["relationships"], list):
        print(f"[{idx}] Stage 2: Routing {len(graph['relationships'])} relationships...")
        for rel in graph["relationships"]:
            tax_data = classify_relation(text, rel)
            rel.update(tax_data)

    return {"id": idx, "original_text": text, "analysis": graph}

def run_batch_test():
    input_file = "examples.txt"
    output_file = "results/extraction_results.json"
    os.makedirs("results", exist_ok=True)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if len(l.strip()) > 5]

        print(f"Starting Two-Stage Pipeline for {len(lines)} examples...\n")
        final_results = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_entry, i + 1, text) for i, text in enumerate(lines)]
            for future in as_completed(futures):
                final_results.append(future.result())

        final_results.sort(key=lambda x: x["id"])

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"\nSuccess! Results saved to {output_file}")

    except FileNotFoundError:
        print("Error: examples.txt not found.")

if __name__ == "__main__":
    run_batch_test()