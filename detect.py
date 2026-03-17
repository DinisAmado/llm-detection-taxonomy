import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_ID = "gpt-4o-mini"

# 2. FINAL PROMPT
MASTER_SYSTEM_PROMPT = """
You are an expert Intelligence Analyst specialized in Social Network Analysis (SNA).

Your task is to analyze a piece of text and:
1. Extract relevant entities (Persons, Groups, Institutions, Locations)
2. Identify meaningful relationships between those entities
3. Classify each relationship using EXACTLY ONE category from the taxonomy

-----------------------------------
### ENTITY EXTRACTION RULES

- Extract ONLY socially relevant entities:
  - Persons (e.g., "Work Colleague", "Mother", "Politician")
  - Groups (e.g., "Men", "Women", "Twitter Users")
  - Institutions (e.g., "Brazilian Supreme Court")

- Extract locations ONLY if they are directly involved in an action (e.g., target of attack)

- DO NOT extract:
  - Objects (food, drinks, generic items)
  - Generic places (room, store, house) UNLESS they are targets of an action (e.g., "burn the house")

- NO PRONOUNS:
  Replace all pronouns with:
  - "Author" (speaker)
  - "Target" (person being addressed)
  - or a clear descriptive role

-----------------------------------
### RELATIONSHIP RULES

- Extract relationships ONLY if there is meaningful interaction, sentiment, or implied action

- IMPORTANT:
  Even if the interaction is implicit (opinions, insults, threats), you MUST extract at least one relationship

- Each relationship must include:
  - source
  - target
  - interaction_type (specific verb)

- GOOD interaction_type examples:
  - "insults"
  - "criticizes"
  - "supports"
  - "threatens to harm"
  - "threatens to burn"
  - "gives gift"
  - "expresses frustration"
  - "expresses opinion"

- BAD (FORBIDDEN):
  - "Action"
  - "Interaction"
  - vague verbs

-----------------------------------
### TAXONOMY (CHOOSE EXACTLY ONE)

- EXTREMIST:
  Explicit violent actions or extreme retaliation (e.g., arson, violence)

- HATE:
  Insults, dehumanization, discrimination toward a person/group

- RADICALISM:
  Extremist ideology or anti-democratic views

- THREAT:
  Clear intent to harm people/property (non-ideological)

- VIOLATED:
  Privacy invasion or rights violation

- EMOTIONAL:
  Anger, frustration, aggressive tone WITHOUT real threat

- SENTIMENTAL:
  Opinions, personal reflections, positive or neutral feelings

-----------------------------------
### CLASSIFICATION RULES

- ALWAYS assign exactly ONE taxonomy_classification
- NEVER skip classification
- NEVER assign multiple labels

- IMPORTANT DISTINCTIONS:
  - Insult → HATE
  - Anger without threat → EMOTIONAL
  - Opinion → SENTIMENTAL
  - Real harm → THREAT or EXTREMIST

-----------------------------------
### OUTPUT FORMAT (STRICT JSON ONLY)

Return ONLY valid JSON.

{
  "entities": [
    {"id": "Name", "type": "Person/Group/Institution/Location"}
  ],
  "relationships": [
    {
      "source": "Name",
      "target": "Name",
      "interaction_type": "Specific action",
      "taxonomy_classification": "ONE_OF_THE_7",
      "confidence_reasoning": "Short explanation"
    }
  ]
}
"""

# 3. JSON loader 
def safe_json_load(content):
    try:
        return json.loads(content)
    except:
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return {"error": "Invalid JSON", "raw_output": content[:300]}


# 4. LLM call
def extract_intelligence(text):
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0.1,
            response_format={"type": "json_object"} 
        )

        content = response.choices[0].message.content.strip()
        return safe_json_load(content)

    except Exception as e:
        return {"error": str(e)}


# 5. Batch processing
def run_batch_test():
    input_file = "examples.txt"
    output_dir = "results"
    output_file = os.path.join(output_dir, "extraction_results.json")

    os.makedirs(output_dir, exist_ok=True)

    results = []

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if len(line.strip()) > 5]

        print(f"Processing {len(lines)} texts...\n")

        for i, text in enumerate(lines):
            print(f"[{i+1}/{len(lines)}] Processing...")

            analysis = extract_intelligence(text)

            results.append({
                "id": i + 1,
                "original_text": text,
                "analysis": analysis
            })

            time.sleep(0.4)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nDone! Results saved to {output_file}")

    except FileNotFoundError:
        print("examples.txt not found")


if __name__ == "__main__":
    run_batch_test()