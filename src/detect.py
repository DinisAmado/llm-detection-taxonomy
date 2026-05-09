import os
import json
import time
import re
import logging
import logging.handlers
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

JSON_FILE = "results/extraction_results.json"
LOG_FILE  = "logs/detection_and_api_log.txt"
DATA_FILE = "data/examples.txt"

_formatter = logging.Formatter("[%(levelname)s] %(asctime)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_file_handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_file_handler.setFormatter(_formatter)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client   = InferenceClient(token=HF_TOKEN)

# Stage 1: entity/relationship extraction
STAGE1_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Tier 1: hate speech filter
MODEL_GROUP_B = "facebook/roberta-hate-speech-dynabench-r4-target"
# Tier 2: sentiment baseline (EN-optimized primary, multilingual fallback)
MODEL_GROUP_A       = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_GROUP_A_MULTI = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
# Tier 3: severe category escalation via LLM
MODEL_GROUP_C = "meta-llama/Meta-Llama-3-8B-Instruct"

GROUP_C_LABELS = {"EXTREMIST", "RADICALISM", "VIOLATED", "THREAT"}

# Confidence thresholds
THRESHOLD_B = 0.75   # hate model minimum score to accept HATE
THRESHOLD_A = 0.60   # sentiment model minimum score to escalate
THRESHOLD_C = 0.85   # LLM minimum confidence to accept severe label
MAX_RELS_PER_ENTRY = 10  # cap to control API cost


STAGE_1_PROMPT = """
You are an expert Intelligence Analyst specialized in Social Network Analysis (SNA).
The text you will receive is written in {lang}. Interpret it accordingly.
Your task is to extract a base graph (Entities and Relationships) from the text.

ENTITY RULES:
- Extract ONLY socially relevant entities. Valid types: Person, Group, Institution, Location.
- "Person" = a human individual (real or abstract like "Author", "Target").
- "Group" = a collective of people (e.g. "immigrants", "police", "team").
- "Institution" = an organisation, company, government body, court, or platform.
- "Location" = a physical or virtual place (city, country, building, website).
- STRICTLY DO NOT extract objects, animals, concepts, emotions, or products.
  BAD: "vinito" (drink), "serie" (product), "cuarto" (room), "house" (object),
       "childhood" (concept), "comfort zone" (concept), "quadro" (object).
  GOOD: "Author", "Target", "Brazilian Supreme Court", "Twitter", "Police".
- entity type must be EXACTLY one of: Person, Group, Institution, Location.
  Never use "concept", "object", "activity", or any other custom type.
- Replace all pronouns with "Author" (the writer) or "Target" (the person addressed).

RELATIONSHIP RULES:
- Extract relationships ONLY between two valid social entities (Person/Group/Institution).
- Do NOT create relationships involving objects, locations, or concepts.
- interaction_type must be a specific, active social verb (e.g. "criticizes", "threatens").
- DO NOT extract structural or trivial relationships.
  BAD: sibling--grows_up_in-->house, parent--resides_in-->house,
       Author--drinks-->vinito, Author--watches-->serie, Author--visits-->supermarket.
  GOOD: Author--threatens-->Target, Group--attacks-->Institution, Author--insults-->Target.
- If the text has no meaningful social interaction between people/groups, return empty arrays.

CRITICAL: OUTPUT ONLY VALID JSON. START DIRECTLY WITH {{:
{{
  "entities": [{{"id": "Name", "type": "Type"}}],
  "relationships": [{{"source": "Name", "target": "Name", "interaction_type": "Verb"}}]
}}
"""


TIER3_PROMPT_TEMPLATE = """
You are a content moderation specialist. Analyze the relationship below and
decide if the text clearly belongs to one of these SEVERE categories:

  EXTREMIST  — calls for terrorism, mass violence, or extermination of a group
  RADICALISM — systematic dehumanization of a group, genocidal rhetoric,
               calls to strip rights from an entire category of people
  VIOLATED   — direct personal violence: assault, arson, physical harm described
               as already happened or being actively planned
  THREAT     — explicit threat directed at a specific person or group

Relation: {source!r} {interaction} {target!r}
Full text: {text!r}

CLASSIFICATION BOUNDARIES — read carefully before deciding:
- Rudeness, insults, swearing, or calling someone stupid → NOT a severe category.
- Complaining about a workplace, city, or social situation using strong language
  → NOT radicalism, even if the language is offensive.
- Expressing an opinion others disagree with (politics, gender, relationships)
  → NOT radicalism unless it calls to systematically harm or exclude a group.
- Sarcasm, frustration, venting → NOT severe.
- "Blackmail" used metaphorically (e.g. "emotional blackmail") → NOT THREAT.
- Only classify RADICALISM if the text targets an entire group with language that
  denies their humanity or calls for their exclusion/persecution as a group.

FEW-SHOT EXAMPLES:
Text: "eres tan estupida que no lo vas a entender" → null (insult, not radicalism)
Text: "Men and women still need each other, this extreme is silly" → null (opinion)
Text: "Your mental health problems are your responsibility, fix yourself" → null (harsh opinion)
Text: "We should cleanse all [ethnic group] from our country" → RADICALISM
Text: "I will burn your house down tonight" → THREAT
Text: "He hit her and she fell to the ground" → VIOLATED

If none of the severe categories clearly applies, set taxonomy_classification to null
and confidence to 0.0. When in doubt, choose null.

OUTPUT ONLY VALID JSON (no extra text, no markdown fences):
{{"taxonomy_classification": "LABEL or null", "confidence_reasoning": "one sentence", "confidence": 0.0}}
"""


def safe_json_load(content):
    content_str = str(content).strip()
    try:
        return json.loads(content_str)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", content_str, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    if content_str:
        logger.warning(f"JSON parse failed — raw: {content_str[:120]!r}")
    return {}


def retry_call(fn, retries=3):
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            wait = 2.5 * (i + 1)
            if i == retries - 1:
                logger.error(f"All {retries} retries exhausted: {e}")
                raise
            logger.warning(f"Retry {i + 1}/{retries}: {e} (waiting {wait:.1f}s)")
            time.sleep(wait)


def call_classif(or_text, sel_model):
    return client.text_classification(text=or_text, model=sel_model)[0]


def call_complet(sel_model, l_info=None, max_tokens=150, temperature=0.1):
    if l_info is None:
        l_info = []
    response = client.chat_completion(
        model=sel_model,
        messages=l_info,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _tier1_group_b(original_text):
    res = retry_call(lambda: call_classif(original_text, MODEL_GROUP_B))
    if res.label != "hate" or res.score < THRESHOLD_B:
        return None
    return {
        "taxonomy_classification": "HATE",
        "confidence_reasoning": "Tier 1 (hate model) detected hate speech with high confidence.",
        "confidence_score": round(res.score * 100, 2),
    }


def _tier2_group_a(original_text):
    # Primary model (EN-optimized)
    res = retry_call(lambda: call_classif(original_text, MODEL_GROUP_A))
    if res.score >= THRESHOLD_A and res.label != "neutral":
        label = "SENTIMENTAL" if res.label == "positive" else "EMOTIONAL"
        return {
            "taxonomy_classification": label,
            "confidence_reasoning": f"Tier 2 (sentiment model) detected {res.label!r} sentiment.",
            "confidence_score": round(res.score * 100, 2),
        }

    # Fallback: multilingual model (PT / ES / FR)
    res_multi = retry_call(lambda: call_classif(original_text, MODEL_GROUP_A_MULTI))
    if res_multi.score >= THRESHOLD_A and res_multi.label != "neutral":
        label = "SENTIMENTAL" if res_multi.label == "positive" else "EMOTIONAL"
        return {
            "taxonomy_classification": label,
            "confidence_reasoning": f"Tier 2 (multilingual sentiment model) detected {res_multi.label!r} sentiment.",
            "confidence_score": round(res_multi.score * 100, 2),
        }

    return None


def _tier3_group_c(original_text, source, interaction, target):
    prompt = TIER3_PROMPT_TEMPLATE.format(
        source=source, interaction=interaction, target=target, text=original_text,
    )
    res_text = retry_call(
        lambda: call_complet(MODEL_GROUP_C, [{"role": "user", "content": prompt}], max_tokens=200)
    )
    data       = safe_json_load(res_text)
    label      = data.get("taxonomy_classification")
    confidence = float(data.get("confidence", 0))

    if label not in GROUP_C_LABELS or confidence < THRESHOLD_C:
        return {
            "taxonomy_classification": "NEUTRAL",
            "confidence_reasoning": "No tier reached sufficient confidence for classification.",
            "confidence_score": 0.0,
        }
    return {
        "taxonomy_classification": label,
        "confidence_reasoning": data.get("confidence_reasoning", ""),
        "confidence_score": round(confidence * 100, 2),
    }


def classify_relation(original_text, rel):
    source      = rel.get("source", "Unknown")
    target      = rel.get("target", "Unknown")
    interaction = rel.get("interaction_type", "interaction")

    try:
        # Tier 1: hate speech
        t1 = _tier1_group_b(original_text)
        if t1:
            return t1

        # Tier 2: sentiment baseline
        t2 = _tier2_group_a(original_text)

        if t2 and t2["taxonomy_classification"] == "SENTIMENTAL":
            return t2  # positive sentiment is terminal, no need to escalate

        # Tier 3: always run — catches severe labels even when Tier 2 finds nothing
        t3 = _tier3_group_c(original_text, source, interaction, target)
        if t3["taxonomy_classification"] != "NEUTRAL":
            return t3

        # Tier 3 found nothing severe — return Tier 2 signal (EMOTIONAL) if present
        if t2:
            return t2

        return {
            "taxonomy_classification": "NEUTRAL",
            "confidence_reasoning": "No tier reached sufficient confidence for classification.",
            "confidence_score": 0.0,
        }

    except Exception as e:
        logger.error(f"classify_relation failed for {source!r} {interaction} {target!r}: {e}", exc_info=True)
        return {
            "taxonomy_classification": "NEUTRAL",
            "confidence_reasoning": f"Fallback due to API error: {e}",
            "confidence_score": 0.0,
        }


def process_entry(idx, text, lang="PT"):
    logger.info(f"[{idx}] Stage 1: Extraction (lang={lang})")
    try:
        s1_res = retry_call(
            lambda: call_complet(
                STAGE1_MODEL,
                [
                    {"role": "system", "content": STAGE_1_PROMPT.format(lang=lang)},
                    {"role": "user",   "content": text},
                ],
                max_tokens=1000,
            )
        )
    except Exception as e:
        logger.error(f"[{idx}] Stage 1 failed: {e}", exc_info=True)
        return {"id": idx, "original_text": text, "analysis": {}}

    graph  = safe_json_load(s1_res)
    logger.info(f"[{idx}] Graph extracted — {len(graph.get('entities', []))} entities, {len(graph.get('relationships', []))} relationships")

    # Fallback: empty graph → classify the raw text directly
    if not graph or not graph.get("relationships"):
        logger.warning(f"[{idx}] Empty graph — applying direct text classification")
        fallback = classify_relation(text, {"source": "Author", "target": "Target", "interaction_type": "interacts"})
        graph = {
            "entities": [{"id": "Author", "type": "Person"}, {"id": "Target", "type": "Person"}],
            "relationships": [{"source": "Author", "target": "Target", "interaction_type": "interacts", **fallback}],
            "_extraction_note": "KG extraction failed — direct classification applied",
        }
        return {"id": idx, "original_text": text, "analysis": graph}

    # Cap relationships to control API cost
    relationships = graph["relationships"]
    if len(relationships) > MAX_RELS_PER_ENTRY:
        logger.warning(f"[{idx}] Capping relationships from {len(relationships)} to {MAX_RELS_PER_ENTRY}")
        graph["relationships"] = relationships[:MAX_RELS_PER_ENTRY]

    logger.info(f"[{idx}] Stage 2: Routing {len(graph['relationships'])} relationships")
    for rel in graph["relationships"]:
        try:
            result = classify_relation(text, rel)
            rel.update(result)
            logger.info(f"[{idx}] {rel.get('source')!r} → {rel.get('target')!r}: {result.get('taxonomy_classification')}")
        except Exception as e:
            logger.error(f"[{idx}] Relation classification failed: {e}", exc_info=True)

    return {"id": idx, "original_text": text, "analysis": graph}


def run_batch_test():
    logger.info("--- Run started ---")
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if len(l.strip()) > 5]
    except FileNotFoundError:
        logger.error(f"Input file not found: {DATA_FILE}")
        return

    logger.info(f"Loaded {len(lines)} entries from {DATA_FILE}")
    final_results = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_entry, i + 1, text): i for i, text in enumerate(lines)}
        for future in as_completed(futures):
            try:
                final_results.append(future.result())
            except Exception as e:
                logger.error(f"Entry failed: {e}", exc_info=True)

    final_results.sort(key=lambda x: x["id"])

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {JSON_FILE}")
    logger.info("--- Run finished ---")


if __name__ == "__main__":
    run_batch_test()