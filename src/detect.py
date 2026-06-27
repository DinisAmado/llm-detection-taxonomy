import os
import json
import re
import logging
import logging.handlers
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal # Added for Pydantic type hinting

from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from pydantic import BaseModel, Field, ValidationError # Added for strict validation

# Optional dependencies
try:
    from langdetect import detect as _langdetect
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

try:
    import emoji as _emoji_lib
    _HAS_EMOJI = True
except ImportError:
    _HAS_EMOJI = False

# --- Configuration & Paths ---
JSON_FILE = "results/extraction_results.json"
LOG_FILE  = "logs/detection_and_api_log.txt"
DATA_FILE = "data/examples.txt"

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Logging Setup ---
_formatter = logging.Formatter(
    "[%(levelname)s] %(asctime)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(_formatter)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)

# --- API Client & Models ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client   = InferenceClient(token=HF_TOKEN)

STAGE1_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

MODEL_GROUP_B       = "facebook/roberta-hate-speech-dynabench-r4-target"
MODEL_GROUP_A       = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_GROUP_A_MULTI = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
MODEL_GROUP_C       = "meta-llama/Meta-Llama-3-8B-Instruct"

GROUP_C_LABELS = {"EXTREMIST", "RADICALISM", "VIOLATED", "THREAT"}

# --- Thresholds & Limits ---
THRESHOLD_B = 0.75   
THRESHOLD_A = 0.60   
THRESHOLD_C = 0.85   
MAX_RELS_PER_ENTRY   = 10   
CLASSIFIER_MAX_CHARS = 500

NON_ENGLISH_LANGS = {"pt", "es", "fr", "de", "it", "nl", "ar", "zh", "ru", "ja", "ko"}

PT_SLANG = {
    "vc": "você", "vcs": "vocês", "tb": "também", "tbm": "também",
    "tô": "estou", "to": "estou", "tá": "está", "ta": "está",
    "q": "que", "pq": "porque", "blz": "beleza", "mt": "muito",
    "mto": "muito", "pfv": "por favor", "pf": "por favor",
    "obg": "obrigado", "n": "não", "ñ": "não", "hj": "hoje",
    "msm": "mesmo", "cmg": "comigo", "c": "com", "kd": "cadê",
    "flw": "falou", "naum": "não", "nn": "não",
    "kkk": "risos", "kkkk": "risos", "kkkkk": "risos",
    "haha": "risos", "hahaha": "risos", "rs": "risos", "rsrs": "risos",
    "xq": "porque", "tmb": "también", "tbn": "también",
    "k": "que", "d": "de", "aki": "aquí", "xfa": "por favor",
}

# --- Pydantic Validation Schemas ---
class EntityModel(BaseModel):
    id: str
    type: str  # Kept flexible as string since clean_extracted_graph normalizes types later

class RelationshipModel(BaseModel):
    source: str
    target: str
    interaction_type: str

class KnowledgeGraphModel(BaseModel):
    entities: List[EntityModel] = Field(default_factory=list)
    relationships: List[RelationshipModel] = Field(default_factory=list)


# --- Text Normalization ---
def normalize_text(text: str, lang: str = "pt") -> str:
    """Cleans up text by removing emojis, URLs, handles, and expanding slang."""
    if _HAS_EMOJI:
        try:
            text = _emoji_lib.demojize(text, language=lang if lang in ("pt", "es") else "en")
        except TypeError:
            text = _emoji_lib.demojize(text)
    else:
        text = re.sub(r"[\U00010000-\U0010FFFF\U0001F300-\U0001F9FF\u2600-\u27BF]+", " ", text, flags=re.UNICODE)

    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    if lang in ["pt", "es"]:
        for abbr, full in PT_SLANG.items():
            text = re.sub(rf"\b{re.escape(abbr)}\b", full, text, flags=re.IGNORECASE)

    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"([!?.]){2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def truncate_for_classifier(text: str, max_chars: int = CLASSIFIER_MAX_CHARS) -> str:
    return text[:max_chars]

def detect_language(text: str, default: str = "pt") -> str:
    if not _HAS_LANGDETECT:
        return default
    try:
        return _langdetect(text)
    except Exception:
        return default

# --- Prompts ---
STAGE_1_PROMPT = """
You are an expert Intelligence Analyst specialized in Social Network Analysis (SNA).
The text you will receive is written in [LANG]. Interpret it accordingly.
Your task is to extract a base graph (Entities and Relationships) from the text.

ENTITY RULES:
1. Extract ONLY explicit, concrete entities from these 4 types: Person, Group, Institution, Location.
   - "Person": A specific named human.
   - "Group": A specific collective of humans explicitly named in the text (e.g., "POLICE", "IMMIGRANTS").
   - "Institution": Specific organizations, companies, or governments.
   - "Location": Physical or geopolitical places.

2. ID EXTRACTION & NORMALIZATION (STRICT):
   - UPPERCASE: Convert all extracted IDs to UPPERCASE to maintain uniformity.
   - NO PRONOUNS/FRAGMENTS: Do NOT extract pronouns (he, she, they, ils, nous) or vague/generic nouns ("guy", "someone", "everyone", "nobody", "people").

3. SEMANTIC CLUSTERING (AUTOMATIC UNIFICATION):
   - You MUST normalize variations of the same concept into a single, broad English entity ID.
   - Examples: Map "migrants", "refugees", "illegals", "syrian immigrants" ALL to "IMMIGRANTS". 
   - Map "US", "USA", "States" ALL to "UNITED STATES".
   - Map "Taxpayer" to "TAXPAYERS". 
   - Always translate the final entity ID to English (e.g., "policia" -> "POLICE").

4. IGNORE HASHTAGS & SLOGANS:
   - NEVER extract political slogans, actions, or hashtags as entities. 
   - Ignore phrases like "BuildTheWall", "LockThemUp", "BoycottNike", "NoAmnesty". These are NOT entities.

5. THE "AUTHOR" & "TARGET" EXCEPTION:
   - If the text uses first-person pronouns (I, me, eu, je) to express an action, map it to the exact ID: "AUTHOR".
   - If the text uses second-person pronouns (you, tu, vc, vous) as the recipient of an action, map it to the exact ID: "TARGET".

RELATIONSHIP RULES:
- Extract relationships ONLY between two valid extracted entities.
- interaction_type MUST be a specific, active social verb in English (e.g., "criticizes", "threatens", "supports").
- If there are no valid concrete entities, return empty arrays.

FEW-SHOT EXAMPLES:
Text: "I hate Muslims."
Output:
{
  "entities": [{"id": "AUTHOR", "type": "Person"}, {"id": "MUSLIMS", "type": "Group"}],
  "relationships": [{"source": "AUTHOR", "target": "MUSLIMS", "interaction_type": "expresses hatred toward"}]
}

Text: "Trump threatens immigrants at the border."
Output:
{
  "entities": [{"id": "TRUMP", "type": "Person"}, {"id": "IMMIGRANTS", "type": "Group"}, {"id": "BORDER", "type": "Location"}],
  "relationships": [{"source": "TRUMP", "target": "IMMIGRANTS", "interaction_type": "threatens"}]
}

Text: "O Governo local é corrupto e a Lyonnaise vai roubar a região."
Output:
{
  "entities": [{"id": "GOVERNO LOCAL", "type": "Institution"}, {"id": "LYONNAISE", "type": "Institution"}, {"id": "REGIÃO", "type": "Location"}],
  "relationships": [{"source": "LYONNAISE", "target": "REGIÃO", "interaction_type": "exploits"}]
}

OUTPUT ONLY VALID JSON (no markdown formatting, no explanations). Start your response exactly with {:
{
  "entities": [{"id": "NAME IN UPPERCASE", "type": "Type"}],
  "relationships": [{"source": "SOURCE_ID", "target": "TARGET_ID", "interaction_type": "Verb"}]
}
""" 

TIER3_PROMPT_TEMPLATE = """
You are a content moderation specialist. Analyze the relationship below and
decide if the text clearly belongs to one of these SEVERE categories:

  EXTREMIST  — calls for terrorism, mass violence, or extermination of a group.
  RADICALISM — systematic dehumanization of an ENTIRE demographic or social group.
  VIOLATED   — direct physical violence/harm described as already happened or actively planned.
  THREAT     — explicit threat to inflict physical harm or severe damage.

Relation: {source!r} {interaction} {target!r}
Full text: {text!r}

CLASSIFICATION BOUNDARIES — read carefully before deciding:
- Refusing to interact with an institution (e.g., "I won't talk to the police") → NOT radicalism.
- Insulting one specific person (e.g., calling a boyfriend a "pedophile", "creep", or "stupid") → NOT radicalism. It is just an individual insult.
- Complaining about a workplace or city using strong language → NOT radicalism.
- Expressing an opinion others disagree with → NOT radicalism unless it calls to systematically harm an entire group.
- Sarcasm, frustration, venting → NOT severe.

If none of the severe categories clearly applies, set taxonomy_classification to null
and confidence to 0.0. When in doubt, choose null.

OUTPUT ONLY VALID JSON. Keep the confidence_reasoning extremely short (MAXIMUM 8 WORDS):
{{"taxonomy_classification": "LABEL or null", "confidence_reasoning": "short reason here", "confidence": 0.0}}
"""

# --- Low-level API Helpers ---
def safe_json_load(content):
    """Safely extracts and parses JSON from a model's string output."""
    content_str = str(content).strip()
    
    if content_str.startswith("```json"):
        content_str = content_str[7:]
    if content_str.endswith("```"):
        content_str = content_str[:-3]
    content_str = content_str.strip()
    
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

@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=2, min=2, max=20),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def call_classif(text: str, model: str):
    """Calls HuggingFace text classification endpoint with automatic retries."""
    safe_text = truncate_for_classifier(text)
    return client.text_classification(text=safe_text, model=model)[0]

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def call_complet(model: str, messages: list, max_tokens: int = 150, temperature: float = 0.1):
    """Calls HuggingFace chat completion endpoint with automatic retries."""
    response = client.chat_completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content

# --- Classification Tiers ---
def _tier1_group_b(text: str):
    res = call_classif(text, MODEL_GROUP_B)
    if res.label != "hate" or res.score < THRESHOLD_B:
        return None
    return {
        "taxonomy_classification": "HATE",
        "confidence_reasoning": "Tier 1 (hate model) detected hate speech with high confidence.",
        "confidence_score": round(res.score * 100, 2),
    }

def _tier2_group_a(text: str, lang: str = "pt"):
    def _classify_and_map(model: str):
        res = call_classif(text, model)
        if res.score >= THRESHOLD_A and res.label != "neutral":
            label = "SENTIMENTAL" if res.label == "positive" else "EMOTIONAL"
            return {
                "taxonomy_classification": label,
                "confidence_reasoning": (
                    f"Tier 2 detected {res.label!r} sentiment "
                    f"(model: {model.split('/')[-1]}, score: {res.score:.2f})."
                ),
                "confidence_score": round(res.score * 100, 2),
            }
        return None

    if lang not in NON_ENGLISH_LANGS:
        result = _classify_and_map(MODEL_GROUP_A)
        if result:
            return result

    return _classify_and_map(MODEL_GROUP_A_MULTI)

def _tier3_group_c(text: str, source: str, interaction: str, target: str):
    prompt = TIER3_PROMPT_TEMPLATE.format(
        source=source, interaction=interaction, target=target, text=text,
    )
    res_text = call_complet(
        MODEL_GROUP_C,
        [{"role": "user", "content": prompt}],
        max_tokens=200,
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

def classify_relation(text: str, rel: dict, lang: str = "pt") -> dict:
    source      = rel.get("source", "Unknown")
    target      = rel.get("target", "Unknown")
    interaction = rel.get("interaction_type", "interaction")

    try:
        t1 = _tier1_group_b(text)
        if t1: return t1

        t2 = _tier2_group_a(text, lang=lang)
        if t2 and t2["taxonomy_classification"] == "SENTIMENTAL":
            return t2

        t3 = _tier3_group_c(text, source, interaction, target)
        if t3["taxonomy_classification"] != "NEUTRAL":
            return t3

        if t2: return t2

        return {
            "taxonomy_classification": "NEUTRAL",
            "confidence_reasoning": "No tier reached sufficient confidence for classification.",
            "confidence_score": 0.0,
        }

    except Exception as e:
        logger.error(
            f"classify_relation failed for {source!r} {interaction} {target!r}: {e}",
            exc_info=True,
        )
        return {
            "taxonomy_classification": "NEUTRAL",
            "confidence_reasoning": f"Fallback due to API error: {e}",
            "confidence_score": 0.0,
        }

# --- Graph Cleaning Guardrail & Normalization ---

ENTITY_NORMALIZATION_MAP = {
    "DAD": "FATHER", "MOM": "MOTHER",
    "PRETOS": "BLACK PEOPLE", "NEGROS": "BLACK PEOPLE",
    "NIGGAS": "BLACK PEOPLE", "NIGGA": "BLACK PEOPLE",
    "POLICIA": "POLICE", "GOVERNO": "GOVERNMENT",
    
    "TAXPAYER": "TAXPAYERS",
    "U.S. TAXPAYERS": "TAXPAYERS",
    "US TAXPAYERS": "TAXPAYERS",
    
    "US": "UNITED STATES",
    "U.S.": "UNITED STATES",
    "USA": "UNITED STATES",
    "STATES": "UNITED STATES",
    
    "MIGRANTS": "IMMIGRANTS",
    "MIGRANT": "IMMIGRANTS",
    "IMMIGRANT": "IMMIGRANTS",
    "IMMIGRANT FAMILIES": "IMMIGRANTS",
    "ECONOMIC MIGRANTS": "IMMIGRANTS",
    "LATIN MIGRANTS": "IMMIGRANTS",
    "AFRICAN MIGRANTS": "IMMIGRANTS",
    "MUSLIM MIGRANTS": "IMMIGRANTS",
    "REFUGEES": "IMMIGRANTS",
    "SYRIAN IMMIGRANTS": "IMMIGRANTS",
    "ILLEGALS": "IMMIGRANTS",
    "REFUGIADOS": "IMMIGRANTS",
    "IMIGRANTES ILEGAIS": "IMMIGRANTS",
    "AFRICAN IMMIGRANTS": "IMMIGRANTS",
    "ARAB IMMIGRANTS": "IMMIGRANTS",
    "SYRIAN REFUGEE": "IMMIGRANTS",
    "CRIMINAL REFUGEES": "IMMIGRANTS",
    "LEGAL IMMIGRANT": "IMMIGRANTS",
    "ILLEGAL ALIEN": "IMMIGRANTS",
    "IMMIGRANT KIDS": "IMMIGRANTS",
    "IMMIGRATION INVASION": "IMMIGRANTS"
}

IGNORED_ENTITIES = {
    "EVERYONE", "TODOS", "SOMEONE", "ALGUÉM", "NOBODY", "NINGUÉM", 
    "ANYONE", "QUALQUER UM", "THEY", "ELES", "PEOPLE", "PESSOAS"
}

def clean_extracted_graph(graph: dict, entry_id: int) -> dict:
    """Cleans extracted graph, applies dictionary normalization, and dynamically isolates vague targets."""
    if not graph:
        return graph
        
    author_aliases = {"EU", "YO", "I", "ME", "MIM", "NÓS", "NOS", "JE", "MOI", "NOUS", "AUTHOR"}
    target_aliases = {"TU", "VOCÊ", "VC", "YOU", "VOUS", "TOI", "THEM", "HE", "SHE", "HIM", "HER", "TARGET"}
    valid_types    = {"Person", "Group", "Institution", "Location"}
    
    valid_entity_ids = set()
    cleaned_entities = []
    
    for ent in graph.get("entities", []):
        ent_id   = str(ent.get("id", "")).strip().upper()
        ent_type = str(ent.get("type", "")).strip()
        
        if not ent_id or ent_id in IGNORED_ENTITIES:
            continue
            
        if ent_id in ENTITY_NORMALIZATION_MAP:
            ent_id = ENTITY_NORMALIZATION_MAP[ent_id]
            
        if ent_id in author_aliases:
            ent_id   = f"AUTHOR_{entry_id}"
            ent_type = "Person"
        elif ent_id in target_aliases:
            ent_id   = f"TARGET_{entry_id}"
            ent_type = "Person"
            
        if ent_type not in valid_types:
            ent_type = "Group"
            
        ent["id"]   = ent_id
        ent["type"] = ent_type
        
        if ent_id not in valid_entity_ids:
            cleaned_entities.append(ent)
            valid_entity_ids.add(ent_id)
            
    graph["entities"] = cleaned_entities
    
    cleaned_relationships = []
    for rel in graph.get("relationships", []):
        src = str(rel.get("source", "")).strip().upper()
        tgt = str(rel.get("target", "")).strip().upper()
        
        if src in ENTITY_NORMALIZATION_MAP: src = ENTITY_NORMALIZATION_MAP[src]
        if tgt in ENTITY_NORMALIZATION_MAP: tgt = ENTITY_NORMALIZATION_MAP[tgt]
        
        if src in author_aliases: src = f"AUTHOR_{entry_id}"
        elif src in target_aliases: src = f"TARGET_{entry_id}"
        
        if tgt in author_aliases: tgt = f"AUTHOR_{entry_id}"
        elif tgt in target_aliases: tgt = f"TARGET_{entry_id}"
        
        rel["source"] = src
        rel["target"] = tgt
        
        if src in valid_entity_ids and tgt in valid_entity_ids and src != tgt:
            cleaned_relationships.append(rel)
            
    graph["relationships"] = cleaned_relationships
    return graph

# --- Entry Processor ---
def process_entry(idx: int, raw_text: str) -> dict:
    lang = detect_language(raw_text)
    logger.info(f"[{idx}] Detected language: {lang!r}")

    text = normalize_text(raw_text, lang=lang)
    logger.info(f"[{idx}] Normalised text: {text[:80]!r}{'…' if len(text) > 80 else ''}")

    logger.info(f"[{idx}] Stage 1: extraction (lang={lang})")
    try:
        s1_res = call_complet(
            STAGE1_MODEL,
            [
                {"role": "system", "content": STAGE_1_PROMPT.replace("[LANG]", lang)},
                {"role": "user",   "content": text},
            ],
            max_tokens=2048,
        )
    except Exception as e:
        logger.error(f"[{idx}] Stage 1 failed: {e}", exc_info=True)
        return {"id": idx, "original_text": raw_text, "normalised_text": text, "analysis": {}}

    graph_raw = safe_json_load(s1_res)
    
    # --- Pydantic Schema Guardrail ---
    try:
        # Validates that keys match and structural types are strictly preserved
        validated_graph = KnowledgeGraphModel(**graph_raw)
        graph = validated_graph.model_dump()
        logger.info(f"[{idx}] Pydantic structural validation passed.")
    except ValidationError as ve:
        # If the LLM structural output is broken, fallback gracefully to prevent crashes
        logger.warning(f"[{idx}] Pydantic structural validation failed due to hallucination: {ve}")
        graph = {"entities": [], "relationships": []}

    # Rest of the pipeline continues safely
    graph = clean_extracted_graph(graph, idx)
    
    logger.info(
        f"[{idx}] Graph: {len(graph.get('entities', []))} entities, "
        f"{len(graph.get('relationships', []))} relationships"
    )

    if not graph or not graph.get("relationships"):
        logger.warning(f"[{idx}] Empty graph — applying direct text classification fallback")
        
        fallback_author = f"AUTHOR_{idx}"
        fallback_target = f"TARGET_{idx}"
        
        fallback = classify_relation(
            text,
            {"source": fallback_author, "target": fallback_target, "interaction_type": "interacts"},
            lang=lang,
        )
        graph = {
            "entities": [
                {"id": fallback_author, "type": "Person"},
                {"id": fallback_target, "type": "Person"},
            ],
            "relationships": [
                {
                    "source": fallback_author,
                    "target": fallback_target,
                    "interaction_type": "interacts",
                    **fallback,
                }
            ],
            "_extraction_note": "KG extraction failed — direct classification applied",
        }
        return {"id": idx, "original_text": raw_text, "normalised_text": text, "analysis": graph}

    relationships = graph["relationships"]
    if len(relationships) > MAX_RELS_PER_ENTRY:
        logger.warning(f"[{idx}] Capping relationships from {len(relationships)} to {MAX_RELS_PER_ENTRY}")
        graph["relationships"] = relationships[:MAX_RELS_PER_ENTRY]

    logger.info(f"[{idx}] Stage 2: classifying {len(graph['relationships'])} relationships")
    for rel in graph["relationships"]:
        try:
            result = classify_relation(text, rel, lang=lang)
            rel.update(result)
            logger.info(f"[{idx}] {rel.get('source')!r} → {rel.get('target')!r}: {result.get('taxonomy_classification')}")
        except Exception as e:
            logger.error(f"[{idx}] Relation classification failed: {e}", exc_info=True)

    return {"id": idx, "original_text": raw_text, "normalised_text": text, "analysis": graph}

# --- Batch Runner ---
def run_batch_test():
    logger.info("--- Run started ---")
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if len(line.strip()) > 5]
    except FileNotFoundError:
        logger.error(f"Input file not found: {DATA_FILE}")
        return

    logger.info(f"Loaded {len(lines)} entries from {DATA_FILE}")
    final_results = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(process_entry, i + 1, text): i
            for i, text in enumerate(lines)
        }
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