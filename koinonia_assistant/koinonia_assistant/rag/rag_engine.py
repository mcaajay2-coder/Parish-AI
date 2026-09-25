from koinonia_assistant.rag.sacrament_normalizer import normalize_sacrament_query
from koinonia_assistant.rag.tamil_utils import is_tamil, normalize_tamil_query
from koinonia_assistant.rag.name_search import (
    resolve_member_and_family,
    extract_person_name_from_query,
    extract_intent_and_person,
    classify_query_intent,
    handle_list_members,
    handle_count_members,
    handle_list_families,
    handle_count_families,
    format_family_response,
    format_member_sacrament_response,
    detect_query_intent,
    fetch_full_family_bundle,
    fetch_member_sacrament_bundle,
    determine_response_scope,
    render_scoped_response,
    build_candidate_prompt,
    RESERVED_GENERIC_WORDS,
)
import datetime
import os
import re
import time
import json
import psycopg2
import pymysql
import torch
from typing import TypedDict, Any, Optional, Dict, List
from dotenv import load_dotenv


# ─── LangSmith Tracing Configuration ──────────────────────────────────────────
def configure_langsmith():
    """Ensure LangSmith tracing is active in the environment."""
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    project = os.environ.get("LANGCHAIN_PROJECT")
    tracing = os.environ.get("LANGCHAIN_TRACING_V2")

    if not api_key:
        try:
            import frappe
            if hasattr(frappe, "conf") and frappe.conf:
                api_key = frappe.conf.get("langchain_api_key")
                project = project or frappe.conf.get("langchain_project")
                if frappe.conf.get("langchain_tracing_v2") is not None:
                    tracing = str(frappe.conf.get("langchain_tracing_v2")).lower()
        except Exception:
            pass

    if not api_key:
        api_key = os.environ.get("LANGCHAIN_API_KEY", "")

    os.environ["LANGCHAIN_TRACING_V2"] = tracing or "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project or "koinonia_assistant"
    os.environ["LANGCHAIN_ENDPOINT"] = os.environ.get("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com"
    return True

configure_langsmith()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '.env')
load_dotenv(dotenv_path)

# Initialize variables
def _get_groq_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            import frappe
            if hasattr(frappe, "conf") and frappe.conf:
                key = frappe.conf.get("groq_api_key")
        except Exception:
            pass
    return key

def _get_groq_model():
    model = os.getenv("GROQ_MODEL")
    if not model:
        try:
            import frappe
            if hasattr(frappe, "conf") and frappe.conf:
                model = frappe.conf.get("groq_model")
        except Exception:
            pass
    m = (model or "openai/gpt-oss-20b").strip()
    if m in ["gpt-oss-20b", "openai-gpt-oss-20b"]:
        m = "openai/gpt-oss-20b"
    elif m in ["gpt-oss-120b", "openai-gpt-oss-120b"]:
        m = "openai/gpt-oss-120b"
    return m

GROQ_API_KEY = _get_groq_key() or "placeholder_key"
GROQ_MODEL = _get_groq_model()
MAX_RETRIES = 3

# pgvector Connection
PG_CONFIG = {
    "host":     os.getenv("PG_HOST",     "postgres-vector"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB",      "parish_vectordb"),
    "user":     os.getenv("PG_USER",    "postgres"),
    "password": os.getenv("PG_PASS",    "password"),
}

# Lazy-loaded BGE-M3 Embeddings
_tokenizer = None
_model = None

def get_bge_model():
    global _tokenizer, _model
    from transformers import AutoTokenizer, AutoModel
    if _tokenizer is None or _model is None:
        print("[BGE-M3] Loading model inside rag_engine...")
        _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        _model = AutoModel.from_pretrained("BAAI/bge-m3")
        _model.eval()
    return _tokenizer, _model

def embed_text(text: str) -> list[float]:
    tokenizer, model = get_bge_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).squeeze().tolist()

# Groq LLM with Transparent Key & Model Rotator
from koinonia_assistant.rag.llm_rotator import get_llm_rotator
llm = get_llm_rotator(temperature=0, default_model=GROQ_MODEL)

# ─── pgvector Schema and History Retrieval ────────────────────────────────────

def clean_query_keywords(query: str) -> list[str]:
    words = re.findall(r'\b\w+\b', query.lower())
    stopwords = {
        "list", "show", "count", "get", "find", "all", "of", "the", "a", "an", 
        "with", "for", "in", "on", "at", "by", "from", "where", "select", 
        "me", "us", "give", "display", "retrieve", "search", "query", "database",
        "table", "tables", "record", "records", "data", "row", "rows"
    }
    keywords = []
    for w in words:
        base = w[:-1] if w.endswith('s') and len(w) > 3 else w
        if base not in stopwords and len(base) > 2:
            keywords.append(base)
    return keywords

def rerank_tables(query_text: str, candidates: list[tuple[str, str]]) -> list[str]:
    if not candidates:
        return []
    if len(candidates) == 1:
        return [candidates[0][0]]
        
    candidates_list = []
    for tname, ddl in candidates:
        lines = ddl.split("\n")
        concept = ""
        for line in lines[:3]:
            if "Entity/Concept:" in line or "Table Name:" in line:
                concept += " " + line.strip()
        candidates_list.append(f"- `{tname}`: {concept.strip()}")
        
    candidates_str = "\n".join(candidates_list)
    
    prompt = f"""You are a database table selector for the KOINONIA sacrament database.
Given the user query: "{query_text}"
Select the top 1 to 3 most relevant tables from the candidate list below that are required to answer this query.
If a single table is sufficient, return only that table. If the query requires a JOIN, return all required tables.

Candidates:
{candidates_str}

Return the selected table names as a comma-separated list. Do not write any explanation, markdown, or code blocks. Just return the table names, e.g. "tabFamily, tabMember"."""

    try:
        response = llm.invoke([("user", prompt)])
        content = response.content.strip()
        selected = [t.strip().strip("`").strip("'").strip('"') for t in content.split(",")]
        candidate_names = {c[0] for c in candidates}
        final_selection = [s for s in selected if s in candidate_names]
        if final_selection:
            return final_selection
    except Exception as e:
        print(f"[rerank_tables] Error during LLM re-ranking: {e}")
        
    return [c[0] for c in candidates[:2]]

def fetch_relevant_schemas(original_query: str, enhanced_query: str, query_embedding: list[float], k: int = 3) -> str:
    conn = psycopg2.connect(**PG_CONFIG)
    candidates = []
    seen = set()
    
    combined_query_text = f"{original_query} {enhanced_query}"
    
    try:
        with conn.cursor() as cur:
            # 1. Substring matching for custom tables
            cur.execute("SELECT table_name, schema_ddl FROM koinonia_table_schemas;")
            all_tables = cur.fetchall()
            for tname, ddl in all_tables:
                clean_t = tname.lower()
                if clean_t.startswith("tab"):
                    clean_t = clean_t[3:]
                
                # Support "anointing of sick", "anointing", "sick", "marriage", "baptism", "communion", "confirmation", "death", "member", "family"
                match_keywords = [clean_t, clean_t.replace("_", " ")]
                if any(kw in combined_query_text.lower() for kw in match_keywords):
                    if tname not in seen:
                        seen.add(tname)
                        candidates.append((tname, ddl))

            # 2. Semantic vector search (Priority 2)
            cur.execute("""
                SELECT table_name, schema_ddl
                FROM koinonia_table_schemas
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding, k))
            for tname, ddl in cur.fetchall():
                if tname not in seen:
                    seen.add(tname)
                    candidates.append((tname, ddl))
                    
        selected_table_names = rerank_tables(original_query, candidates)
        schema_map = {tname: ddl for tname, ddl in candidates}
        final_results = []
        for tname in selected_table_names:
            if tname in schema_map:
                final_results.append((tname, schema_map[tname]))
                
    finally:
        conn.close()
        
    pruned_results = []
    for tname, ddl in final_results:
        pruned_results.append(f"{tname}:\n{ddl}")
        
    return pruned_results

def fetch_relevant_fields(query_embedding: list[float], k: int = 5) -> list[str]:
    conn = psycopg2.connect(**PG_CONFIG)
    results = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, field_name, field_type, field_label, description,
                       (embedding <=> %s::vector) as distance
                FROM koinonia_field_schemas
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_embedding, k))
            for row in cur.fetchall():
                results.append(f"- Table `{row[0]}`, Column `{row[1]}` (Type: {row[2]}, Label: {row[3]}): {row[4]}")
    except Exception as e:
        print("[RAG Context] Error fetching relevant fields:", e)
    finally:
        conn.close()
    return results

def fetch_few_shot_examples(query_embedding: list[float], k: int = 2) -> str:
    lines = [
        "Here are examples of correct sacrament question → SQL mappings:",
        "  Q: \"List members of family FAM-2011-00001 along with their relations and age\"",
        "  SQL: SELECT first_name, last_name, relationship_id, age FROM tabMember WHERE family_id = 'FAM-2011-00001'",
        "  Q: \"Count how many baptisms were conducted in Holy Cross Parish during 2024\"",
        "  SQL: SELECT COUNT(*) FROM tabBaptism WHERE bapt_parish_id = 'Holy Cross Parish' AND YEAR(bapt_date) = 2024",
        "  Q: \"Show communion sacrament details for child Mary Britto\"",
        "  SQL: SELECT c.name, c.first_name, c.last_name, c.fhc_date, c.fhc_place FROM tabCommunion c WHERE c.first_name = 'Mary' AND c.last_name = 'Britto'",
        "  Q: \"List marriages in 2023 solemnized by Rev. Fr. Stephen Raj\"",
        "  SQL: SELECT name, bridegroom_name, bride_name, mrg_date FROM tabMarriage WHERE mrg_minister = 'Rev. Fr. Stephen Raj' AND YEAR(mrg_date) = 2023"
    ]
    
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT user_question, generated_sql
                FROM koinonia_query_history
                WHERE correctness_flag = 1
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding, k))
            rows = cur.fetchall()
    except Exception as e:
        print("[History] Error fetching dynamic few-shots:", e)
        rows = []
    finally:
        conn.close()
        
    if rows:
        lines.append("\nDynamic Examples from History:")
        for q, s in rows:
            lines.append(f'  Q: "{q}"\n  SQL: {s}')
            
    return "\n".join(lines)

def log_query_history(question: str, sql: str, embedding: list[float]) -> int:
    conn = psycopg2.connect(**PG_CONFIG)
    row_id = -1
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id FROM koinonia_query_history 
                WHERE user_question = %s 
                LIMIT 1;
            """, (question,))
            existing = cur.fetchone()
            if existing:
                return existing[0]

            cur.execute("""
                INSERT INTO koinonia_query_history (user_question, generated_sql, embedding, correctness_flag)
                VALUES (%s, %s, %s::vector, NULL)
                RETURNING id;
            """, (question, sql, embedding))
            row_id = cur.fetchone()[0]
        conn.commit()
    except Exception as e:
        print("[History] Error logging query history:", e)
    finally:
        conn.close()
    return row_id

def update_correctness_flag(query_id: int, is_correct: int):
    if query_id < 0:
        return
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE koinonia_query_history SET correctness_flag = %s WHERE id = %s;
            """, (is_correct, query_id))
        conn.commit()
    except Exception as e:
        print("[History] Error updating correctness flag:", e)
    finally:
        conn.close()

# ─── LangGraph State Definition ───────────────────────────────────────────────


# ????????? Disambiguation & Phonetic Helpers ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

SACRAMENT_INTENTS = {
    'baptism': ['baptism', 'baptised', 'baptized', 'bapt', 'babt', 'christening', 'ஞானஸ்நானம்', 'திருமுழுக்கு', 'ஞானஸ்தானம்'],
    'marriage': ['marriage', 'married', 'wedding', 'matrimony', 'mrg', 'திருமணம்', 'கல்யாணம்', 'விவாகம்'],
    'communion': ['communion', 'first communion', 'first holy communion', 'eucharist', 'fhc', 'first', 'holy', 'puthunanmai', 'pothunanmai', 'puthunamai', 'pothunamai', 'puthunmai', 'pothunmai', 'புதுநன்மை', 'பொதுநன்மை', 'போதுநன்மை', 'pathi', 'sollunga', 'solunga', 'patri', 'patti', 'puthunanmai', 'pothunanmai', 'puthunamai', 'pothunamai', 'நற்கருணை', 'முதல் நற்கருணை', 'புதுநன்மை', 'பொதுநன்மை'],
    'confirmation': ['confirmation', 'confirmed', 'chrismation', 'cnf', 'உறுதிப்பூசுதல்', 'உறுதிபூசுதல்', 'திடப்படுத்தல்'],
    'death': ['death', 'died', 'deceased', 'funeral', 'burial', 'cemetery', 'இறப்பு', 'மரணம்', 'அடக்கம்'],
    'sacraments': ['sacrament', 'sacraments', 'all sacraments', 'records', 'திருவருட்சாதனம்', 'திருவருட்சாதனங்கள்']
}

SACRAMENT_STOPWORDS = {
    'baptism', 'baptised', 'baptized', 'bapt', 'babt', 'christening',
    'ஞானஸ்நானம்', 'திருமுழுக்கு', 'ஞானஸ்தானம்', 'ஞானஸ்நான', 'திருமுழுக்குத்',
    'marriage', 'married', 'wedding', 'matrimony', 'mrg',
    'திருமணம்', 'கல்யாணம்', 'விவாகம்', 'திருமண',
    'communion', 'first communion', 'first holy communion', 'eucharist', 'fhc', 'first', 'holy', 'puthunanmai', 'pothunanmai', 'puthunamai', 'pothunamai', 'puthunmai', 'pothunmai', 'புதுநன்மை', 'பொதுநன்மை', 'போதுநன்மை', 'pathi', 'sollunga', 'solunga', 'patri', 'patti',
    'நற்கருணை', 'முதல் நற்கருணை', 'திவ்விய நற்கருணை', 'புது நன்மை', 'நற்கருணைப்',
    'confirmation', 'confirmed', 'chrismation', 'cnf',
    'உறுதிப்பூசுதல்', 'உறுதிபூசுதல்', 'உறுதிப் பூசுதல்',
    'death', 'died', 'deceased', 'funeral', 'burial', 'cemetery',
    'மரண பதிவு', 'மரண', 'இறப்பு', 'அடக்கம்', 'கல்லறை',
    'sacrament', 'sacraments', 'all sacraments', 'records',
    'திருவருட்சாதனம்', 'திருவருட்சாதனங்கள்', 'அருட்சாதனம்', 'அருட்சாதனங்கள்', 'தேவத்திரவிய அனுமானம்'
}

TAMIL_STOPWORDS = {
    'குடும்பம்', 'குடும்பங்கள்', 'குடும்ப', 'குடும்பத்தின்', 'குடும்பத்தினர்',
    'உறுப்பினர்', 'உறுப்பினர்கள்', 'அங்கத்தினர்', 'விவரம்', 'விவரங்கள்',
    'தகவல்', 'பங்கு', 'பங்கின்', 'மறைமாவட்டம்', 'அட்டை', 'பதிவு', 'எண்',
    'ஏன்', 'அப்படி', 'வருது', 'வருகிறது', 'சொல்லு', 'சொல்லுங்க',
    'தெரியுமா', 'எப்படி', 'என்ன', 'யாரு', 'யாருடைய', 'இருக்காங்க',
    'இருக்க', 'இருக்கும்', 'காட்டு', 'கொடு', 'பத்தி', 'பற்றி',
    'உள்ள', 'எடுக்க', 'பார்க்க', 'வேண்டும்', 'வேணும்'
}

GREETING_STOPWORDS = {
    "hi", "hello", "hey", "hai", "hola", "namaste", "vanakkam", "வணக்கம்",
    "good", "morning", "afternoon", "evening", "night", "gm", "gn",
    "thanks", "thank", "you", "bye", "goodbye", "welcome", "ok", "okay",
    "yes", "no", "yeah", "yep", "nope", "help", "sure", "fine", "great"
}

DISAMBIGUATION_STOPWORDS = {
    'tell', 'me', 'about', 'the', 'family', 'families', 'household', 'member', 'members',
    'details', 'detail', 'info', 'information', 'their', 'his', 'her', 'who',
    'is', 'are', 'was', 'were', 'parishioner', 'parishioners', 'show', 'list', 'get', 'give',
    'find', 'records', 'record', 'recurse', 'recourse', 'recorse', 'recods', 'rekords', 'rekord',
    'all', 'in', 'parish', 'diocese', 'please',
    'what', 'which', 'view', 'of', 'and', 'with', 'for', 'to', 'a', 'an',
    'how', 'many', 'total', 'count', 'name', 'names', 'person', 'persons',
    'similar', 'duplicate', 'same', 'one', 'more', 'have', 'has', 'had',
    'certificate', 'certificates', 'register', 'registers', 'report', 'reports', 'status',
    'generate', 'genenate', 'diagram', 'diagrams', 'chart', 'charts', 'graph', 'graphs',
    'plot', 'plots', 'visualize', 'visualization', 'trend', 'trends', 'growth', 'growing',
    'decline', 'increase', 'decrease', 'whether', 'weather', 'or', 'not', 'last', 'past',
    'years', 'year', 'yearly', 'annual', 'annually', 'months', 'month', 'monthly',
    'breakdown', 'distribution', 'percentage', 'ratio', 'comparison', 'compare',
    'analytics', 'census', 'demographics', 'overview', 'summary', 'stats', 'statistics',
    'why', 'so', 'like', 'that', 'this', 'coming', 'comes', 'come', 'shows', 'showing',
    'does', 'do', 'did', 'it'
}.union(SACRAMENT_STOPWORDS).union(TAMIL_STOPWORDS).union(GREETING_STOPWORDS).union(RESERVED_GENERIC_WORDS)

def extract_search_terms(query_text: str):
    clean = re.sub(r"\(\s*ID:\s*[^)]+\)", "", query_text, flags=re.IGNORECASE)
    clean = re.sub(r"\b(?:FAM|MEM)-[A-Za-z0-9\-]+\b", "", clean, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z0-9\.\u0B80-\u0BFF]+", clean)
    filtered = []
    for w in words:
        w_clean = w.strip(".")
        w_lower = w_clean.lower()
        if w_lower in DISAMBIGUATION_STOPWORDS:
            continue
        if re.search(r"[\u0B80-\u0BFF]", w_clean):
            if len(w_clean) > 1:
                filtered.append(w_clean)
        else:
            if len(w_clean) >= 2 or (len(w_clean) == 1 and w_clean.isalpha()):
                filtered.append(w_clean)
    return filtered

def get_phonetic_variants(token: str):
    t = token.lower().strip()
    variants = [t]
    if 'thony' in t or 'tony' in t or t.startswith(('anto', 'antho')) or 'அந்தோ' in t:
        variants.extend(['anton', 'anthony', 'antony', 'anto', 'antho', 'அந்தோணி', 'அந்தோனி', 'அந்தோணிராஜ்', 'அந்தோனிராஜ்'])
    if t.startswith(('selv', 'silv')) or 'செல்வ' in t or 'சில்வ' in t:
        variants.extend(['selvan', 'selvam', 'selva', 'silva', 'silvan', 'silvam', 'selvaraj', 'செல்வம்', 'செல்வன்', 'செல்வராஜ்', 'சில்வா'])
    elif t.endswith(('vam', 'van', 'va')):
        stem = t[:-3] if len(t) > 3 else t
        variants.extend([stem + 'vam', stem + 'van', stem + 'va'])
    if t.endswith(('raj', 'raja', 'rajan')):
        stem = t[:-3] if len(t) > 3 else t
        variants.extend([stem + 'raj', stem + 'raja', stem + 'rajan'])
    if 'babu' in t:
        variants.extend(['babu', 'baabu'])
    if 'samy' in t or 'swamy' in t:
        variants.extend(['samy', 'swamy', 'sammy'])
    if 'mari' in t or 'mary' in t or 'மரி' in t or 'மேரி' in t:
        variants.extend(['maria', 'mary', 'mari', 'மரியா', 'மேரி'])
    if 'jose' in t or 'ஜோசப்' in t or 'சூசை' in t:
        variants.extend(['joseph', 'jose', 'jos', 'ஜோசப்', 'சூசை'])
    if 'paul' in t:
        variants.extend(['paul', 'paulose'])
    if 'francis' in t or 'பிரான்சிஸ்' in t:
        variants.extend(['francis', 'fransis', 'பிரான்சிஸ்'])
    if 'xavier' in t or 'savari' in t or 'சவேரி' in t or 'சேவியர்' in t:
        variants.extend(['xavier', 'savari', 'saveri', 'சவேரியார்', 'சேவியர்'])
    if 'yash' in t:
        variants.extend(['yash', 'yaash'])
    return list(set(variants))

def ensure_frappe_connected():
    try:
        import frappe
        if not frappe.db:
            frappe.connect()
    except Exception:
        pass

def find_disambiguation_candidates(query_text: str, user_parish: str = None, user_diocese: str = None):
    # If query already contains explicit ID, do not disambiguate
    if (
        re.search(r"\b(FAM-[A-Z0-9\-]+|MEM-[A-Z0-9\-]+)\b", query_text, re.IGNORECASE) or
        re.search(r"\b(?:Member\s*ID|Member)[:\s]+[0-9]+\b", query_text, re.IGNORECASE) or
        re.search(r"\b(?:Family\s*ID|Family)[:\s]+[0-9]+\b", query_text, re.IGNORECASE) or
        re.search(r"\bCard[:\s]+[A-Z0-9/]+\b", query_text, re.IGNORECASE)
    ):
        return None

    # Do not disambiguate greetings, chitchat, or basic conversational queries
    q_lower = query_text.lower()
    GREETINGS_SET = {
        "hi", "hello", "hey", "hai", "hola", "namaste", "vanakkam", "வணக்கம்",
        "good morning", "good afternoon", "good evening", "good night", "gm", "gn",
        "thanks", "thank you", "bye", "goodbye", "ok", "okay", "yes", "no", "help"
    }
    q_stripped = re.sub(r"[^\w\s\u0B80-\u0BFF]", "", q_lower).strip()
    if q_stripped in GREETINGS_SET or any(phrase in q_lower for phrase in ["how are you", "who are you", "what can you do", "introduce yourself", "help me"]):
        return None

    # Do not disambiguate purely aggregate, analytical, chart, diagram, or trend questions
    q_lower = query_text.lower()
    AGGREGATE_ANALYTICAL_KEYWORDS = [
        'how many', 'total', 'count', 'list all', 'statistics', 'stats', 'chart', 'charts',
        'diagram', 'diagrams', 'graph', 'graphs', 'plot', 'plots', 'trend', 'trends',
        'growing', 'growth', 'decline', 'increase', 'decrease', 'whether', 'weather',
        'last 5 years', 'last 10 years', 'last 3 years', 'last year', 'past 5 years',
        'years', 'yearly', 'annual', 'annually', 'monthly', 'by year', 'by month',
        'summary', 'overview', 'across all', 'similar', 'same name', 'duplicate', 'homonym',
        'distribution', 'breakdown', 'analytics', 'visualization', 'visualize',
        'demographics', 'census', 'comparison', 'compare', 'percentage', 'ratio',
        'ஒரே பெயர்', 'ஒரே மாதிரியான', 'வளர்ச்சி', 'விளக்கப்படம்', 'வரைபடம்', 'புள்ளிவிவரம்'
    ]
    if any(k in q_lower for k in AGGREGATE_ANALYTICAL_KEYWORDS):
        return None

    # 3-Level Member Matching in resolve_member_and_family handles all member/family disambiguation.
    # Legacy substring LIKE '%term%' disambiguation is disabled per Section 7.
    return None

    terms = extract_search_terms(query_text)
    if not terms or len(terms) > 5 or all(t.lower() in RESERVED_GENERIC_WORDS for t in terms):
        return None

    ensure_frappe_connected()
    import frappe

    # Base filters
    fam_where = []
    fam_params = []
    if user_parish:
        fam_where.append("f.parish_id LIKE %s")
        fam_params.append(f"%{user_parish}%")
    elif user_diocese and user_diocese != "All Dioceses":
        fam_where.append("f.diocese_id = %s")
        fam_params.append(user_diocese)

    mem_where = []
    mem_params = []
    if user_parish:
        mem_where.append("(m.parish_id LIKE %s OR f.parish_id LIKE %s)")
        mem_params.extend([f"%{user_parish}%", f"%{user_parish}%"])
    elif user_diocese and user_diocese != "All Dioceses":
        mem_where.append("(m.diocese_id = %s OR f.diocese_id = %s)")
        mem_params.extend([user_diocese, user_diocese])

    # Search Families
    term_fam_clauses = []
    fam_search_params = list(fam_params)
    for term in terms:
        variants = get_phonetic_variants(term)
        var_sql = " OR ".join(["f.reference LIKE %s" for _ in variants])
        term_fam_clauses.append(f"({var_sql})")
        for v in variants:
            fam_search_params.append(f"%{v}%")

    fam_rows = []
    if term_fam_clauses:
        where_str = " AND ".join(fam_where + ["(" + " OR ".join(term_fam_clauses) + ")"])
        sql_fam = f"""
            SELECT f.name, f.reference, f.family_register_number, f.street, f.parish_id, f.mobile
            FROM `tabFamily` f
            WHERE {where_str}
            LIMIT 10
        """
        try:
            fam_rows = frappe.db.sql(sql_fam, fam_search_params, as_dict=True)
        except Exception:
            fam_rows = []

    # Search Members - match on first_name, middle_name, OR last_name individually
    term_mem_clauses = []
    mem_search_params = list(mem_params)
    for term in terms:
        variants = get_phonetic_variants(term)
        # Match any individual name field OR the full concat OR family reference
        per_variant = []
        for v in variants:
            per_variant.append(
                "(m.first_name LIKE %s OR m.middle_name LIKE %s OR m.last_name LIKE %s "
                "OR CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) LIKE %s "
                "OR f.reference LIKE %s)"
            )
            mem_search_params.extend([f"%{v}%", f"%{v}%", f"%{v}%", f"%{v}%", f"%{v}%"])
        term_mem_clauses.append("(" + " OR ".join(per_variant) + ")")

    mem_rows = []
    if term_mem_clauses:
        where_str = " AND ".join(mem_where + ["(" + " OR ".join(term_mem_clauses) + ")"])
        sql_mem = (
            "SELECT m.name as member_id, "
            "m.first_name, m.middle_name, m.last_name, "
            "CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) as full_name, "
            "m.family_id, m.street as mem_street, m.parish_id as mem_parish, "
            "m.dob, m.relationship_id, "
            "f.reference as fam_reference, f.family_register_number, "
            "f.street as fam_street, f.parish_id as fam_parish "
            "FROM `tabMember` m "
            "LEFT JOIN `tabFamily` f ON f.name = m.family_id "
            f"WHERE {where_str} "
            "LIMIT 20"
        )
        try:
            mem_rows = frappe.db.sql(sql_mem, mem_search_params, as_dict=True)
        except Exception as e:
            import frappe as _frappe
            _frappe.log_error(f'[Disambiguation] Member search error: {e}')
            mem_rows = []

    # Compile options offering BOTH Family Details and Sacrament Retrieval
    options = []
    seen = set()

    for r in fam_rows:
        fid = r["name"]
        fname = r["reference"] or "Family"
        if fid not in seen:
            seen.add(fid)
            place = r["street"] or r["parish_id"] or user_parish or "Parish"
            card_no = r["family_register_number"] or fid
            
            if sacrament_intent:
                options.append({
                    "type": "sacraments",
                    "full_name": fname,
                    "place": place,
                    "card_no": card_no,
                    "family_id": fid,
                    "prompt": f"{query_text} (ID: {fid}, Card: {card_no})",
                    "display_text": query_text
                })
            elif is_family_query:
                options.append({
                    "type": "family",
                    "full_name": fname,
                    "place": place,
                    "card_no": card_no,
                    "family_id": fid,
                    "prompt": f"{query_text} (ID: {fid}, Card: {card_no})",
                    "display_text": query_text
                })
            else:
                options.append({
                    "type": "family",
                    "full_name": fname,
                    "place": place,
                    "card_no": card_no,
                    "family_id": fid,
                    "prompt": f"{query_text} (Family ID: {fid}, Card: {card_no})",
                    "display_text": f"{fname} (Family)"
                })

    # Each matched member gets their OWN card (not grouped by family)
    # so searching 'Laxmi' shows ALL members named Laxmi individually.
    for mr in mem_rows:
        mid = mr["member_id"]
        if mid in seen:
            continue
        seen.add(mid)

        mname = (mr["full_name"] or "").strip()
        if not mname:
            continue
        fid = mr.get("family_id")
        fam_ref = mr.get("fam_reference") or mname
        card_no = mr.get("family_register_number") or fid or "N/A"
        place = (mr.get("mem_street") or mr.get("fam_street")
                 or mr.get("fam_parish") or user_parish or "Parish")

        # Build a label showing which name part matched, e.g. 'Jose Antony (son)'
        relation = mr.get("relationship_id") or ""
        display_label = mname
        if relation:
            display_label = f"{mname} ({relation})"

        if sacrament_intent:
            # Show member's sacrament records
            options.append({
                "type": "sacraments",
                "full_name": display_label,
                "place": place,
                "card_no": card_no,
                "family_id": fid,
                "member_id": mid,
                "prompt": f"{query_text} for {mname} (Member ID: {mid}, Family: {fid})",
                "display_text": f"{mname} — Sacraments"
            })
        elif is_family_query:
            # Show the family this member belongs to
            if fid and fid not in seen:
                seen.add(fid)
                options.append({
                    "type": "family",
                    "full_name": f"{fam_ref} (family of {mname})",
                    "place": place,
                    "card_no": card_no,
                    "family_id": fid,
                    "member_id": mid,
                    "prompt": f"{query_text} (Family ID: {fid}, Card: {card_no})",
                    "display_text": f"{fam_ref} (Family)"
                })
        else:
            # General: ONE unified clickable card per member (keyed by member_id)
            cand_prompt = build_candidate_prompt(query_text, None, mname, mid)
            options.append({
                "type": "member",
                "full_name": display_label,
                "place": place,
                "card_no": card_no,
                "family_id": fid,
                "member_id": mid,
                "prompt": cand_prompt,
                "display_text": f"{mname} ({fam_ref})"
            })

    if len(options) > 1:
        # Score options: options matching more query terms appear first
        def score_opt(o):
            fn = (o.get('full_name') or '').lower()
            return sum(1 for t in terms if any(v in fn for v in get_phonetic_variants(t)))
        options.sort(key=score_opt, reverse=True)

        clean_terms = [t for t in terms if t.lower() not in SACRAMENT_STOPWORDS and t.lower() not in DISAMBIGUATION_STOPWORDS]
        clean_term_display = ' '.join(clean_terms) if clean_terms else ' '.join(terms)
        parish_label = f" in {user_parish}" if user_parish else ""
        return {
            "message": f"Multiple records found matching '{clean_term_display}'{parish_label}. Please select an option to view details:",
            "options": options[:3]
        }
    elif len(options) == 1:
        return {
            "single_candidate": options[0]
        }

    return None


class GraphState(TypedDict):
    question:          str
    original_query:    str
    detected_language: str
    normalized_query:  str
    intent_query:      str
    entity_query:      str
    canonical_terms:   Dict[str, Any]
    final_response_language: str
    history:           list[dict[str, str]]
    route:             str
    enhanced_query:    str
    relevant_tables:   list[str]
    relevant_fields:   list[str]
    few_shot_examples: str
    query_embedding:   list
    generated_sql:     str
    llm_explanation:   str
    sql_result:        Any
    error_message:     str
    retry_count:       int
    final_answer:      str
    history_id:        int
    user_id:           Optional[str]
    user_role:         str
    user_parish:       str
    user_diocese:      Optional[str]
    authorization_context: Dict[str, Any]
    authorization_check: str
    authorization_result: str
    sql_validation_result: str
    authorized_record_count: int
    request_id:        str
    classified_intent: str
    speed_tier:        str
    intent_info:       Dict[str, Any]
    deterministic_reply: Optional[str]
    chart_data:        Optional[Dict[str, Any]]
    disambiguation:    Optional[Dict[str, Any]]
    suggested_questions: Optional[List[str]]
    analysis_metadata: Optional[Dict[str, Any]]

# ─── Graph Nodes ──────────────────────────────────────────────────────────────

def resolve_permissions_node(state: GraphState) -> GraphState:
    """
    MANDATORY STEP 0 SECURITY NODE (Sections 50–58, 73):
    Executes BEFORE router, intent classification, SQL generation, vector search, or analytics.
    Builds the central authorization_context and enforces explicit scope boundaries.
    """
    req_id = state.get("request_id", "N/A")
    orig_q = state.get("original_query") or state["question"]
    from koinonia_assistant.rag.tamil_utils import detect_query_language
    from koinonia_assistant.rag.analytics_engine import (
        build_authorization_context,
        check_explicit_scope_violation,
    )

    lang = detect_query_language(orig_q)
    auth_ctx = build_authorization_context(
        user_id=state.get("user_id") or "User",
        user_role=state.get("user_role") or "Parishioner",
        user_parish=state.get("user_parish"),
        user_diocese=state.get("user_diocese"),
    )
    scope_check = check_explicit_scope_violation(orig_q, auth_ctx, detected_language=lang)

    print(
        f"\n[LANGGRAPH] RESOLVE PERMISSIONS NODE | request_id={req_id} "
        f"| user_id={auth_ctx['user_id']} | role={auth_ctx['role']} "
        f"| scope_type={auth_ctx['scope_type']} | scope_name={auth_ctx['scope_name']} "
        f"| auth_result={scope_check['authorization_result']} | lang={lang}"
    )

    if not scope_check["authorized"]:
        return {
            **state,
            "original_query": orig_q,
            "detected_language": lang,
            "final_response_language": "ta" if lang == "ta" else "en",
            "authorization_context": auth_ctx,
            "authorization_check": "PRE_RETRIEVAL_SCOPE_ENFORCEMENT",
            "authorization_result": scope_check["authorization_result"],
            "sql_validation_result": "BLOCKED_PRE_EXECUTION",
            "authorized_record_count": 0,
            "route": "authorization_denied",
            "classified_intent": "AUTHORIZATION_DENIED",
            "speed_tier": "FAST",
            "deterministic_reply": scope_check["message"],
            "sql_result": [],
            "generated_sql": "",
        }

    return {
        **state,
        "original_query": orig_q,
        "detected_language": lang,
        "final_response_language": "ta" if lang == "ta" else "en",
        "authorization_context": auth_ctx,
        "authorization_check": "PRE_RETRIEVAL_SCOPE_ENFORCEMENT",
        "authorization_result": "AUTHORIZED",
        "route": "authorized",
    }


def decide_permission(state: GraphState):
    if state.get("route") == "authorization_denied":
        return "authorization_denied"
    return "authorized"


def router_node(state: GraphState) -> GraphState:
    req_id = state.get("request_id", "N/A")
    q_raw = (state.get("original_query") or state["question"]).strip()
    q = q_raw.lower()

    from koinonia_assistant.rag.analytics_engine import classify_langgraph_intent
    intent_info = classify_langgraph_intent(q_raw)
    c_intent = intent_info.get("intent", "GENERAL_DATABASE_QUERY")
    s_tier = intent_info.get("speed_tier", "FAST")
    lang = intent_info.get("detected_language") or state.get("detected_language", "en")

    print(
        f"[LANGGRAPH] ROUTER NODE | request_id={req_id} | language={lang} "
        f"| intent={c_intent} | speed_tier={s_tier} | original_query='{q_raw}'"
    )

    common_state = {
        **state,
        "question": q_raw,
        "original_query": q_raw,
        "detected_language": lang,
        "normalized_query": intent_info.get("normalized_query", ""),
        "intent_query": intent_info.get("intent_query", c_intent),
        "entity_query": intent_info.get("entity_query", ""),
        "canonical_terms": intent_info.get("canonical_terms", {}),
        "final_response_language": intent_info.get("final_response_language", "ta" if lang == "ta" else "en"),
        "classified_intent": c_intent,
        "speed_tier": s_tier,
        "intent_info": intent_info,
    }

    if c_intent == "GREETING" or any(phrase in q for phrase in ["how are you", "who are you", "what can you do", "introduce yourself"]):
        return {
            **common_state,
            "route": "greeting",
            "classified_intent": "GREETING",
        }

    if c_intent == "FORECAST" or (c_intent == "COMPARISON" and intent_info.get("include_forecast")):
        return {
            **common_state,
            "route": "forecast_node",
        }

    if c_intent in ("HISTORICAL_ANALYSIS", "STATISTICAL_ANALYSIS", "TREND_ANALYSIS", "COMPARISON"):
        return {
            **common_state,
            "route": "analytics_node",
        }

    if c_intent in ("LIST", "COUNT", "MEMBER_SEARCH", "FAMILY_SEARCH", "SACRAMENT_SEARCH"):
        return {
            **common_state,
            "route": "database_lookup_node",
        }

    return {
        **common_state,
        "route": "text_to_sql",
        "classified_intent": "GENERAL_DATABASE_QUERY",
    }

def unclear_node(state: GraphState) -> GraphState:
    is_ta = state.get("detected_language") == "ta" or any('\u0B80' <= c <= '\u0BFF' for c in state["question"])
    if is_ta:
        answer = (
            "🤔 மன்னிக்கவும், நீங்கள் கேட்டது எனக்கு சரியாகப் புரியவில்லை.\n\n"
            "நான் KOINONIA பங்கு உதவியாளர். நான் பங்கு குடும்பப் பதிவேடு மற்றும் திருவருட்சாதனப் பதிவேடுகளைத் தேட உதவ முடியும். "
            "உதாரணமாக நீங்கள் கேட்கலாம்:\n"
            "- *'கடந்த 10 ஆண்டுகளில் நடைபெற்ற முதல் நற்கருணை நிகழ்வுகளின் எண்ணிக்கையைத் தரவும்'*\n"
            "- *'அந்தோணி ராஜின் திருமுழுக்கு நிலை என்ன?'*\n"
            "- *'YLG/001 குடும்பத்தின் உறுப்பினர்களை காட்டு'*\n\n"
            "நான் உங்களுக்கு எப்படி உதவலாம் என்று சொல்லுங்கள்!"
        )
    else:
        answer = (
            "🤔 I'm sorry, I didn't quite get that.\n\n"
            "I am the KOINONIA Parish Assistant. I can help you search the parish family register and sacrament records. "
            "Try asking me things like:\n"
            "- *'Show baptism statistics for the last 10 years'*\n"
            "- *'What is the baptism status of Antony Raj?'*\n"
            "- *'Get family details of YLG/001'*\n\n"
            "Please let me know how I can assist you!"
        )
    return {**state, "final_answer": answer}

def greeting_node(state: GraphState) -> GraphState:
    import frappe
    user_name = frappe.db.get_value("User", frappe.session.user, "first_name") or "there"
    is_ta = state.get("detected_language") == "ta" or any('\u0B80' <= c <= '\u0BFF' for c in state["question"])
    scope_name = (state.get("authorization_context") or {}).get("scope_name") or state.get("user_parish") or "பங்கு"
    if is_ta:
        answer = (
            f"👋 வணக்கம் {user_name}! நான் **KOINONIA பங்கு உதவியாளர்** ({scope_name}).\n\n"
            "குடும்ப அட்டை (Family Card), உறுப்பினர் விவரங்கள் மற்றும் திருவருட்சாதனப் பதிவேடுகளை (திருமுழுக்கு, முதல் நற்கருணை, உறுதிப்பூசுதல், திருமணம்) நேரடியாகத் தமிழில் தேடவும், புள்ளிவிவரப் பகுப்பாய்வு செய்யவும் நான் உதவுவேன்!"
        )
    else:
        answer = (
            f"👋 Hello {user_name}! I'm the **KOINONIA Parish Assistant** ({scope_name}).\n\n"
            "I can help you search family registers, member data, and sacrament registers (Baptism, First Holy Communion, Confirmation, Marriage) and run verified statistical analyses within your authorized parish scope."
        )
    return {**state, "final_answer": answer}

ENHANCE_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a multilingual query understanding assistant for the KOINONIA Parish Assistant app.
STRICT RULES (Sections 34–36):
1. Understand Tamil and English queries directly.
2. NEVER transliterate Tamil sentences into Tanglish (e.g., never produce 'Katantha 10 Aantukalil...').
3. Extract the structured database search intent while preserving canonical Koinonia Catholic terminology:
   - திருமுழுக்கு / ஞானஸ்நானம் -> Baptism (tabBaptism)
   - முதல் நற்கருணை / முதல் திருவிருந்து -> First Holy Communion (tabCommunion)
   - உறுதிப்பூசுதல் -> Confirmation (tabConfirmation)
   - திருமணம் -> Marriage (tabMarriage)
   - குடும்ப அட்டை -> Family Card (tabFamily)
   - உறுப்பினர் -> Member (tabMember)
4. Resolve pronouns using Conversation History if applicable.

Conversation History:
{history_context}

Return a concise structured intent summary for SQL generation without Tanglish conversion."""),
    ("human", "Original User Question ({language}): {question}"),
])

def enhance_query_node(state: GraphState) -> GraphState:
    """
    Preserves `original_query` and `question` untouched (Section 35).
    Never overwrites Tamil input with Tanglish.
    """
    orig_q = state.get("original_query") or state["question"]
    lang = state.get("detected_language") or "en"
    intent_info = state.get("intent_info") or {}
    print(f"[enhance_query] Preserving original_query ({lang}): '{orig_q}'")

    if lang == "ta" and intent_info.get("normalized_query"):
        structured_q = intent_info["normalized_query"]
        embedding = embed_text(orig_q)
        return {
            **state,
            "question": orig_q,
            "original_query": orig_q,
            "enhanced_query": structured_q,
            "normalized_query": structured_q,
            "query_embedding": embedding,
        }

    history = state.get("history", [])
    history_context = ""
    if history:
        history_lines = []
        for msg in history[-4:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.get('content', '')[:200]}")
        history_context = "\n".join(history_lines)

    response = llm.invoke(ENHANCE_QUERY_PROMPT.format_messages(
        history_context=history_context or "(no prior conversation)",
        language=lang,
        question=orig_q,
    ))
    enhanced = response.content.strip()
    embedding = embed_text(orig_q)
    return {
        **state,
        "question": orig_q,
        "original_query": orig_q,
        "enhanced_query": enhanced,
        "query_embedding": embedding,
    }

def retrieve_context_node(state: GraphState) -> GraphState:
    print("[retrieve_context] Retrieving schema context and few-shots...")
    relevant_tables = fetch_relevant_schemas(state["question"], state["enhanced_query"], state["query_embedding"])
    relevant_fields = fetch_relevant_fields(state["query_embedding"])
    few_shots = fetch_few_shot_examples(state["query_embedding"])
    return {**state, "relevant_tables": relevant_tables, "relevant_fields": relevant_fields, "few_shot_examples": few_shots}

SQL_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert MariaDB SQL query writer for the KOINONIA Parish Assistant app.

CRITICAL: The current date is {current_date}. If the user asks for relative timeframes like "past 15 years", calculate it relative to this date!

## Database Tables and Relationships
You are querying a MariaDB database with the following custom tables:
{relevant_tables}

## Relevant Fields (Semantically Matched Columns)
Use these exact column names when matching user concepts:
{relevant_fields}

{few_shot_examples}

### STRICT RELATIONSHIP RULES:
1. `tabMember` links to `tabFamily` via `tabMember.family_id = tabFamily.name`.
2. Sacrament registers (`tabBaptism`, `tabCommunion`, `tabConfirmation`, `tabMarriage`, `tabAnointing Of Sick`, `tabDeath`) represent sacrament events.
3. Sacrament records DO NOT have direct foreign keys to `tabMember`. To find a member's sacrament record, you MUST join or filter by matching names:
   `tabBaptism.first_name = tabMember.first_name AND tabBaptism.last_name = tabMember.last_name`
   (And similarly for `tabCommunion`, `tabConfirmation`, `tabAnointing Of Sick`, `tabDeath`).
4. For `tabMarriage`, query marriages between bridegrooms and brides using columns:
   `bridegroom_name`, `bridegroom_last_name`, `bride_name`, `bride_last_name`.
5. Primary key columns in all Frappe tables are named `name` (e.g. `tabFamily.name`, `tabMember.name`), NOT `id`.
6. FUZZY NAME SEARCHING: Transliterated names rarely match exactly in the database due to spelling variations (e.g., Surya vs Suriya). When searching by a person's name, ALWAYS use the full name but generate multiple phonetic spelling variations for that name and use them in an IN clause or multiple LIKE statements (e.g., `first_name IN ('Surya', 'Suriya', 'Sooriya')`). DO NOT truncate the name to 3 letters.
7. SACRAMENT FALLBACK (CRITICAL): Many sacrament dates are stored directly in `tabMember` without a formal registry entry in `tabBaptism`/etc. When searching for a specific person's sacrament details (e.g. baptism), DO NOT use the sacrament tables (tabBaptism) as they are often empty. Instead, ALWAYS query the `tabMember` table DIRECTLY, as it contains the `bapt_date`, `bapt_parish_id`, etc.
Example for Baptism:
SELECT name, first_name, last_name, bapt_date, bapt_parish_id FROM tabMember WHERE first_name IN ('Adaikala', 'Adaikalam') AND bapt_date IS NOT NULL
(Apply this same logic for Communion, Confirmation, Marriage, and Death using fhc_date, cnf_date, mrg_date, death_date in tabMember).
7. SACRAMENT FALLBACK (CRITICAL): Many sacrament dates are stored directly in `tabMember` without a formal registry entry in `tabBaptism`/etc. When searching for a specific person's sacrament details, you MUST ALWAYS query BOTH tables using a UNION. 
Example for Baptism:
SELECT 'Baptism' AS Sacrament, name, first_name, last_name, bapt_date, bapt_parish_id, bapt_minister FROM tabBaptism WHERE first_name IN ('Adaikala', 'Adaikalam') 
UNION ALL 
SELECT 'Baptism (from Member)', name, first_name, last_name, bapt_date, bapt_parish_id, NULL AS bapt_minister FROM tabMember WHERE first_name IN ('Adaikala', 'Adaikalam') AND bapt_date IS NOT NULL
(Apply this same UNION logic for Communion, Confirmation, Marriage, and Death, using NULL for columns that don't exist in tabMember).

### ROLE-BASED JURISDICTION BOUNDARIES (CRITICAL):
- **Bishop**: Has access to all records in all parishes/vicariates/dioceses.
- **Parish Priest**: Can ONLY access records belonging to his own parish.
- Current User Role: `{user_role}`
- Current User Parish: `{user_parish}`

If the Current User Role is "Parish Priest", you MUST append WHERE filter clauses to restrict queries to their parish:
- For `tabFamily`: `parish_id = '{user_parish}'`
- For `tabMember`: `parish_id = '{user_parish}'`
- For `tabBaptism`: `bapt_parish_id = '{user_parish}'` or `parish_id = '{user_parish}'`
- For `tabCommunion`: `fhc_place = '{user_parish}'` or `parish_id = '{user_parish}'`
- For `tabConfirmation`: `cnf_parish_id = '{user_parish}'` or `parish_id = '{user_parish}'`
- For `tabMarriage`: `mrg_parish_id = '{user_parish}'` or `parish_id = '{user_parish}'`
- For `tabAnointing Of Sick`: `anointing_parish_id = '{user_parish}'` or `parish_id = '{user_parish}'`
- For `tabDeath`: `death_parish_id = '{user_parish}'` or `parish_id = '{user_parish}'`

### QUERY GENERATION RULES:
- ONLY write SELECT queries. NEVER write INSERT, UPDATE, DELETE, or ALTER queries.
- NEVER use SELECT *. Always SELECT specific columns relevant to the user query (e.g., name, first_name, last_name, dob, bapt_date, etc.) to minimize token size.
- If the query could return a list of rows, ALWAYS apply LIMIT 20 to prevent query payload size limit errors.
- Do NOT guess or invent column names. Only use columns shown in the schemas.
- Do NOT wrap SQL in markdown or HTML. Just return the SQL string.
- If the question cannot be answered with a query on these tables, return "UNSUPPORTED".
"""),
    ("human", "User question: {enhanced_query}"),
])

def generate_sql_node(state: GraphState) -> GraphState:
    print("[generate_sql] Generating SQL query...")
    q = state["enhanced_query"] or state["question"]
    q_low = q.lower()
    role = state.get("user_role") or "Bishop"
    parish = state.get("user_parish") or ""

    # ── Branch A0: Deterministic Tamil / Tanglish Structured Parser ─────────
    orig_q = state.get("question", "")
    target_q = orig_q if any('\u0B80' <= c <= '\u0BFF' for c in orig_q) else q
    try:
        from koinonia_assistant.rag.tamil_query_parser import parse_tamil_query, build_canonical_sql_from_struct
        tamil_struct = parse_tamil_query(target_q, parish)
        canonical_sql, canonical_expl = build_canonical_sql_from_struct(tamil_struct, parish)
        if not canonical_sql and target_q != q:
            tamil_struct = parse_tamil_query(q, parish)
            canonical_sql, canonical_expl = build_canonical_sql_from_struct(tamil_struct, parish)
        if canonical_sql:
            print(f"[generate_sql] Matched Tamil Structured Intent '{tamil_struct.get('intent')}': {canonical_expl}")
            return {
                **state,
                "generated_sql": canonical_sql,
                "llm_explanation": canonical_expl
            }
    except Exception as e:
        print(f"[generate_sql] Tamil parser exception: {e}")

    fam_id_match = re.search(r'\b(?:Family\s*ID|Family)[:\s]+([0-9]+)\b', q, re.IGNORECASE) or re.search(r'\b(FAM-[0-9A-Z\-]+)\b', q, re.IGNORECASE)
    mem_id_match = re.search(r'\b(?:Member\s*ID|Member)[:\s]+([0-9]+)\b', q, re.IGNORECASE) or re.search(r'\b(MEM-[0-9A-Z\-]+)\b', q, re.IGNORECASE)

    is_baptism_req = any(k in q_low for k in ['bapt', 'babt', 'bap', 'ஞானஸ்நானம்', 'திருமுழுக்கு'])
    is_marriage_req = any(k in q_low for k in ['marr', 'wed', 'matrimony', 'திருமணம்', 'விவாகம்'])
    is_communion_req = any(k in q_low for k in ['comm', 'fhc', 'eucharist', 'நற்கருணை', 'முதல் நற்கருணை', 'புதுநன்மை', 'பொதுநன்மை', 'puthunanmai', 'pothunamai', 'pothunanmai'])
    is_confirmation_req = any(k in q_low for k in ['conf', 'cnf', 'உறுதிப்பூசுதல்', 'உறுதிபூசுதல்'])
    is_death_req = any(k in q_low for k in ['death', 'died', 'deceased', 'burial', 'funeral', 'இறப்பு', 'அடக்கம்'])
    is_sacraments_req = any(k in q_low for k in ['sacrament', 'sacraments', 'sacramental', 'திருவருட்சாதனம்', 'திருவருட்சாதனங்கள்', 'திருவருட்சாதனங்களின்', 'திருவருட்சாதன'])

    # ??? Branch A: Explicit Family or Member ID Match (Canonical Deterministic SQL) ??????
    if fam_id_match or mem_id_match:
        if fam_id_match:
            fam_id = fam_id_match.group(1).strip()
            if is_baptism_req:
                det_sql = f"""SELECT 
    f.reference AS `Family Head`,
    f.street AS `Address / Place`,
    CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) AS `Member Name`,
    m.relationship_id AS `Relationship`,
    m.gender AS `Gender`,
    m.dob AS `Date of Birth`,
    COALESCE(DATE_FORMAT(m.bapt_date, '%d-%b-%Y'), 'Not Recorded') AS `Baptism Date`,
    COALESCE(m.bapt_parish_id, f.parish_id, 'Parish') AS `Baptism Parish`,
    CASE WHEN m.bapt_date IS NOT NULL THEN 'Baptized' ELSE 'Pending / Not Recorded' END AS `Baptism Status`
FROM `tabFamily` f
LEFT JOIN `tabMember` m ON m.family_id = f.name
WHERE f.name = '{fam_id}' AND (f.parish_id LIKE '%{parish}%' OR '{parish}' = '')
ORDER BY FIELD(m.relationship_id, 'Head of Family', 'Husband', 'Wife', 'Father', 'Mother', 'Son', 'Daughter') ASC, m.creation ASC;"""
                expl = f"Retrieving baptism records for family {fam_id}"
            elif is_marriage_req:
                det_sql = f"""SELECT 
    f.reference AS `Family Head`,
    f.street AS `Address / Place`,
    CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) AS `Member Name`,
    m.relationship_id AS `Relationship`,
    m.gender AS `Gender`,
    m.dob AS `Date of Birth`,
    COALESCE(DATE_FORMAT(m.mrg_date, '%d-%b-%Y'), 'Not Recorded') AS `Marriage Date`,
    COALESCE(m.mrg_parish_id, f.parish_id, 'Parish') AS `Marriage Parish`,
    CASE WHEN m.mrg_date IS NOT NULL THEN 'Married' ELSE 'Unmarried / Not Recorded' END AS `Marriage Status`
FROM `tabFamily` f
LEFT JOIN `tabMember` m ON m.family_id = f.name
WHERE f.name = '{fam_id}' AND (f.parish_id LIKE '%{parish}%' OR '{parish}' = '')
ORDER BY FIELD(m.relationship_id, 'Head of Family', 'Husband', 'Wife', 'Father', 'Mother', 'Son', 'Daughter') ASC, m.creation ASC;"""
                expl = f"Retrieving marriage records for family {fam_id}"
            elif is_communion_req:
                det_sql = f"""SELECT 
    f.reference AS `Family Head`,
    f.street AS `Address / Place`,
    CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) AS `Member Name`,
    m.relationship_id AS `Relationship`,
    m.gender AS `Gender`,
    m.dob AS `Date of Birth`,
    COALESCE(DATE_FORMAT(m.fhc_date, '%d-%b-%Y'), 'Not Recorded') AS `First Communion Date`,
    COALESCE(m.fhc_parish_id, f.parish_id, 'Parish') AS `Communion Parish`,
    CASE WHEN m.fhc_date IS NOT NULL THEN 'Received' ELSE 'Pending / Not Recorded' END AS `Communion Status`
FROM `tabFamily` f
LEFT JOIN `tabMember` m ON m.family_id = f.name
WHERE f.name = '{fam_id}' AND (f.parish_id LIKE '%{parish}%' OR '{parish}' = '')
ORDER BY FIELD(m.relationship_id, 'Head of Family', 'Husband', 'Wife', 'Father', 'Mother', 'Son', 'Daughter') ASC, m.creation ASC;"""
                expl = f"Retrieving communion records for family {fam_id}"
            elif is_confirmation_req:
                det_sql = f"""SELECT 
    f.reference AS `Family Head`,
    f.street AS `Address / Place`,
    CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) AS `Member Name`,
    m.relationship_id AS `Relationship`,
    m.gender AS `Gender`,
    m.dob AS `Date of Birth`,
    COALESCE(DATE_FORMAT(m.cnf_date, '%d-%b-%Y'), 'Not Recorded') AS `Confirmation Date`,
    COALESCE(m.cnf_parish_id, f.parish_id, 'Parish') AS `Confirmation Parish`,
    CASE WHEN m.cnf_date IS NOT NULL THEN 'Confirmed' ELSE 'Pending / Not Recorded' END AS `Confirmation Status`
FROM `tabFamily` f
LEFT JOIN `tabMember` m ON m.family_id = f.name
WHERE f.name = '{fam_id}' AND (f.parish_id LIKE '%{parish}%' OR '{parish}' = '')
ORDER BY FIELD(m.relationship_id, 'Head of Family', 'Husband', 'Wife', 'Father', 'Mother', 'Son', 'Daughter') ASC, m.creation ASC;"""
                expl = f"Retrieving confirmation records for family {fam_id}"
            elif is_sacraments_req:
                det_sql = f"""SELECT 
    f.reference AS `Family Head`,
    f.street AS `Address / Place`,
    CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) AS `Member Name`,
    m.relationship_id AS `Relationship`,
    m.gender AS `Gender`,
    m.dob AS `Date of Birth`,
    COALESCE(DATE_FORMAT(m.bapt_date, '%d-%b-%Y'), '-') AS `Baptism Date`,
    COALESCE(DATE_FORMAT(m.fhc_date, '%d-%b-%Y'), '-') AS `First Communion Date`,
    COALESCE(DATE_FORMAT(m.cnf_date, '%d-%b-%Y'), '-') AS `Confirmation Date`,
    COALESCE(DATE_FORMAT(m.mrg_date, '%d-%b-%Y'), '-') AS `Marriage Date`
FROM `tabFamily` f
LEFT JOIN `tabMember` m ON m.family_id = f.name
WHERE f.name = '{fam_id}' AND (f.parish_id LIKE '%{parish}%' OR '{parish}' = '')
ORDER BY FIELD(m.relationship_id, 'Head of Family', 'Husband', 'Wife', 'Father', 'Mother', 'Son', 'Daughter') ASC, m.creation ASC;"""
                expl = f"Retrieving all sacramental records for family {fam_id}"
            else:
                det_sql = f"""SELECT 
    f.reference AS `Family Name / Head`,
    f.family_register_number AS `Family Card Number`,
    f.street AS `Address / Street`,
    f.mobile AS `Family Contact`,
    CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) AS `Member Name`,
    m.relationship_id AS `Relationship`,
    m.gender AS `Gender`,
    m.dob AS `Date of Birth`,
    m.mobile AS `Mobile`,
    m.living_status AS `Living Status`,
    CASE WHEN m.bapt_date IS NOT NULL THEN 'Yes' ELSE 'No' END AS `Baptism`,
    CASE WHEN m.fhc_date IS NOT NULL THEN 'Yes' ELSE 'No' END AS `First Holy Communion`,
    CASE WHEN m.cnf_date IS NOT NULL THEN 'Yes' ELSE 'No' END AS `Confirmation`,
    CASE WHEN m.mrg_date IS NOT NULL THEN 'Yes' ELSE 'No' END AS `Marriage`
FROM `tabFamily` f
LEFT JOIN `tabMember` m ON m.family_id = f.name
WHERE f.name = '{fam_id}' AND (f.parish_id LIKE '%{parish}%' OR '{parish}' = '')
ORDER BY FIELD(m.relationship_id, 'Head of Family', 'Husband', 'Wife', 'Father', 'Mother', 'Son', 'Daughter') ASC, m.creation ASC;"""
                expl = f"Retrieving complete family records for {fam_id}"

            print(f"[generate_sql] Explicit Family ID {fam_id}. Returning canonical SQL.")
            return {**state, "generated_sql": det_sql, "llm_explanation": expl}

        elif mem_id_match:
            mem_id = mem_id_match.group(1).strip()
            det_sql = f"""SELECT 
    CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) AS `Member Name`,
    f.reference AS `Family Head`,
    m.relationship_id AS `Relationship`,
    m.gender AS `Gender`,
    m.dob AS `Date of Birth`,
    COALESCE(m.street, f.street, 'N/A') AS `Address / Place`,
    COALESCE(DATE_FORMAT(m.bapt_date, '%d-%b-%Y'), 'Not Recorded') AS `Baptism Date`,
    COALESCE(DATE_FORMAT(m.fhc_date, '%d-%b-%Y'), 'Not Recorded') AS `First Communion Date`,
    COALESCE(DATE_FORMAT(m.cnf_date, '%d-%b-%Y'), 'Not Recorded') AS `Confirmation Date`,
    COALESCE(DATE_FORMAT(m.mrg_date, '%d-%b-%Y'), 'Not Recorded') AS `Marriage Date`
FROM `tabMember` m
LEFT JOIN `tabFamily` f ON f.name = m.family_id
WHERE m.name = '{mem_id}' AND (m.parish_id LIKE '%{parish}%' OR '{parish}' = '');"""
            expl = f"Retrieving records for member {mem_id}"
            print(f"[generate_sql] Explicit Member ID {mem_id}. Returning canonical SQL.")
            return {**state, "generated_sql": det_sql, "llm_explanation": expl}

        # ─── Branch B: Diagram, Trend, Predictive & Scenario Aggregations ─────────────
    import datetime
    current_year = datetime.datetime.now().year
    
    # Extract dynamic year filter
    match = re.search(r'(?:last|past|கடந்த)\s+(\d+)\s+(?:years|ஆண்டுகளில்|வருடங்களில்)', q_low)
    num_years = int(match.group(1)) if match else None
    
    is_predict_query = any(k in q_low for k in ['predict', 'prediction', 'predictive', 'forecast', 'forecasting', 'future', 'projection', 'எதிர்கால', 'கணிப்பு', 'கணிக்க'])
    if not num_years and is_predict_query:
        num_years = 5

    year_filter_bapt = f" AND YEAR(m.bapt_date) >= {current_year - num_years}" if num_years else ""
    year_filter_mrg = f" AND YEAR(m.mrg_date) >= {current_year - num_years}" if num_years else ""
    year_filter_cnf = f" AND YEAR(m.cnf_date) >= {current_year - num_years}" if num_years else ""
    year_filter_fhc = f" AND YEAR(m.fhc_date) >= {current_year - num_years}" if num_years else ""
    year_filter_death = f" AND YEAR(m.death_date) >= {current_year - num_years}" if num_years else ""
    year_filter_fam = f" AND YEAR(f.creation) >= {current_year - num_years}" if num_years else ""

    is_analytics_req = any(k in q_low for k in [
        'diagram', 'chart', 'graph', 'plot', 'trend', 'trends', 'growth', 'growing', 'ஆண்டுவாரியாக', 'எத்தனை', 'நிகழ்வுகள்', 'ஆண்டுகளில்', 'வருடங்களில்', 
        'weather growing', 'whether growing', 'வரைபடம்', 'விளக்கப்படம்', 
        'predict', 'prediction', 'predictive', 'forecast', 'forecasting', 'future', 
        'project', 'projection', 'எதிர்கால', 'கணிக்க', 'கணிப்பு', 'வளர்ச்சி'
    ])

    # Scenario: Parish division or sub-parish creation (e.g. splitting Yelagiri with 35 families)
    is_split_scenario = any(k in q_low for k in ['பிரித்து', 'கிளைப்பங்கு', 'split', 'divide', 'sub-parish', 'sub parish', 'branch parish', 'இரண்டாகப்'])
    if is_split_scenario:
        parish_target = parish or ""
        for p_name in ['Yelagiri', 'ஏலகிரி', 'Jolarpet', 'Tirupattur', 'Vellore']:
            if p_name.lower() in q_low or p_name in q:
                parish_target = 'Yelagiri' if ('yelagiri' in p_name.lower() or 'ஏலகிரி' in p_name) else p_name
                break
        p_where = f"WHERE f.parish_id LIKE '%{parish_target}%'" if parish_target else ""
        split_sql = f"""SELECT 
    COUNT(DISTINCT f.name) AS `Total Families`,
    COUNT(m.name) AS `Total Members`,
    SUM(CASE WHEN m.gender = 'Male' THEN 1 ELSE 0 END) AS `Male Members`,
    SUM(CASE WHEN m.gender = 'Female' THEN 1 ELSE 0 END) AS `Female Members`,
    COALESCE(f.parish_id, '{parish_target}') AS `Parish`
FROM `tabFamily` f
LEFT JOIN `tabMember` m ON m.family_id = f.name
{p_where}
GROUP BY f.parish_id;"""
        print(f"[generate_sql] Parish split scenario matched for {parish_target}.")
        return {**state, "generated_sql": split_sql, "llm_explanation": f"Retrieving baseline family and member statistics for parish {parish_target} to model division scenario"}

    # Confirmation Trend / Prediction
    if is_analytics_req and (is_confirmation_req or any(k in q_low for k in ['conf', 'cnf', 'உறுதிப்பூசுதல்', 'உறுதிபூசுதல்'])):
        p_filter = f"AND (m.parish_id LIKE '%{parish}%' OR f.parish_id LIKE '%{parish}%')" if parish else ""
        diag_sql = f"""SELECT 
    YEAR(m.cnf_date) AS `Year`,
    COUNT(m.name) AS `Confirmations Count`
FROM `tabMember` m
LEFT JOIN `tabFamily` f ON f.name = m.family_id
WHERE m.cnf_date IS NOT NULL
  {p_filter} {year_filter_cnf}
GROUP BY `Year`
ORDER BY `Year` ASC;"""
        print("[generate_sql] Confirmation analytics/predictive query matched.")
        return {**state, "generated_sql": diag_sql, "llm_explanation": "Aggregating annual confirmations for trend and predictive analysis"}

    # Baptism Trend / Prediction
    if is_analytics_req and (is_baptism_req or any(k in q_low for k in ['bapt', 'babt', 'bap', 'ஞானஸ்நானம்', 'திருமுழுக்கு'])):
        p_filter = f"AND (m.parish_id LIKE '%{parish}%' OR f.parish_id LIKE '%{parish}%')" if parish else ""
        diag_sql = f"""SELECT 
    YEAR(m.bapt_date) AS `Year`,
    COUNT(m.name) AS `Baptisms Count`
FROM `tabMember` m
LEFT JOIN `tabFamily` f ON f.name = m.family_id
WHERE m.bapt_date IS NOT NULL
  {p_filter} {year_filter_bapt}
GROUP BY `Year`
ORDER BY `Year` ASC;"""
        print("[generate_sql] Baptism analytics/predictive query matched.")
        return {**state, "generated_sql": diag_sql, "llm_explanation": "Aggregating annual baptisms for trend and predictive analysis"}

    # Marriage Trend / Prediction
    if is_analytics_req and (is_marriage_req or any(k in q_low for k in ['marr', 'wed', 'matrimony', 'திருமணம்'])):
        p_filter = f"AND (m.parish_id LIKE '%{parish}%' OR f.parish_id LIKE '%{parish}%')" if parish else ""
        diag_sql = f"""SELECT 
    YEAR(m.mrg_date) AS `Year`,
    COUNT(m.name) AS `Marriages Count`
FROM `tabMember` m
LEFT JOIN `tabFamily` f ON f.name = m.family_id
WHERE m.mrg_date IS NOT NULL
  {p_filter} {year_filter_mrg}
GROUP BY `Year`
ORDER BY `Year` ASC;"""
        print("[generate_sql] Marriage analytics/predictive query matched.")
        return {**state, "generated_sql": diag_sql, "llm_explanation": "Aggregating annual marriages for trend and predictive analysis"}

    # First Communion Trend / Prediction
    if is_analytics_req and (is_communion_req or any(k in q_low for k in ['comm', 'fhc', 'eucharist', 'நற்கருணை'])):
        p_filter = f"AND (m.parish_id LIKE '%{parish}%' OR f.parish_id LIKE '%{parish}%')" if parish else ""
        diag_sql = f"""SELECT 
    YEAR(m.fhc_date) AS `Year`,
    COUNT(m.name) AS `First Communions Count`
FROM `tabMember` m
LEFT JOIN `tabFamily` f ON f.name = m.family_id
WHERE m.fhc_date IS NOT NULL
  {p_filter} {year_filter_fhc}
GROUP BY `Year`
ORDER BY `Year` ASC;"""
        print("[generate_sql] First communion analytics/predictive query matched.")
        return {**state, "generated_sql": diag_sql, "llm_explanation": "Aggregating annual first communions for trend and predictive analysis"}

    # Death Trend / Prediction
    if is_analytics_req and (is_death_req or any(k in q_low for k in ['death', 'burial', 'deceased', 'இறப்பு'])):
        p_filter = f"AND (m.parish_id LIKE '%{parish}%' OR f.parish_id LIKE '%{parish}%')" if parish else ""
        diag_sql = f"""SELECT 
    YEAR(m.death_date) AS `Year`,
    COUNT(m.name) AS `Deaths Count`
FROM `tabMember` m
LEFT JOIN `tabFamily` f ON f.name = m.family_id
WHERE m.death_date IS NOT NULL
  {p_filter} {year_filter_death}
GROUP BY `Year`
ORDER BY `Year` ASC;"""
        print("[generate_sql] Death analytics/predictive query matched.")
        return {**state, "generated_sql": diag_sql, "llm_explanation": "Aggregating annual deaths for trend and predictive analysis"}

    # Family Registrations Trend / Prediction
    if is_analytics_req and any(k in q_low for k in ['family', 'families', 'member', 'parishioner', 'குடும்ப']):
        p_filter = f"WHERE (f.parish_id LIKE '%{parish}%')" if parish else ""
        diag_sql = f"""SELECT 
    YEAR(f.creation) AS `Year`,
    COUNT(f.name) AS `Registered Families Count`
FROM `tabFamily` f
{p_filter}
GROUP BY `Year`
ORDER BY `Year` ASC;"""
        print("[generate_sql] Family registration trend query matched.")
        return {**state, "generated_sql": diag_sql, "llm_explanation": "Aggregating annual family registrations for trend analysis"}

        # All Sacraments Breakdown & Totals (with optional specific year filter)
    is_all_sacraments = (
        is_sacraments_req or 
        'sacrament' in q_low or 
        'sacramental' in q_low or 
        'திருவருட்சாதன' in q or 
        'கூட்டுத்தொகை' in q or 
        'all sacraments' in q_low
    )
    if is_all_sacraments and not (is_baptism_req and not ('all' in q_low or 'அனைத்து' in q or 'கூட்டு' in q)):
        year_match = re.search(r'\b(19\d\d|20\d\d)\b', q)
        p_filter = f"AND (parish_id LIKE '%{parish}%')" if parish else ""
        if year_match:
            tgt_yr = year_match.group(1)
            diag_sql = f"""SELECT 'Baptism (ஞானஸ்நானம்)' AS `Sacrament`, COUNT(*) AS `Total Count` FROM `tabMember` WHERE bapt_date IS NOT NULL AND YEAR(bapt_date) = {tgt_yr} {p_filter}
UNION ALL
SELECT 'First Communion (முதல் நற்கருணை)', COUNT(*) FROM `tabMember` WHERE fhc_date IS NOT NULL AND YEAR(fhc_date) = {tgt_yr} {p_filter}
UNION ALL
SELECT 'Confirmation (உறுதிப்பூசுதல்)', COUNT(*) FROM `tabMember` WHERE cnf_date IS NOT NULL AND YEAR(cnf_date) = {tgt_yr} {p_filter}
UNION ALL
SELECT 'Marriage (திருமணம்)', COUNT(*) FROM `tabMember` WHERE mrg_date IS NOT NULL AND YEAR(mrg_date) = {tgt_yr} {p_filter};"""
            expl = f"Aggregating all sacraments breakdown and total count for year {tgt_yr}"
        else:
            diag_sql = f"""SELECT 'Baptism (ஞானஸ்நானம்)' AS `Sacrament`, COUNT(*) AS `Total Count` FROM `tabMember` WHERE bapt_date IS NOT NULL {p_filter}
UNION ALL
SELECT 'First Communion (முதல் நற்கருணை)', COUNT(*) FROM `tabMember` WHERE fhc_date IS NOT NULL {p_filter}
UNION ALL
SELECT 'Confirmation (உறுதிப்பூசுதல்)', COUNT(*) FROM `tabMember` WHERE cnf_date IS NOT NULL {p_filter}
UNION ALL
SELECT 'Marriage (திருமணம்)', COUNT(*) FROM `tabMember` WHERE mrg_date IS NOT NULL {p_filter};"""
            expl = "Aggregating sacraments breakdown for diagram visualization"
        print(f"[generate_sql] All sacraments query matched. Expl: {expl}")
        return {**state, "generated_sql": diag_sql, "llm_explanation": expl}

    # ─── Branch B3: Deterministic "Total baptized / statistics" queries ──────
    # These queries must ALWAYS produce the same SQL against the same tables.
    # The LLM must NEVER be allowed to invent column labels or choose tables freely.
    is_total_query = any(k in q_low for k in [
        'total', 'how many', 'count', 'statistics', 'stats',
        'எத்தனை', 'மொத்தம்', 'புள்ளிவிவரம்', 'கணக்கு'
    ])
    is_members_query = any(k in q_low for k in ['member', 'members', 'உறுப்பினர்', 'உறுப்பினர்கள்'])
    is_families_query = any(k in q_low for k in ['famil', 'families', 'குடும்பம்', 'குடும்பங்கள்'])
    is_bapt_count = (is_baptism_req and is_total_query and not is_analytics_req
                     and not fam_id_match and not mem_id_match)
    is_parish_stats = (is_total_query and (is_members_query or is_families_query)
                       and not is_analytics_req and not fam_id_match and not mem_id_match)

    p_filter_mem = f"AND m.parish_id LIKE '%{parish}%'" if parish else ""
    p_filter_fam = f"AND f.parish_id LIKE '%{parish}%'" if parish else ""

    # "How many members are baptized in my parish?" → count from tabMember.bapt_date
    # SINGLE consistent query — NEVER mix with tabBaptism for count queries
    if is_bapt_count and not is_families_query:
        bapt_count_sql = (
            "SELECT "
            "COUNT(*) AS `Total Baptized Members` "
            "FROM `tabMember` m "
            f"WHERE m.bapt_date IS NOT NULL {p_filter_mem};"
        )
        print("[generate_sql] Deterministic baptism count query matched.")
        return {**state,
                "generated_sql": bapt_count_sql,
                "llm_explanation": "Counting members with a recorded baptism date in tabMember"}

    # "Total baptized members AND family statistics in my parish"
    if is_parish_stats and is_baptism_req:
        stats_sql = (
            "SELECT "
            "'Families (with Card Number)' AS `Category`, "
            "COUNT(DISTINCT f.name) AS `Count` "
            "FROM `tabFamily` f "
            f"WHERE f.family_register_number IS NOT NULL {p_filter_fam} "
            "UNION ALL "
            "SELECT "
            "'Total Members' AS `Category`, "
            "COUNT(m.name) AS `Count` "
            "FROM `tabMember` m "
            f"WHERE 1=1 {p_filter_mem} "
            "UNION ALL "
            "SELECT "
            "'Baptized Members' AS `Category`, "
            "COUNT(m.name) AS `Count` "
            "FROM `tabMember` m "
            f"WHERE m.bapt_date IS NOT NULL {p_filter_mem};"
        )
        print("[generate_sql] Deterministic parish baptism+family stats query matched.")
        return {**state,
                "generated_sql": stats_sql,
                "llm_explanation": "Aggregating parish family count, total members, and baptized members from tabMember"}

    # "How many families / members in my parish?" (no baptism)
    if is_parish_stats and not is_baptism_req:
        gen_stats_sql = (
            "SELECT "
            "'Families' AS `Category`, "
            "COUNT(DISTINCT f.name) AS `Count` "
            "FROM `tabFamily` f "
            f"WHERE 1=1 {p_filter_fam} "
            "UNION ALL "
            "SELECT "
            "'Total Members' AS `Category`, "
            "COUNT(m.name) AS `Count` "
            "FROM `tabMember` m "
            f"WHERE 1=1 {p_filter_mem};"
        )
        print("[generate_sql] Deterministic general parish stats query matched.")
        return {**state,
                "generated_sql": gen_stats_sql,
                "llm_explanation": "Aggregating parish family count and total members"}

    # ??? Branch C: LLM Standard SQL Generation ??????
    response = llm.invoke(SQL_GEN_PROMPT.format_messages(
        current_date=datetime.date.today().strftime('%Y-%m-%d'),
        relevant_tables="\n\n".join(state["relevant_tables"]),
        relevant_fields="\n".join(state["relevant_fields"]),
        few_shot_examples=state["few_shot_examples"],
        user_role=state["user_role"],
        user_parish=state["user_parish"],
        enhanced_query=state["enhanced_query"]
    ))
    sql = response.content.strip().strip("`").strip("sql").strip()
    print(f"[generate_sql] SQL Generated:\n  {sql}")
    return {**state, "generated_sql": sql}


def validate_sql_node(state: GraphState) -> GraphState:
    print("[validate_sql] Validating SQL in sandbox...")
    sql = state["generated_sql"]
    
    if sql == "UNSUPPORTED":
        return {**state, "error_message": "Unsupported query"}

    # Basic safety checks
    sql_upper = sql.upper().strip()
    if not sql_upper.startswith("SELECT"):
        return {**state, "error_message": "Only SELECT queries are allowed."}
        
    for forbidden in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE", "CREATE"]:
        if re.search(r'\b' + forbidden + r'\b', sql_upper):
            return {**state, "error_message": f"Query contains forbidden keyword: {forbidden}"}

    # Strict Role-Based Jurisdiction Validation for Parish Priest
    role = state.get("user_role")
    parish = state.get("user_parish")
    if role == "Parish Priest" and parish:
        sql_lower = sql.lower()
        if parish.lower() not in sql_lower:
            err_msg = f"Security Violation: As a Parish Priest of '{parish}', you are restricted from querying records outside your parish. Ensure your query filters by parish: `{parish}`."
            print(f"[validate_sql] {err_msg}")
            return {**state, "error_message": err_msg}

    # EXPLAIN Sandbox check in Frappe MariaDB
    import frappe
    try:
        # Run EXPLAIN to validate syntax and table access
        explain_sql = f"EXPLAIN {sql}"
        frappe.db.sql(explain_sql)
        print("[validate_sql] SQL validated successfully.")
        return {**state, "error_message": ""}
    except Exception as e:
        error_msg = str(e)
        print(f"[validate_sql] SQL Validation Failed: {error_msg}")
        return {**state, "error_message": error_msg}

SQL_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL rewrite assistant for a MariaDB database in KOINONIA Parish Assistant.
The SQL query you generated failed with a database error.
Rewrite the SQL query to fix the error.

Table Schemas:
{relevant_tables}

Failed SQL:
{failed_sql}

Database Error:
{error_message}

Return ONLY the corrected SQL query — no explanation, no markdown."""),
    ("human", "Fix the SQL query."),
])

def rewrite_sql_node(state: GraphState) -> GraphState:
    print(f"[rewrite_sql] Rewriting SQL. Retry count: {state.get('retry_count', 0) + 1}...")
    retry = state.get("retry_count", 0) + 1
    
    response = llm.invoke(SQL_REWRITE_PROMPT.format_messages(
        relevant_tables="\n\n".join(state["relevant_tables"]),
        failed_sql=state["generated_sql"],
        error_message=state["error_message"]
    ))
    
    sql = response.content.strip().strip("`").strip("sql").strip()
    print(f"[rewrite_sql] New SQL generated:\n  {sql}")
    return {**state, "generated_sql": sql, "retry_count": retry}

def execute_sql_node(state: GraphState) -> GraphState:
    print("[execute_sql] Running SQL against MariaDB...")
    import frappe
    try:
        rows = frappe.db.sql(state["generated_sql"], as_dict=True)
        print(f"[execute_sql] Query returned {len(rows)} rows.")
        return {**state, "sql_result": rows, "error_message": ""}
    except Exception as e:
        error_msg = str(e)
        print(f"[execute_sql] Query execution failed: {error_msg}")
        return {**state, "error_message": error_msg}

RESPONSE_FORMAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the KOINONIA Parish Assistant.
Your task is to present the database query results to the user in a polite, helpful, and beautifully formatted response.

User Question: {question}
SQL Query Run: {sql}
Results (JSON):
{result_json}

Instructions:
1. If the results are empty, politely inform the user that no records were found matching their search.
2. If there are results, format them clearly. Use markdown lists or markdown tables to structure the data beautifully.
3. Be professional and brief. Keep the tone friendly and parish-focused.
4. Do NOT include technical jargon (like database names, column structures, or table join terms) unless specifically requested. Do NOT print the raw SQL unless requested.
5. If the query was flagged as UNSUPPORTED, explain politely that you are only able to query sacrament and family registers, and prompt them to rephrase.
6. IMPORTANT: If the User Question is in a non-English language (e.g., Tamil), you MUST write your final formatted response entirely in that same language.
7. CRITICAL: NEVER display internal database IDs like 'FAM-...' or 'MEM-...' to the user. Instead, use the Family Card Number ('family_register_number') or member names for display.
11. ANTI-HALLUCINATION RULE (CRITICAL): When presenting numbers and counts, you MUST use ONLY the exact column names and values returned by the SQL query. NEVER:
   - Add qualifiers or labels not in the data (e.g. do NOT say 'unique names', 'registered', 'active', 'verified' unless the SQL explicitly used DISTINCT or a filter for that).
   - Round, estimate, or interpret numbers. Show the exact value from the database.
   - Combine or subtract counts from different queries. Each number comes from exactly the SQL shown.
   - Invent subtotals or percentages not present in the result JSON.
   Example: If the SQL result has {{'Total Baptized Members': 43}}, display exactly '43'. Do NOT say '127 unique members' or any other figure.
12. CONSISTENCY RULE: The label you show to the user must exactly match what the SQL column alias says. If the column is `Total Baptized Members`, display it as 'Total Baptized Members'. Do not rename it.
8. PREDICTIVE ANALYSIS & FUTURE PROJECTIONS:
   - When the user asks to predict future growth, forecast trends, or project numbers based on historical records:
     a. Display the historical baseline data in a clear markdown table (Year | Historical Count).
     b. Calculate and state the observed growth rate / trend (e.g. average annual count, yearly net increase/decrease).
     c. Provide a mathematically projected forecast table for the future years requested (e.g., Year 2027 to 2031 with Projected Count).
     d. Provide practical pastoral and administrative insights (catechism preparations, community outreach, resource planning).
10. GRAND TOTALS & SACRAMENT SUMS:
   - When the user asks for the sum, total count, or breakdown of all sacraments (e.g. '2025 ஆம் ஆண்டில் மறைமாவட்டம் முழுவதும் நடைபெற்ற அனைத்து திருவருட்சாதனங்களின் கூட்டுத்தொகை'):
     a. Display the counts of each sacrament in a clean table.
     b. Compute and clearly display the Grand Total / Sum (`கூட்டுத்தொகை`) of all sacraments combined!
     c. If the question was asked in Tamil, the entire response must be written in Tamil.

9. WHAT-IF & SCENARIO ANALYSIS (Parish Division / Sub-Parish Creation):
   - When the user asks about splitting a parish or creating a new sub-parish with a specific number of families (e.g., 35 families separated from Yelagiri):
     a. Present the current parish baseline (Total Families, Total Members).
     b. Show a structured comparison breakdown: Current Parish vs Remaining Main Parish vs New Sub-Parish.
     c. Provide constructive pastoral insights on governance, pastoral care, and community impact.
"""),
    ("human", "Format the response."),
])


def _smart_fallback_format(raw_results: list, question: str, sql: str, is_tamil: bool) -> str:
    """
    Smart fallback renderer used when the LLM formatting call fails.
    Detects query intent from question + SQL and renders either:
      - A properly formatted aggregate/statistics summary (for count/total queries)
      - A markdown table with a record count header (for list queries)
    instead of the generic "Found N records" string.
    """
    import re as _re

    if not raw_results:
        return ("இந்த நிபந்தனைகளுக்கு பொருந்தும் பதிவுகள் இல்லை." if is_tamil
                else "No matching records found in the parish registry for your inquiry.")

    q_low = (question or "").lower()
    sql_low = (sql or "").lower()

    # --- Detect if this is an AGGREGATE / STATS query ---------------------------
    aggregate_signals = [
        'count', 'sum(', 'total', 'statistics', 'stats', 'how many', 'breakdown',
        'group by', 'aggregate', 'sacrament names', 'grand total',
        'மொத்தம்', 'புள்ளிவிவரம்', 'எத்தனை', 'count(*)', 'count(', 'sum('
    ]
    is_aggregate = (
        any(s in q_low for s in aggregate_signals) or
        any(s in sql_low for s in ['count(', 'sum(', 'group by'])
    )

    lines = []

    if is_aggregate and isinstance(raw_results, list) and raw_results:
        # ── Aggregate: render a clean summary table ───────────────────────────
        if is_tamil:
            lines.append("### 📊 உங்கள் பங்கின் புள்ளிவிவரங்கள்\n")
        else:
            lines.append("### 📊 Parish Statistics\n")

        first = raw_results[0]
        if isinstance(first, dict):
            keys = list(first.keys())

            # Single-row aggregate (e.g. one row with total_members, total_families)
            if len(raw_results) == 1:
                for k, v in first.items():
                    label = k.replace('_', ' ').title()
                    lines.append(f"- **{label}**: {v}")
            else:
                # Multi-row aggregate (e.g. sacrament breakdown)
                # Render as markdown table
                lines.append("| " + " | ".join(k.replace('_', ' ').title() for k in keys) + " |")
                lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
                grand_total = 0
                total_col = None
                for k in keys:
                    if any(t in k.lower() for t in ['count', 'total', 'sum']):
                        total_col = k
                        break
                for r in raw_results:
                    lines.append("| " + " | ".join(str(r.get(k, '')) for k in keys) + " |")
                    if total_col:
                        try:
                            grand_total += int(r.get(total_col, 0) or 0)
                        except (ValueError, TypeError):
                            pass
                if total_col and grand_total:
                    if is_tamil:
                        lines.append(f"\n**மொத்தம்: {grand_total}**")
                    else:
                        lines.append(f"\n**Grand Total: {grand_total}**")

    else:
        # ── List/Record query: render markdown table with count header ─────────
        n = len(raw_results)
        if is_tamil:
            lines.append(f"உங்கள் வினவலுக்கு **{n}** பதிவுகள் கண்டறியப்பட்டன:\n")
        else:
            lines.append(f"Found **{n}** record{'s' if n != 1 else ''} matching your query:\n")

        if isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], dict):
            # Strip internal ID columns for display
            skip_cols = set()
            for r in raw_results[:1]:
                for k in r:
                    if _re.match(r'^(name|id)$', k, _re.IGNORECASE):
                        val = str(r.get(k, ''))
                        if _re.match(r'^(FAM|MEM|fam|mem)-', val):
                            skip_cols.add(k)

            headers = [k for k in raw_results[0].keys() if k not in skip_cols]
            if headers:
                lines.append("| " + " | ".join(h.replace('_', ' ').title() for h in headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for r in raw_results[:20]:
                    lines.append("| " + " | ".join(str(r.get(h, '') or '') for h in headers) + " |")
                if n > 20:
                    if is_tamil:
                        lines.append(f"\n_மேலும் {n - 20} பதிவுகள் உள்ளன. Excel/PDF-ஐ பயன்படுத்தி அனைத்தையும் பதிவிறக்கவும்._")
                    else:
                        lines.append(f"\n_...and {n - 20} more records. Use Excel/PDF to download all._")

    return "\n".join(lines)


def format_response_node(state: GraphState) -> GraphState:
    print("[format_response] Formatting final LLM response...")
    orig_q = state.get("original_query") or state["question"]
    lang = state.get("detected_language") or ("ta" if any('\u0B80' <= c <= '\u0BFF' for c in orig_q) else "en")
    is_ta = lang == "ta"
    auth_ctx = state.get("authorization_context") or {}
    scope_type = auth_ctx.get("scope_type", "PARISH" if state.get("user_parish") else "DIOCESE")
    scope_name = auth_ctx.get("scope_name") or state.get("user_parish") or "Authorized Scope"

    if state.get("error_message") == "Unsupported query" or state.get("generated_sql") == "UNSUPPORTED":
        if is_ta:
            ans = (
                "மன்னிக்கவும், என்னால் திருவருட்சாதன பதிவேடுகள் (திருமுழுக்கு, முதல் நற்கருணை, உறுதிப்பூசுதல், திருமணம், இறப்பு) "
                "மற்றும் பங்கு குடும்ப/உறுப்பினர் பதிவேடுகளை மட்டுமே தேட முடியும். "
                "தயவுசெய்து இந்த பதிவேடுகள் தொடர்பான கேள்வியைக் கேளுங்கள், நான் உங்களுக்காக தேடித் தருகிறேன்!"
            )
        else:
            ans = (
                "I'm sorry, but I can only search sacrament registers (Baptism, First Holy Communion, Confirmation, Marriage, and Death) "
                "and parish Family/Member registers. Please ask a question related to these registries, and I'll be happy to search for you!"
            )
        return {**state, "final_answer": ans}

    if state.get("error_message"):
        ans = (
            "I encountered an issue searching the parish registries:\n\n"
            f"`{state['error_message']}`\n\n"
            "Please try selecting one of the suggested search options below."
        )
        return {**state, "final_answer": ans}

    req_id = state.get("request_id", "N/A")
    c_intent = state.get("classified_intent", "GENERAL_DATABASE_QUERY")
    det_reply = state.get("deterministic_reply")

    print(f"[LANGGRAPH] LLM / FORMATTER NODE | request_id={req_id} | intent={c_intent} | lang={lang} | scope={scope_name}")

    if det_reply:
        # For analytical, historical, trend, forecast, and comparison queries, invoke LLM to append grounded executive interpretation
        if c_intent in ("HISTORICAL_ANALYSIS", "STATISTICAL_ANALYSIS", "TREND_ANALYSIS", "FORECAST", "COMPARISON"):
            meta = state.get("analysis_metadata") or {}
            datasets = meta.get("datasets") or {}
            summary_payload = {}
            for k, ds in datasets.items():
                summary_payload[ds.get("ta_label" if is_ta else "metric_label", k)] = {
                    "scope_type": ds.get("scope_type"),
                    "scope_name": ds.get("scope_name"),
                    "authorized_record_count": ds.get("authorized_record_count"),
                    "historical_period": f"{ds.get('start_year')}-{ds.get('end_year')}",
                    "statistics": {
                        sk: sv for sk, sv in (ds.get("stats") or {}).items() if sk != "yoy_rows"
                    },
                    "forecast": {
                        "method": (ds.get("forecast") or {}).get("method"),
                        "forecast_period": (ds.get("forecast") or {}).get("forecast_period"),
                        "first_3_projections": ((ds.get("forecast") or {}).get("forecast_rows") or [])[:3],
                        "last_projection": ((ds.get("forecast") or {}).get("forecast_rows") or [-1])[-1] if ((ds.get("forecast") or {}).get("forecast_rows")) else None,
                    } if ds.get("forecast") else None,
                }
            try:
                if is_ta:
                    analytical_prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are the Pastoral & Statistical Analyst for the Koinonia Assistant.
STRICT TAMIL & SECURITY RULES (Sections 36, 37, 42, 68):
1. Understand the user's Tamil query directly. Preserve Tamil terminology.
2. Do NOT transliterate Tamil into Tanglish.
3. Answer in natural, idiomatic Catholic Tamil using ONLY the configured Koinonia canonical terminology:
   - Baptism -> திருமுழுக்கு
   - First Holy Communion -> முதல் நற்கருணை
   - Confirmation -> உறுதிப்பூசுதல்
   - Marriage -> திருமணம்
   - Parish -> பங்கு ({scope_name})
4. NEVER invent or alter numbers. Cite ONLY the verified statistics for {scope_name} provided in the JSON.
5. Keep your commentary concise (5-8 lines) under two short Tamil bullets:
   - **பதிவு செய்யப்பட்ட தரவுகளின் சுருக்கம் (Observed Data Findings)**
   - **புள்ளிவிவர மற்றும் மேய்ப்புப் பணி விளக்கம் (Statistical & Pastoral Interpretation)**"""),
                        ("human", "Original Tamil Question: {question}\nAuthorized Scope: {scope_name}\nIntent: {intent}\nVerified Calculations JSON:\n{stats_json}")
                    ])
                    llm_resp = llm.invoke(analytical_prompt.format_messages(
                        question=orig_q,
                        scope_name=scope_name,
                        intent=c_intent,
                        stats_json=json.dumps(summary_payload, default=str, ensure_ascii=False, indent=2)
                    ))
                    commentary = llm_resp.content.strip()
                    if commentary:
                        det_reply = f"{det_reply}\n\n#### 💡 புள்ளிவிவர மற்றும் மேய்ப்புப் பணி விளக்கம்\n{commentary}"
                else:
                    analytical_prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are the Pastoral & Statistical Analyst for the Koinonia Assistant.
You are given VERIFIED statistical & forecasting results computed strictly within the user's authorized scope ({scope_name}).
STRICT RULES (Sections 68, 69, 72):
1. NEVER invent, alter, or guess any numbers. Cite ONLY the exact statistics and projected ranges provided in the JSON for {scope_name}.
2. NEVER refer to 'Diocesan Registry Benchmark' if the scope is a Parish ({scope_name}). Always state that the analysis is strictly scoped to {scope_name}.
3. Explicitly separate your explanation into two short bullet sections:
   - **Observed Data Findings ({scope_name}):** Summarize the exact historical counts, peak/low years, percentage change, and OLS regression slope.
   - **Statistical & Pastoral Interpretation:** Explain what the trend direction, volatility, and (if present) projected forecast ranges mean in practice.
4. Keep the commentary concise (6-10 lines maximum)."""),
                        ("human", "User Question: {question}\nAuthorized Scope: {scope_name}\nIntent: {intent}\nVerified Calculations JSON:\n{stats_json}")
                    ])
                    llm_resp = llm.invoke(analytical_prompt.format_messages(
                        question=orig_q,
                        scope_name=scope_name,
                        intent=c_intent,
                        stats_json=json.dumps(summary_payload, default=str, indent=2)
                    ))
                    commentary = llm_resp.content.strip()
                    if commentary:
                        det_reply = f"{det_reply}\n\n#### 💡 Executive Statistical & Pastoral Interpretation ({scope_name})\n{commentary}"
            except Exception as llm_err:
                print(f"[LANGGRAPH] LLM analytical commentary fallback: {llm_err}")

        # Layer 8 Response Scope Validation (Section 72)
        if scope_type == "PARISH" and "Diocesan Registry Benchmark" in det_reply:
            det_reply = det_reply.replace("Diocesan Registry Benchmark", scope_name)

        print(f"[LANGGRAPH] RESPONSE NODE | request_id={req_id} | intent={c_intent} | status=VERIFIED_COMPLETE")
        return {**state, "final_answer": det_reply}

    raw_results = state["sql_result"]
    if not raw_results or len(raw_results) == 0:
        if is_ta:
            ans = f"**{scope_name}** பதிவேட்டில் இந்த நிபந்தனைகளுக்குப் பொருந்தும் பதிவுகள் எதுவும் இல்லை."
        else:
            ans = f"No matching records found in the **{scope_name}** registry for your inquiry."
        return {**state, "final_answer": ans}

    if isinstance(raw_results, list) and len(raw_results) > 20:
        raw_results = raw_results[:20]
    result_json = json.dumps(raw_results, default=str, ensure_ascii=False, indent=1)

    try:
        response = llm.invoke(RESPONSE_FORMAT_PROMPT.format_messages(
            question=orig_q,
            sql=state["generated_sql"],
            result_json=result_json
        ))
        raw_ans = response.content.strip()
    except Exception as e:
        print(f"[format_response] LLM formatting failed: {e}. Using smart fallback.")
        raw_ans = _smart_fallback_format(raw_results, orig_q, state.get('generated_sql', ''), is_ta)

    clean_ans = re.sub(r'\s*\(?\s*ID\s*:\s*(?:FAM|MEM|fam|mem)-[A-Za-z0-9\-]+\)?', '', raw_ans, flags=re.IGNORECASE)
    clean_ans = re.sub(r'\s*\(?\s*(?:Family|Member)\s*ID\s*:\s*[A-Za-z0-9\-]+\)?', '', clean_ans, flags=re.IGNORECASE)
    clean_ans = re.sub(r'\b(?:FAM|MEM)-[A-Za-z0-9\-]+\b', '', clean_ans)
    clean_ans = re.sub(r'\n\s*[-*]\s*\*\*(?:Family|Member) ID:\*\*.*', '', clean_ans)
    clean_ans = re.sub(r'\n\s*[-*]\s*(?:Family|Member) ID:.*', '', clean_ans)

    print(f"[LANGGRAPH] RESPONSE NODE | request_id={req_id} | intent={c_intent} | status=SQL_FORMAT_COMPLETE")
    return {**state, "final_answer": clean_ans.strip()}


def database_lookup_node(state: GraphState) -> GraphState:
    """
    LangGraph node that executes verified, authorization-constrained database lookups for:
    - LIST (including qualified sacrament member lists like 'List any 10 members who got 3 Sacrements')
    - COUNT (Strictly scoped to the user's authorized parish when scope_type == 'PARISH')
    - MEMBER_SEARCH / FAMILY_SEARCH / SACRAMENT_SEARCH (including Family Card & Tamil names)
    """
    req_id = state.get("request_id", "N/A")
    question = state.get("original_query") or state["question"]
    c_intent = state.get("classified_intent", "LIST")
    intent_info = state.get("intent_info") or {}
    auth_ctx = state.get("authorization_context") or {}
    scope_type = auth_ctx.get("scope_type", "PARISH" if state.get("user_parish") else "DIOCESE")
    scope_name = auth_ctx.get("scope_name") or state.get("user_parish") or "Diocesan Registry"
    # tabMember and tabFamily store parish_id as the human parish name ('Yelagiri Parish')
    user_parish = (scope_name if scope_type == "PARISH" else None) or state.get("user_parish")
    user_diocese = auth_ctx.get("diocese_id") or state.get("user_diocese")
    is_ta = state.get("detected_language") == "ta"

    print(f"[LANGGRAPH] DATABASE NODE | request_id={req_id} | intent={c_intent} | scope_type={scope_type} | scope_name={scope_name}")

    # 1. Qualified Member Sacrament List (e.g. "List any 10 members who got 3 Sacrements")
    if c_intent == "LIST" and intent_info.get("sub_intent") == "QUALIFIED_MEMBER_SACRAMENT_LIST":
        ensure_frappe_connected()
        from koinonia_assistant.rag.analytics_engine import execute_qualified_member_sacrament_list
        res = execute_qualified_member_sacrament_list(
            sacrament_count_filter=intent_info.get("sacrament_count_filter"),
            metrics=intent_info.get("metrics") or [],
            limit=intent_info.get("limit", 10),
            user_parish=user_parish,
            user_diocese=user_diocese,
            auth_ctx=auth_ctx,
        )
        return {
            **state,
            "route": "handled",
            "deterministic_reply": res["reply"],
            "generated_sql": res["generated_sql"],
            "sql_result": res["data"],
            "sql_validation_result": "PASSED_ALL_SECURITY_CHECKS",
            "authorized_record_count": res["records_retrieved"],
            "suggested_questions": [
                "List any 10 members who got 2 Sacraments",
                f"Show baptism statistics for the last 10 years in {scope_name}",
                "Compare baptism, confirmation and marriage over the last 10 years",
            ],
        }

    # 2. Standard LIST_MEMBERS / LIST_FAMILIES
    if c_intent == "LIST":
        sub = intent_info.get("sub_intent")
        req_count = intent_info.get("limit", 10)
        name_filter = intent_info.get("filter")
        if sub == "LIST_FAMILIES":
            res = handle_list_families(requested_count=req_count, user_parish=user_parish, user_diocese=user_diocese)
        else:
            res = handle_list_members(requested_count=req_count, name_filter=name_filter, user_parish=user_parish, user_diocese=user_diocese)
        return {
            **state,
            "route": "handled",
            "deterministic_reply": res["reply"],
            "generated_sql": res.get("generated_sql", ""),
            "sql_result": res.get("data", []),
            "sql_validation_result": "PASSED_ALL_SECURITY_CHECKS",
            "authorized_record_count": len(res.get("data", [])),
            "suggested_questions": res.get("suggested_questions", []),
        }

    # 3. COUNT queries (Strictly scoped to authorized parish when scope_type == 'PARISH' — NEVER leaks diocesan total)
    if c_intent == "COUNT":
        sub = intent_info.get("sub_intent")
        metrics = intent_info.get("metrics") or []
        if sub == "COUNT_MEMBERS" and not metrics:
            res = handle_count_members(user_parish=user_parish, user_diocese=user_diocese)
            return {
                **state,
                "route": "handled",
                "deterministic_reply": res["reply"],
                "generated_sql": res.get("generated_sql", ""),
                "sql_result": res.get("data", []),
                "sql_validation_result": "PASSED_ALL_SECURITY_CHECKS",
                "authorized_record_count": 1,
                "suggested_questions": res.get("suggested_questions", []),
            }
        elif sub == "COUNT_FAMILIES" and not metrics:
            res = handle_count_families(user_parish=user_parish, user_diocese=user_diocese)
            return {
                **state,
                "route": "handled",
                "deterministic_reply": res["reply"],
                "generated_sql": res.get("generated_sql", ""),
                "sql_result": res.get("data", []),
                "sql_validation_result": "PASSED_ALL_SECURITY_CHECKS",
                "authorized_record_count": 1,
                "suggested_questions": res.get("suggested_questions", []),
            }
        elif metrics:
            ensure_frappe_connected()
            import frappe
            from koinonia_assistant.rag.analytics_engine import SACRAMENT_METRICS, authorization_sql_validator
            m_key = metrics[0]
            meta = SACRAMENT_METRICS[m_key]
            tbl = meta["table"]
            d_col = meta["date_col"]
            p_cols = meta["parish_cols"]

            yr_match = re.search(r"\b(19\d\d|20\d\d)\b", question)
            target_yr = int(yr_match.group(1)) if yr_match else None

            where_c = ["1=1"]
            params = []
            if target_yr:
                where_c.append(f"YEAR(`{d_col}`) = %s")
                params.append(target_yr)

            if scope_type == "PARISH" and user_parish:
                p_or = " OR ".join([f"`{c}` = %s OR `{c}` LIKE %s" for c in p_cols])
                where_c.append(f"({p_or})")
                for _ in p_cols:
                    params.extend([user_parish, f"%{user_parish}%"])

            sql_exec = f"SELECT COUNT(*) AS total_count FROM `{tbl}` WHERE {' AND '.join(where_c)}"
            val_res = authorization_sql_validator(sql_exec, auth_ctx)
            if not val_res["valid"]:
                raise PermissionError(f"SQL Validation Failed: {val_res['reason']}")

            cnt_val = int(frappe.db.sql(sql_exec, tuple(params))[0][0])
            yr_str = f" in **{target_yr}**" if target_yr else ""
            if is_ta:
                reply = (
                    f"### 📊 {scope_name} — {meta['ta_label']} எண்ணிக்கை\n"
                    f"- **அனுமதிக்கப்பட்ட பங்கு ({scope_name}):** மொத்தம் **`{cnt_val:,}`** {meta['ta_label']} பதிவுகள் உள்ளன."
                )
            else:
                reply = (
                    f"### 📊 {scope_name} — Verified {meta['label']} Record Count{yr_str}\n"
                    f"- **Authorized Scope (`{scope_name}`):** **`{cnt_val:,}`** {meta['label'].lower()} record(s){yr_str}"
                )
            data_rows = [
                {"Scope": scope_name, "Sacrament": meta["ta_label"] if is_ta else meta["label"], "Year": target_yr or "All Years", "Total Records": cnt_val}
            ]
            return {
                **state,
                "route": "handled",
                "deterministic_reply": reply,
                "generated_sql": sql_exec,
                "sql_result": data_rows,
                "sql_validation_result": val_res["reason"],
                "authorized_record_count": cnt_val,
                "suggested_questions": [
                    f"What was the number of {meta['label'].lower()}s each year for the last 10 years?",
                    f"How has {meta['label'].lower()} changed over the last 10 years?",
                    f"Forecast {meta['label'].lower()} for the next 10 years",
                ],
            }

    # 4. Family Card Lookup (Sections 64 & 67: Verify card belongs to authorized parish BEFORE retrieving members)
    card_match = re.search(r"\b([A-Z]{2,5}/\d{1,5})\b", question, re.IGNORECASE)
    if card_match:
        ensure_frappe_connected()
        import frappe
        card_code = card_match.group(1).upper()
        where_f = ["UPPER(family_register_number) = %s"]
        params_f: List[Any] = [card_code]
        if scope_type == "PARISH" and user_parish:
            where_f.append("(parish_id = %s OR parish_id LIKE %s)")
            params_f.extend([user_parish, f"%{user_parish}%"])
        fam_rows = frappe.db.sql(
            f"SELECT name, parish_id, family_register_number, reference FROM `tabFamily` WHERE {' AND '.join(where_f)} LIMIT 1",
            tuple(params_f),
            as_dict=True,
        )
        if not fam_rows:
            msg = (
                f"உங்கள் அனுமதிக்கப்பட்ட பங்கு எல்லையில் (**{scope_name}**) `{card_code}` என்ற குடும்ப அட்டை விவரம் எதுவும் கண்டறியப்படவில்லை."
                if is_ta
                else f"No family record matching `{card_code}` is available within your authorized parish scope (**{scope_name}**)."
            )
            return {
                **state,
                "route": "handled",
                "deterministic_reply": msg,
                "generated_sql": "",
                "sql_result": [],
                "sql_validation_result": "BLOCKED_UNAUTHORIZED_FAMILY_CARD",
                "authorized_record_count": 0,
                "suggested_questions": [],
            }
        fid = fam_rows[0]["name"]
        bundle = fetch_full_family_bundle(fid, user_parish)
        if bundle.get("family"):
            scope = determine_response_scope(question)
            head_mem = bundle.get("members", [{}])[0] if bundle.get("members") else {}
            reply, suggestions = render_scoped_response(scope, head_mem, None, bundle)
            return {
                **state,
                "route": "handled",
                "deterministic_reply": reply,
                "generated_sql": f"SELECT * FROM `tabFamily` WHERE name = '{fid}' AND parish_id = '{user_parish or ''}'",
                "sql_result": bundle.get("members", []),
                "sql_validation_result": "PASSED_ALL_SECURITY_CHECKS",
                "authorized_record_count": len(bundle.get("members", [])),
                "suggested_questions": suggestions,
            }

    # 5. Explicit ID or 3-Level Member / Family / Sacrament Resolution
    fam_id_match = re.search(r'\b(?:Family\s*ID|Family)[:\s]+([0-9]+)\b', question, re.IGNORECASE) or re.search(r'\bFAM-([0-9A-Z\-]+)\b', question, re.IGNORECASE)
    mem_id_match = re.search(r'\b(?:Member\s*ID|Member)[:\s]+([0-9]+)\b', question, re.IGNORECASE) or re.search(r'\bMEM-([0-9A-Z\-]+)\b', question, re.IGNORECASE)

    if fam_id_match or mem_id_match:
        intent = detect_query_intent(question)
        if mem_id_match and (intent in ["sacraments", "sacrament_details"] or not fam_id_match):
            mid = mem_id_match.group(1)
            sac_bundle = fetch_member_sacrament_bundle(mid, user_parish)
            if sac_bundle.get("member"):
                m = sac_bundle.get("member")
                m["member_id"] = m.get("name")
                m["full_name"] = sac_bundle.get("full_name") or f"{m.get('first_name', '')} {m.get('last_name', '')}".strip()
                fam_bundle = fetch_full_family_bundle(m.get("family_id"), user_parish)
                m["family_register_number"] = fam_bundle.get("family", {}).get("family_register_number") or m.get("family_id")
                scope = determine_response_scope(question)
                reply, suggestions = render_scoped_response(scope, m, sac_bundle, fam_bundle)
                data_payload = fam_bundle.get("members", []) if scope in ['FAMILY_MEMBERS_ONLY', 'FAMILY_DETAILS'] else [m]
                return {
                    **state,
                    "route": "handled",
                    "deterministic_reply": reply,
                    "generated_sql": f"SELECT * FROM `tabMember` WHERE name = '{mid}'",
                    "sql_result": data_payload,
                    "sql_validation_result": "PASSED_ALL_SECURITY_CHECKS",
                    "authorized_record_count": len(data_payload),
                    "suggested_questions": suggestions,
                }
        elif fam_id_match:
            fid = fam_id_match.group(1)
            bundle = fetch_full_family_bundle(fid, user_parish)
            if bundle.get("family"):
                scope = determine_response_scope(question)
                head_mem = bundle.get("members", [{}])[0] if bundle.get("members") else {}
                reply, suggestions = render_scoped_response(scope, head_mem, None, bundle)
                return {
                    **state,
                    "route": "handled",
                    "deterministic_reply": reply,
                    "generated_sql": f"SELECT * FROM `tabFamily` WHERE name = '{fid}'",
                    "sql_result": bundle.get("members", []),
                    "sql_validation_result": "PASSED_ALL_SECURITY_CHECKS",
                    "authorized_record_count": len(bundle.get("members", [])),
                    "suggested_questions": suggestions,
                }

    # 3-Level Member / Family / Sacrament Resolution (Supports both English & Tamil names via person_name entity)
    person_name = intent_info.get("person_name") or extract_person_name_from_query(question)
    if person_name:
        # Build internal lookup query for resolve_member_and_family without mutating state["question"]
        lookup_q = question
        if is_ta and person_name and not is_tamil(person_name):
            sac_code = intent_info.get("sacrament") or "BAPTISM"
            lookup_q = f"What is the {sac_code.lower().replace('_', ' ')} status of {person_name}?"
        person_res = resolve_member_and_family(
            lookup_q,
            user_parish=user_parish,
            user_diocese=user_diocese
        )
        if person_res.get("status") == "exact":
            m = person_res.get("matched_member")
            sac_bundle = person_res.get("sacrament_bundle")
            fam_bundle = person_res.get("family_bundle")
            scope = person_res.get("response_scope") or determine_response_scope(lookup_q)
            reply, suggestions = render_scoped_response(scope, m, sac_bundle, fam_bundle)
            data_payload = fam_bundle.get("members", []) if scope in ['FAMILY_MEMBERS_ONLY', 'FAMILY_DETAILS'] else [m]
            return {
                **state,
                "route": "handled",
                "deterministic_reply": reply,
                "generated_sql": f"SELECT * FROM `tabMember` WHERE name = '{m.get('member_id')}' AND parish_id = '{user_parish or ''}'",
                "sql_result": data_payload,
                "sql_validation_result": "PASSED_ALL_SECURITY_CHECKS",
                "authorized_record_count": len(data_payload),
                "suggested_questions": suggestions,
            }
        elif person_res.get("status") == "candidates":
            cands = person_res.get("candidates", [])
            disambig_obj = {
                "message": (
                    "பின்வரும் பங்கு உறுப்பினர்களில் யாரைக் குறிப்பிடுகிறீர்கள்?"
                    if is_ta
                    else "Did you mean one of the following parishioners?"
                ),
                "options": cands[:3]
            }
            return {
                **state,
                "route": "handled",
                "deterministic_reply": (
                    "பல பொருத்தமான பதிவுகள் கண்டறியப்பட்டுள்ளன. விவரங்களைப் பார்க்க கீழே உள்ளவற்றில் ஒன்றைத் தேர்ந்தெடுக்கவும்:"
                    if is_ta
                    else "Multiple matching records found. Please select an option to view details:"
                ),
                "generated_sql": "",
                "sql_result": [],
                "disambiguation": disambig_obj,
                "sql_validation_result": "PASSED_ALL_SECURITY_CHECKS",
                "authorized_record_count": len(cands[:3]),
                "suggested_questions": [c.get("prompt") for c in cands[:3]],
            }
        elif person_res.get("status") == "not_found":
            parish_label = f" in {user_parish}" if user_parish else ""
            return {
                **state,
                "route": "handled",
                "deterministic_reply": f"No parishioner or family records found matching '{person_name}'{parish_label}. Please verify the spelling or check with the parish office.",
                "generated_sql": "",
                "sql_result": [],
                "sql_validation_result": "PASSED_ALL_SECURITY_CHECKS",
                "authorized_record_count": 0,
                "suggested_questions": [],
            }

    return {**state, "route": "fallback_sql"}


def analytics_node(state: GraphState) -> GraphState:
    """
    LangGraph node for HISTORICAL_ANALYSIS, STATISTICAL_ANALYSIS, TREND_ANALYSIS, and COMPARISON.
    Receives authorization_context and queries ONLY the user's authorized scope (Sections 54–56, 68–70).
    """
    req_id = state.get("request_id", "N/A")
    intent_info = state.get("intent_info") or {}
    auth_ctx = state.get("authorization_context") or {}
    c_intent = state.get("classified_intent", "HISTORICAL_ANALYSIS")
    print(
        f"[LANGGRAPH] ANALYTICS NODE | request_id={req_id} | intent={c_intent} "
        f"| scope_type={auth_ctx.get('scope_type')} | scope_name={auth_ctx.get('scope_name')} "
        f"| metrics={intent_info.get('metrics')}"
    )

    ensure_frappe_connected()
    from koinonia_assistant.rag.analytics_engine import run_analytics_or_forecast_pipeline

    res = run_analytics_or_forecast_pipeline(
        intent_info=intent_info,
        question=state.get("original_query") or state["question"],
        user_parish=auth_ctx.get("parish_id") or state.get("user_parish"),
        user_diocese=auth_ctx.get("diocese_id") or state.get("user_diocese"),
        auth_ctx=auth_ctx,
    )
    return {
        **state,
        "deterministic_reply": res["deterministic_markdown"],
        "generated_sql": res["generated_sql"],
        "sql_result": res["table_rows"],
        "chart_data": res["chart"],
        "analysis_metadata": res,
        "sql_validation_result": "PASSED_ALL_SECURITY_CHECKS",
        "authorized_record_count": res.get("authorized_record_count", 0),
        "suggested_questions": res.get("suggested_questions", []),
    }


def forecast_node(state: GraphState) -> GraphState:
    """
    LangGraph node for FORECAST and multi-series COMPARISON + FORECAST queries.
    Uses ONLY the user's authorized scope historical data (Section 61).
    """
    req_id = state.get("request_id", "N/A")
    intent_info = dict(state.get("intent_info") or {})
    auth_ctx = state.get("authorization_context") or {}
    intent_info["include_forecast"] = True
    if not intent_info.get("forecast_horizon"):
        intent_info["forecast_horizon"] = 10

    c_intent = state.get("classified_intent", "FORECAST")
    print(
        f"[LANGGRAPH] FORECAST NODE | request_id={req_id} | intent={c_intent} "
        f"| scope_type={auth_ctx.get('scope_type')} | scope_name={auth_ctx.get('scope_name')} "
        f"| horizon={intent_info.get('forecast_horizon')} yrs"
    )

    ensure_frappe_connected()
    from koinonia_assistant.rag.analytics_engine import run_analytics_or_forecast_pipeline

    res = run_analytics_or_forecast_pipeline(
        intent_info=intent_info,
        question=state.get("original_query") or state["question"],
        user_parish=auth_ctx.get("parish_id") or state.get("user_parish"),
        user_diocese=auth_ctx.get("diocese_id") or state.get("user_diocese"),
        auth_ctx=auth_ctx,
    )
    return {
        **state,
        "deterministic_reply": res["deterministic_markdown"],
        "generated_sql": res["generated_sql"],
        "sql_result": res["table_rows"],
        "chart_data": res["chart"],
        "analysis_metadata": res,
        "sql_validation_result": "PASSED_ALL_SECURITY_CHECKS",
        "authorized_record_count": res.get("authorized_record_count", 0),
        "suggested_questions": res.get("suggested_questions", []),
    }


def decide_route(state: GraphState):
    return state["route"]

def decide_db_lookup(state: GraphState):
    if state.get("route") == "fallback_sql":
        return "fallback_sql"
    return "handled"

def decide_validation(state: GraphState):
    if state.get("error_message") == "Unsupported query":
        return "unsupported"
    elif state.get("error_message"):
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "retry"
        else:
            return "failed"
    else:
        return "valid"

def decide_execution(state: GraphState):
    if state.get("error_message"):
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "retry"
        else:
            return "failed"
    else:
        return "success"

# ─── Assemble LangGraph Workflow (Section 58: Authorization BEFORE Router/SQL/Analytics) ───

workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("resolve_permissions", resolve_permissions_node)
workflow.add_node("router", router_node)
workflow.add_node("greeting", greeting_node)
workflow.add_node("unclear", unclear_node)
workflow.add_node("database_lookup_node", database_lookup_node)
workflow.add_node("analytics_node", analytics_node)
workflow.add_node("forecast_node", forecast_node)
workflow.add_node("enhance_query", enhance_query_node)
workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("validate_sql", validate_sql_node)
workflow.add_node("rewrite_sql", rewrite_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("format_response", format_response_node)

# Set Mandatory Entry Point: resolve_permissions (Section 58)
workflow.set_entry_point("resolve_permissions")

workflow.add_conditional_edges(
    "resolve_permissions",
    decide_permission,
    {
        "authorized": "router",
        "authorization_denied": "format_response",
    }
)

# Add Conditional Edges from Router
workflow.add_conditional_edges(
    "router",
    decide_route,
    {
        "greeting": "greeting",
        "unclear": "unclear",
        "database_lookup_node": "database_lookup_node",
        "analytics_node": "analytics_node",
        "forecast_node": "forecast_node",
        "text_to_sql": "enhance_query"
    }
)

workflow.add_conditional_edges(
    "database_lookup_node",
    decide_db_lookup,
    {
        "handled": "format_response",
        "fallback_sql": "enhance_query"
    }
)

workflow.add_edge("analytics_node", "format_response")
workflow.add_edge("forecast_node", "format_response")
workflow.add_edge("greeting", END)
workflow.add_edge("unclear", END)
workflow.add_edge("enhance_query", "retrieve_context")
workflow.add_edge("retrieve_context", "generate_sql")
workflow.add_edge("generate_sql", "validate_sql")

workflow.add_conditional_edges(
    "validate_sql",
    decide_validation,
    {
        "valid": "execute_sql",
        "retry": "rewrite_sql",
        "unsupported": "format_response",
        "failed": "format_response"
    }
)

workflow.add_edge("rewrite_sql", "validate_sql")

workflow.add_conditional_edges(
    "execute_sql",
    decide_execution,
    {
        "success": "format_response",
        "retry": "rewrite_sql",
        "failed": "format_response"
    }
)

workflow.add_edge("format_response", END)

# Compile Graph
app = workflow.compile()

# ─── Public Invocation Entrypoint ──────────────────────────────────────────────

def generate_follow_up_suggestions(question: str, sql_result: Any):
    is_ta = any('\u0B80' <= c <= '\u0BFF' for c in (question or ""))
    if is_ta:
        return [
            "கடந்த 10 ஆண்டுகளில் திருமுழுக்கு எண்ணிக்கை என்ன?",
            "முதல் நற்கருணை எண்ணிக்கையில் ஏற்பட்ட மாற்றத்தை காட்டு",
            "அடுத்த 10 ஆண்டுகளுக்கான கணிப்பை வழங்கவும்",
        ]
    suggestions = []
    if isinstance(sql_result, list) and len(sql_result) > 0:
        first_row = sql_result[0]
        name = first_row.get("Member Name") or first_row.get("Family Head") or first_row.get("Family Name / Head") or first_row.get("full_name")
        if name:
            clean_name = re.sub(r'\(.*?\)', '', str(name)).strip()
            suggestions.append(f"Show sacrament records for {clean_name}")
            suggestions.append(f"Show baptism records for {clean_name}")
            suggestions.append(f"Show family details for {clean_name}")

    if not suggestions:
        suggestions = [
            "What was the number of baptisms each year for the last 10 years?",
            "How has baptism changed over the last 10 years?",
            "Based on the last 10 years of baptism records, how might the next 10 years look?",
        ]

    return suggestions[:4]


def run_query(
    question: str,
    history: list = None,
    user_role: str = "Bishop",
    user_parish: str = None,
    user_diocese: str = None,
    request_id: str = None,
    user_id: str = None,
) -> dict:
    """
    Unified Server-First Entrypoint:
    - Preserves `original_query` EXACTLY as entered (Sections 24–27, 48: NEVER converts Tamil to Tanglish).
    - Executes `resolve_permissions` as Step 0 inside LangGraph BEFORE any database retrieval (Sections 50–76).
    """
    import uuid
    import time
    from koinonia_assistant.rag.tamil_utils import detect_query_language

    start_ts = time.time()
    req_id = request_id or str(uuid.uuid4())
    trace_id = f"lg-trace-{req_id[:12]}"
    original_question = (question or "").strip()
    detected_lang = detect_query_language(original_question)

    print("\n" + "=" * 76)
    print(f"[BACKEND] REQUEST RECEIVED | request_id={req_id} | trace_id={trace_id} | lang={detected_lang}")
    print(f"[BACKEND] Original Query (Immutable): '{original_question}' | Role: {user_role} | Parish: {user_parish}")
    print("=" * 76)

    # Embed original_question & log query history to Postgres (without overwriting original_question!)
    embedding = embed_text(original_question)
    history_id = log_query_history(original_question, "", embedding)

    # Pre-resolve intent & authorization metadata so LangSmith Root Input tab shows all search/query fields
    from koinonia_assistant.rag.analytics_engine import (
        classify_langgraph_intent,
        build_authorization_context,
        check_explicit_scope_violation,
    )
    pre_intent = classify_langgraph_intent(original_question)
    pre_auth = build_authorization_context(
        user_id=user_id or "User",
        user_role=user_role or "Parishioner",
        user_parish=user_parish,
        user_diocese=user_diocese,
    )
    pre_scope_check = check_explicit_scope_violation(original_question, pre_auth, detected_language=detected_lang)

    # Build initial LangGraph state with `question` and `original_query` strictly equal to `original_question`
    initial_state: GraphState = {
        "question": original_question,
        "original_query": original_question,
        "detected_language": detected_lang,
        "normalized_query": pre_intent.get("normalized_query", ""),
        "intent_query": pre_intent.get("intent_query", pre_intent.get("intent", "")),
        "entity_query": pre_intent.get("entity_query", pre_intent.get("person_name") or ""),
        "canonical_terms": pre_intent.get("canonical_terms", {}),
        "final_response_language": "ta" if detected_lang == "ta" else "en",
        "history": history or [],
        "route": "",
        "enhanced_query": pre_intent.get("normalized_query", ""),
        "relevant_tables": ["tabMember", "tabFamily", "tabBaptism", "tabCommunion", "tabConfirmation", "tabMarriage"],
        "relevant_fields": [],
        "few_shot_examples": "",
        "query_embedding": embedding,
        "generated_sql": "",
        "llm_explanation": "",
        "sql_result": None,
        "error_message": "",
        "retry_count": 0,
        "final_answer": "",
        "history_id": history_id,
        "user_id": user_id or "User",
        "user_role": user_role,
        "user_parish": user_parish,
        "user_diocese": user_diocese,
        "authorization_context": pre_auth,
        "authorization_check": "PRE_RETRIEVAL_SCOPE_ENFORCEMENT",
        "authorization_result": pre_scope_check.get("authorization_result", "AUTHORIZED"),
        "sql_validation_result": "PENDING_VALIDATION",
        "authorized_record_count": 0,
        "request_id": req_id,
        "classified_intent": pre_intent.get("intent", ""),
        "speed_tier": pre_intent.get("speed_tier", "FAST"),
        "intent_info": pre_intent,
        "deterministic_reply": None,
        "chart_data": None,
        "disambiguation": None,
        "suggested_questions": None,
        "analysis_metadata": None,
    }

    # Guarantee LangSmith trace capture with original_query & authorization metadata
    configure_langsmith()
    callbacks = []
    try:
        from langchain_core.tracers import LangChainTracer
        tracer = LangChainTracer(project_name=os.environ.get("LANGCHAIN_PROJECT", "koinonia_assistant"))
        callbacks.append(tracer)
    except Exception as te:
        print(f"[LangSmith] Tracer init warning: {te}")

    config = {
        "callbacks": callbacks,
        "run_name": "LangGraph",
        "metadata": {
            "request_id": req_id,
            "trace_id": trace_id,
            "original_query": original_question,
            "detected_language": detected_lang,
            "user_id": user_id or "User",
            "user_role": user_role,
            "user_parish": user_parish or "Diocesan Scope",
        },
    } if callbacks else {"run_name": "LangGraph"}

    # Execute LangGraph Pipeline (starts at resolve_permissions -> router -> ...)
    final_state = app.invoke(initial_state, config=config)

    processing_time = round(time.time() - start_ts, 3)
    final_reply = final_state.get("final_answer", "Error resolving request.")
    raw_sql_res = final_state.get("sql_result") or []
    gen_sql = final_state.get("generated_sql", "")
    c_intent = final_state.get("classified_intent") or "GENERAL_DATABASE_QUERY"
    s_tier = final_state.get("speed_tier") or "FAST"
    chart_payload = final_state.get("chart_data")
    disambig = final_state.get("disambiguation")
    suggested_qs = final_state.get("suggested_questions") or generate_follow_up_suggestions(original_question, raw_sql_res)

    # Strip internal backend database ID columns from UI display rows (keep IDs backend-only)
    BACKEND_ONLY_ID_COLS = {
        "member_id", "Member ID", "member id",
        "family_id", "Family ID", "family id",
        "name", "id", "ID", "is_family_head", "parish_bcc_id"
    }
    sql_res = []
    if isinstance(raw_sql_res, list):
        for row in raw_sql_res:
            if isinstance(row, dict):
                cleaned_row = {
                    k: v for k, v in row.items()
                    if k not in BACKEND_ONLY_ID_COLS and not str(k).lower().endswith("_id")
                }
                if cleaned_row:
                    sql_res.append(cleaned_row)
            else:
                sql_res.append(row)
    meta = final_state.get("analysis_metadata") or {}
    auth_ctx_out = final_state.get("authorization_context") or {}

    hist_period = None
    forecast_period = None
    forecast_method = None
    if meta and meta.get("datasets"):
        first_ds = next(iter(meta["datasets"].values()), {})
        hist_period = f"{first_ds.get('start_year')}-{first_ds.get('end_year')}"
        f_obj = first_ds.get("forecast") or {}
        forecast_period = f_obj.get("forecast_period")
        forecast_method = f_obj.get("method")

    print("=" * 76)
    print(
        f"[BACKEND] RESPONSE READY | request_id={req_id} | trace_id={trace_id} "
        f"| lang={final_state.get('detected_language')} | intent={c_intent} "
        f"| scope={auth_ctx_out.get('scope_name')} | auth_res={final_state.get('authorization_result')} "
        f"| time={processing_time}s"
    )
    print("=" * 76)

    return {
        "request_id": req_id,
        "trace_id": trace_id,
        "original_query": final_state.get("original_query", original_question),
        "detected_language": final_state.get("detected_language", detected_lang),
        "normalized_query": final_state.get("normalized_query", ""),
        "canonical_terms": final_state.get("canonical_terms", {}),
        "final_response_language": final_state.get("final_response_language", "ta" if detected_lang == "ta" else "en"),
        "intent": c_intent,
        "speed_tier": s_tier,
        "status": "success",
        "authorization_context": auth_ctx_out,
        "authorization_check": final_state.get("authorization_check", "PRE_RETRIEVAL_SCOPE_ENFORCEMENT"),
        "authorization_result": final_state.get("authorization_result", "AUTHORIZED"),
        "sql_validation_result": final_state.get("sql_validation_result", "PASSED_ALL_SECURITY_CHECKS"),
        "authorized_record_count": final_state.get("authorized_record_count", 0),
        "scope_type": auth_ctx_out.get("scope_type"),
        "scope_id": auth_ctx_out.get("scope_id"),
        "scope_name": auth_ctx_out.get("scope_name"),
        "reply": final_reply,
        "answer": final_reply,
        "generated_sql": gen_sql,
        "data": sql_res,
        "chart": chart_payload,
        "disambiguation": disambig,
        "suggested_questions": suggested_qs,
        "analysis_type": "forecast" if c_intent == "FORECAST" else ("historical" if c_intent in ("HISTORICAL_ANALYSIS", "STATISTICAL_ANALYSIS", "TREND_ANALYSIS", "COMPARISON") else "operational"),
        "historical_period": hist_period,
        "forecast_period": forecast_period,
        "method": forecast_method,
        "processing_time": processing_time,
        "query_id": history_id,
    }

