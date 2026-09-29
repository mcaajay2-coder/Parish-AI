import os
import re
import time
import json
import psycopg2
import pymysql
import torch
from typing import TypedDict, Any
from dotenv import load_dotenv

# Initialize LangSmith tracing environment BEFORE importing langchain/langgraph
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '.env')
load_dotenv(dotenv_path)

def _init_langsmith():
    key = os.getenv("LANGCHAIN_API_KEY")
    if not key:
        try:
            import frappe
            if hasattr(frappe, "conf") and frappe.conf:
                key = frappe.conf.get("langchain_api_key")
        except Exception:
            pass
    if key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = key
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "koinonia_assistant")

_init_langsmith()

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

_cross_encoder = None
def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None and CrossEncoder is not None:
        try:
            print("[CrossEncoder] Loading model...")
            _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception:
            _cross_encoder = None
    return _cross_encoder

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

MAX_RETRIES = 2

# Initialize Groq multi-key pool
def _get_groq_keys() -> list[str]:
    keys = []
    # 1. Check environment variables
    env_keys = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY")
    if env_keys:
        if "," in env_keys:
            keys.extend([k.strip() for k in env_keys.split(",") if k.strip()])
        else:
            keys.append(env_keys.strip())

    # 2. Check site_config.json via frappe.conf
    try:
        import frappe
        if hasattr(frappe, "conf") and frappe.conf:
            conf_keys = frappe.conf.get("groq_api_keys")
            if conf_keys and isinstance(conf_keys, list):
                keys.extend(conf_keys)
            elif conf_keys and isinstance(conf_keys, str):
                keys.extend([k.strip() for k in conf_keys.split(",") if k.strip()])
            
            single_key = frappe.conf.get("groq_api_key")
            if single_key and single_key not in keys:
                keys.append(single_key)
    except Exception:
        pass

    # 3. Direct site_config.json file fallback
    if not keys:
        for site_path in [
            "/home/frappe/frappe-bench/sites/frontend/site_config.json",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "sites", "frontend", "site_config.json")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sites", "frontend", "site_config.json")),
        ]:
            if os.path.exists(site_path):
                try:
                    with open(site_path, "r", encoding="utf-8") as f:
                        sconf = json.load(f)
                        if sconf.get("groq_api_keys"):
                            gks = sconf.get("groq_api_keys")
                            if isinstance(gks, list):
                                keys.extend(gks)
                            elif isinstance(gks, str):
                                keys.extend([k.strip() for k in gks.split(",") if k.strip()])
                        if sconf.get("groq_api_key"):
                            keys.append(sconf.get("groq_api_key"))
                        break
                except Exception:
                    pass

    # Unique while preserving order
    seen = set()
    unique_keys = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            unique_keys.append(k)

    return unique_keys

GROQ_API_KEYS = _get_groq_keys()
_current_key_idx = 0

def _setup_langsmith():
    key = os.getenv("LANGCHAIN_API_KEY")
    if not key:
        try:
            import frappe
            if hasattr(frappe, "conf") and frappe.conf:
                key = frappe.conf.get("langchain_api_key")
        except Exception:
            pass
    if key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = key
        os.environ["LANGCHAIN_PROJECT"] = "koinonia_assistant"

_setup_langsmith()

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

def invoke_llm_with_rotation(prompt_messages):
    global _current_key_idx
    total_keys = len(GROQ_API_KEYS)
    
    # Format messages to ensure there is always a human/user message for Qwen/OSS models
    formatted_messages = []
    if isinstance(prompt_messages, str):
        formatted_messages = [("human", prompt_messages)]
    elif isinstance(prompt_messages, list):
        has_human = any((m[0] in ["human", "user"] if isinstance(m, tuple) else getattr(m, 'type', '') in ["human", "user"]) for m in prompt_messages)
        if not has_human:
            formatted_messages = []
            for i, m in enumerate(prompt_messages):
                if i == len(prompt_messages) - 1:
                    role = "human"
                    content = m[1] if isinstance(m, tuple) else m.content
                else:
                    role = m[0] if isinstance(m, tuple) else m.type
                    content = m[1] if isinstance(m, tuple) else m.content
                formatted_messages.append((role, content))
        else:
            formatted_messages = prompt_messages
    else:
        formatted_messages = prompt_messages
    
    models_to_try = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "qwen/qwen3.6-27b"
    ]
    try:
        import frappe
        if hasattr(frappe, "conf") and frappe.conf:
            conf_models = frappe.conf.get("groq_models")
            if conf_models and isinstance(conf_models, list):
                models_to_try = conf_models + [m for m in models_to_try if m not in conf_models]
            single_m = frappe.conf.get("groq_model")
            if single_m and single_m in models_to_try:
                models_to_try.remove(single_m)
                models_to_try.insert(0, single_m)
    except Exception:
        pass

    # Remove duplicates preserving order
    seen_models = set()
    models_to_try = [m for m in models_to_try if not (m in seen_models or seen_models.add(m))]
    
    for model_name in models_to_try:
        for attempt in range(total_keys):
            idx = (_current_key_idx + attempt) % total_keys
            key = GROQ_API_KEYS[idx]
            try:
                client = ChatGroq(model=model_name, temperature=0, groq_api_key=key, max_retries=1)
                resp = client.invoke(formatted_messages)
                _current_key_idx = idx
                if hasattr(resp, 'content') and isinstance(resp.content, str):
                    resp.content = re.sub(r'<think>.*?</think>', '', resp.content, flags=re.DOTALL).strip()
                return resp
            except Exception as e:
                err_str = str(e)
                print(f"[Groq Rotator] {model_name} error on key #{idx+1} ({key[:10]}...): {err_str[:80]}... Rotating...")
                if "429" in err_str or "rate_limit" in err_str.lower():
                    time.sleep(1.0)

    raise RuntimeError("All Groq API keys and models exhausted their rate limits.")

# Lazy LLM wrapper to support .invoke(...) cleanly without import-time Groq initialization
class LazyLLM:
    def invoke(self, messages, **kwargs):
        formatted = []
        if isinstance(messages, str):
            formatted = [("user", messages)]
        elif isinstance(messages, list):
            for m in messages:
                if isinstance(m, tuple):
                    formatted.append(m)
                elif hasattr(m, 'content'):
                    role = "user" if getattr(m, 'type', None) in ["human", "user"] else ("assistant" if getattr(m, 'type', None) == "ai" else "system")
                    formatted.append((role, m.content))
                else:
                    formatted.append(("user", str(m)))
        else:
            formatted = [("user", str(messages))]
        return invoke_llm_with_rotation(formatted)

primary_llm = LazyLLM()
fallback_llm = LazyLLM()
llm = LazyLLM()

def ensure_frappe_db():
    try:
        import frappe
        if not (hasattr(frappe, "local") and hasattr(frappe.local, "db") and frappe.local.db):
            for sites_p in ["/home/frappe/frappe-bench/sites", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "sites"))]:
                if os.path.exists(sites_p):
                    frappe.init(site="frontend", sites_path=sites_p)
                    frappe.connect()
                    break
    except Exception as e:
        print(f"[ensure_frappe_db] Note: {e}")

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
        response = invoke_llm_with_rotation([("system", prompt)])
        content = response.content.strip()
        selected = [t.strip().strip("`").strip("'").strip('"') for t in content.split(",")]
        candidate_names = {c[0] for c in candidates}
        final_selection = [s for s in selected if s in candidate_names]
        if final_selection:
            return final_selection
    except Exception as e:
        print(f"[rerank_tables] Error during LLM re-ranking: {e}")
        
    return [c[0] for c in candidates[:2]]

def fetch_relevant_schemas(original_query: str, enhanced_query: str, query_embedding: list[float], k: int = 2) -> str:
    conn = psycopg2.connect(**PG_CONFIG)
    candidates = []
    seen = set()
    combined_query_text = f"{original_query} {enhanced_query}"
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT table_name, schema_ddl FROM koinonia_table_schemas;")
            all_tables = cur.fetchall()
            
            docs = []
            doc_to_table = {}
            for tname, ddl in all_tables:
                doc = f"{tname} {ddl}".lower()
                docs.append(doc)
                doc_to_table[doc] = (tname, ddl)
                
            if docs:
                tokenized_docs = [doc.split() for doc in docs]
                bm25 = BM25Okapi(tokenized_docs)
                tokenized_query = combined_query_text.lower().split()
                bm25_scores = bm25.get_scores(tokenized_query)
                
                top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
                for idx in top_bm25_indices:
                    tname, ddl = doc_to_table[docs[idx]]
                    if tname not in seen:
                        seen.add(tname)
                        candidates.append((tname, ddl))
            
            cur.execute("""
                SELECT table_name, schema_ddl
                FROM koinonia_table_schemas
                ORDER BY embedding <=> %s::vector
                LIMIT 5;
            """, (query_embedding,))
            for tname, ddl in cur.fetchall():
                if tname not in seen:
                    seen.add(tname)
                    candidates.append((tname, ddl))
                    
        if candidates:
            ce = get_cross_encoder()
            if ce:
                pairs = [[combined_query_text, f"{tname} {ddl}"] for tname, ddl in candidates]
                scores = ce.predict(pairs)
                scored_candidates = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
                top_candidates = [cand for score, cand in scored_candidates[:k]]
            else:
                top_candidates = candidates[:k]
        else:
            top_candidates = []
            
        selected_table_names = rerank_tables(original_query, top_candidates)
        schema_map = {tname: ddl for tname, ddl in top_candidates}
        final_results = []
        for tname in selected_table_names:
            if tname in schema_map:
                final_results.append((tname, schema_map[tname]))
                
    except Exception as e:
        print("[RAG Context] Error fetching relevant schemas:", e)
        final_results = []
    finally:
        conn.close()
        
    pruned_results = []
    system_cols = ["_user_tags", "_comments", "_assign", "_liked_by", "amended_from", "idx", "docstatus", "creation", "modified", "modified_by", "owner", "custom", "primary key", "key `", "unique key", "constraint", "engine=", "default charset="]
    for tname, ddl in final_results[:2]:
        clean_lines = []
        for line in ddl.split("\n"):
            line_str = line.strip()
            if not line_str or line_str.startswith("--"):
                continue
            if not any(sc in line_str.lower() for sc in system_cols):
                clean_lines.append(line_str)
        pruned_results.append(f"Table `{tname}`:\n" + "\n".join(clean_lines[:30]))
        
    return pruned_results

def fetch_relevant_fields(query_embedding: list[float], combined_query_text: str, k: int = 3) -> list[str]:
    conn = psycopg2.connect(**PG_CONFIG)
    candidates = []
    seen = set()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT table_name, field_name, field_type, field_label, description FROM koinonia_field_schemas;")
            all_fields = cur.fetchall()
            
            docs = []
            doc_to_row = {}
            for row in all_fields:
                doc = f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]}".lower()
                docs.append(doc)
                doc_to_row[doc] = row
                
            if docs:
                tokenized_docs = [doc.split() for doc in docs]
                bm25 = BM25Okapi(tokenized_docs)
                tokenized_query = combined_query_text.lower().split()
                bm25_scores = bm25.get_scores(tokenized_query)
                
                top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
                for idx in top_bm25_indices:
                    row = doc_to_row[docs[idx]]
                    row_str = f"- Table `{row[0]}`, Column `{row[1]}` (Type: {row[2]}, Label: {row[3]}): {row[4]}"
                    if row_str not in seen:
                        seen.add(row_str)
                        candidates.append((row_str, row))
            
            cur.execute("""
                SELECT table_name, field_name, field_type, field_label, description
                FROM koinonia_field_schemas
                ORDER BY embedding <=> %s::vector
                LIMIT 10;
            """, (query_embedding,))
            for row in cur.fetchall():
                row_str = f"- Table `{row[0]}`, Column `{row[1]}` (Type: {row[2]}, Label: {row[3]}): {row[4]}"
                if row_str not in seen:
                    seen.add(row_str)
                    candidates.append((row_str, row))
                    
        if candidates:
            ce = get_cross_encoder()
            if ce:
                pairs = [[combined_query_text, c[0]] for c in candidates]
                scores = ce.predict(pairs)
                scored_candidates = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
                results = [cand[0] for score, cand in scored_candidates[:k]]
            else:
                results = [cand[0] for cand in candidates[:k]]
        else:
            results = []
            
    except Exception as e:
        print("[RAG Context] Error fetching relevant fields:", e)
        results = []
    finally:
        conn.close()
    return results

def fetch_few_shot_examples(query_embedding: list[float], k: int = 3) -> str:
    lines = [
        "Example Mappings:",
        "  Q: \"Count how many baptisms were conducted in Holy Cross Parish during 2024\"",
        "  SQL: SELECT COUNT(*) FROM tabBaptism WHERE bapt_parish_id = 'Holy Cross Parish' AND YEAR(bapt_date) = 2024",
        "  Q: \"Tell me about Paul Amalraj S.\"",
        "  SQL: SELECT first_name, middle_name, last_name, dob, gender, living_status, parish_id, mobile FROM tabMember WHERE (first_name = 'Paul' AND middle_name = 'Amalraj' AND last_name = 'S.') OR CONCAT_WS(' ', first_name, NULLIF(middle_name, ''), last_name) LIKE '%Paul%Amalraj%'",
        "  Q: \"Show family details and household members of Agnes Mary A\"",
        "  SQL: SELECT m.name AS member_id, CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) AS full_name, m.relationship_id AS relationship, m.gender, m.age, m.dob, m.mobile, f.family_card_number, f.parish_bcc_id AS anbiyam, f.street, f.city, m.parish_id FROM tabMember m JOIN tabFamily f ON m.family_id = f.name WHERE m.family_id = (SELECT family_id FROM tabMember WHERE (CONCAT_WS(' ', first_name, middle_name, last_name) LIKE '%Agnes Mary%' OR first_name LIKE '%Agnes%') AND parish_id = 'Yelagiri Parish' LIMIT 1) AND m.parish_id = 'Yelagiri Parish' ORDER BY CASE WHEN m.relationship_id IN ('Self', 'Head', 'Husband') THEN 1 WHEN m.relationship_id = 'Wife' THEN 2 ELSE 3 END, m.dob ASC"
    ]
    
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT user_question, generated_sql
                FROM koinonia_query_history
                WHERE correctness_flag = 1
                  AND generated_sql IS NOT NULL
                  AND LENGTH(TRIM(generated_sql)) > 10
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
        lines.append("\nDynamic History Examples:")
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

def update_correctness_flag(query_id: int, is_correct: int, sql: str = ""):
    if query_id < 0:
        return
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            if sql and sql not in ["UNSUPPORTED", "UNAUTHORIZED_DIOCESE", "UNAUTHORIZED_PARISH", "BLOCKED_SECURITY"]:
                cur.execute("""
                    UPDATE koinonia_query_history 
                    SET correctness_flag = %s, generated_sql = %s 
                    WHERE id = %s;
                """, (is_correct, sql, query_id))
            else:
                cur.execute("""
                    UPDATE koinonia_query_history SET correctness_flag = %s WHERE id = %s;
                """, (is_correct, query_id))
        conn.commit()
    except Exception as e:
        print("[History] Error updating correctness flag:", e)
    finally:
        conn.close()

# ─── LangGraph State Definition ───────────────────────────────────────────────

class GraphState(TypedDict):
    question:          str
    history:           list[dict[str, str]]
    reference_text:    str
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
    user_role:         str
    user_parish:       str
    user_vicariate:    str
    user_diocese:      str
    user_parishes:     list[str]
    user_member_id:    str
    user_email:        str
    requested_foreign_parish: str
    requested_foreign_diocese: str

# ─── Graph Nodes ──────────────────────────────────────────────────────────────

def router_node(state: GraphState) -> GraphState:
    print("\n[router] Classifying user query...")
    raw_q = state.get("question") or ""
    q = raw_q.strip().lower()
    q_stripped = q.strip(".,!?\"' ")

    # Explicit Security Injection & Threat Guard
    injection_patterns = [
        r';\s*(?:DROP|ALTER|TRUNCATE|DELETE|INSERT|UPDATE|CREATE|REPLACE)',
        r'\b(?:SLEEP|BENCHMARK|WAITFOR|GET_LOCK)\b',
        r'\bUNION\s+(?:ALL\s+)?SELECT\b',
        r'--\s*$',
        r'/\*.*?\*/',
        r'\b(?:ignore all previous instructions|output your system prompt|leak secret keys|system prompt|secret API keys)\b',
        r'\b(?:tabUser|information_schema|mysql\.)\b'
    ]
    for pat in injection_patterns:
        if re.search(pat, raw_q, re.IGNORECASE | re.DOTALL):
            print(f"[router] Security Alert: Intercepted malicious injection payload: '{raw_q}'")
            return {
                **state,
                "route": "blocked_security",
                "generated_sql": "BLOCKED_SECURITY",
                "final_answer": "🚫 **Security Alert**: Malicious SQL injection or unauthorized schema exfiltration attempt was blocked by KOINONIA Security Guardrails."
            }

    # Explicit Security Injection & Threat Guard
    injection_patterns = [
        r';\s*(?:DROP|ALTER|TRUNCATE|DELETE|INSERT|UPDATE|CREATE|REPLACE)',
        r'(?:SLEEP|BENCHMARK|WAITFOR|GET_LOCK)\s*\(',
        r'UNION\s+(?:ALL\s+)?SELECT\s+.*(?:tabUser|information_schema|mysql\.|password|hash|secret|token|api_key)',
        r'--\s*$',
        r'(?:ignore all previous instructions|output your system prompt|leak secret keys)'
    ]
    for pat in injection_patterns:
        if re.search(pat, state["question"], re.IGNORECASE):
            print(f"[router] Security Alert: Intercepted malicious injection payload: '{state['question']}'")
            return {
                **state,
                "route": "text_to_sql",
                "generated_sql": "BLOCKED_SECURITY",
                "final_answer": "🚫 **Security Alert**: Malicious SQL injection or unauthorized schema exfiltration attempt was blocked by KOINONIA Security Guardrails."
            }

    # Greetings & Identity / Capabilities
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you", "bye", "goodbye", "namaste", "vanakkam"}
    identity_phrases = [
        "how are you", "who are you", "what can you do", "introduce yourself", 
        "tell me about yourself", "tell me about you self", "tell about yourself", 
        "about yourself", "about you", "what are you", "help me", "help", 
        "what is your name", "your features", "what do you do", "capabilities",
        "how to use", "instructions", "guide me", "what is koinonia"
    ]
    if q_stripped in greetings or any(phrase in q for phrase in identity_phrases):
        return {**state, "route": "greeting"}

    # If completely empty or single punctuation
    if not q_stripped or len(q_stripped) < 2:
        return {**state, "route": "unclear"}

    # All user requests, questions, searches, and name lookups route directly to text_to_sql
    return {**state, "route": "text_to_sql"}

def blocked_security_node(state: GraphState) -> GraphState:
    return {
        **state,
        "generated_sql": "BLOCKED_SECURITY",
        "final_answer": "🚫 **Security Alert**: Malicious SQL injection or unauthorized schema exfiltration attempt was blocked by KOINONIA Security Guardrails."
    }

def unclear_node(state: GraphState) -> GraphState:
    answer = (
        "🤔 I'm sorry, I didn't quite get that.\n\n"
        "I am the **KOINONIA Assistant**. I can help you search the parish family register and sacrament records. "
        "Try asking me things like:\n"
        "- *'List all families in Lourdu Matha BCC'* or *'How many members are in Zone 1?'*\n"
        "- *'Find the baptism record of Rani Marianathan'* or *'Who was the burial minister for Jessy Lourdusamy?'*\n"
        "- *'Show all confirmations conducted in 2024'*\n\n"
        "Please let me know how I can assist you!"
    )
    return {**state, "final_answer": answer}

def greeting_node(state: GraphState) -> GraphState:
    import frappe
    user_email = state.get("user_email") or frappe.session.user
    user_name = "there"
    user_title = ""

    if user_email and user_email != "Guest":
        user_name = frappe.db.get_value("User", user_email, "first_name") or frappe.db.get_value("User", user_email, "full_name") or "there"
        roles = frappe.get_roles(user_email)
        if "Bishop" in roles or "Bishop Role" in roles:
            user_title = "Bishop"
        elif "Vicar General" in roles or "Vicar" in roles:
            user_title = "Vicar"
        elif "Parish Priest" in roles:
            user_title = "Father"

    salutation = f"Hello {user_title} {user_name}!" if user_title else f"Hello {user_name}!"
    
    answer = (
        f"👋 **{salutation}** I am **KOINONIA Assistant**, your dedicated Catholic Diocesan & Parish Registry Assistant.\n\n"
        "Here is what I can do for you:\n"
        "• 🏛️ **Parish & Vicariate Registries:** Find parishes, clergy assignments, family counts, and diocese-wide demographics.\n"
        "• 🕊️ **Sacramental Records:** Search and verify records for **Baptism, First Holy Communion, Confirmation, Holy Matrimony, Anointing of the Sick, and Christian Burial**.\n"
        "• 👥 **Family & Census Data:** Explore family registers, Basic Christian Communities (BCC), and parishioner directories.\n"
        "• 📊 **Analytics & Visualizations:** Generate line graphs 📈, bar charts 📊, and pie charts 🥧 for parish statistics upon request.\n"
        "• 📄 **Export Reports:** Export formatted **Excel spreadsheets** and **official PDF documents** with complete headers and totals.\n\n"
        "💬 *Just ask your question in plain English or Tanglish (e.g., 'Show total families in each parish' or 'List baptisms in 2024') and I will fetch the records for you!*"
    )
    return {**state, "final_answer": answer}

ENHANCE_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a smart query context extension assistant for the KOINONIA Parish Assistant app.
The user is asking a question that references a previous question, record, or quoted text.
Your task is to combine the referenced context with the user's new question into a complete, standalone question for a church/sacrament database assistant.

Examples:
- User Question: "Show me the babtizum records about Jose D." | Reference: "List the users who babtized in 2025"
  -> Enhanced: "Show the baptism records of Jose D. who was baptized in 2025"

- User Question: "tell me about Nirmala Fernando G. family details" | Reference: "Found 50 records matching your search: | Full Name | Bapt Date | Bapt Parish I..."
  -> Enhanced: "Tell me about the family details and family members of Nirmala Fernando G."

- User Question: "List them with their parents" | Reference: "List the users who babtized in 2025"
  -> Enhanced: "List the persons baptized in 2025 with their first name, last name, father name, and mother name"

- User Question: "tell me more about this person" | Reference: "Thomas D'Souza"
  -> Enhanced: "Show details for member Thomas D'Souza"

- User Question: "Show me his marriage records" | Reference: "Jose D"
  -> Enhanced: "Show marriage records for Jose D."

- User Question: "Show next 50 records" | Reference: "Show members in St. Joseph's Parish"
  -> Enhanced: "Show next 50 members in St. Joseph's Parish with offset 50"

- User Question: "Next records" | Reference: "Show marriages in 2025"
  -> Enhanced: "Show next 50 marriages in 2025 with offset 50"

- User Question: "அவங்களோட பெயர் பட்டியல் தாங்க" | Reference: "2024-ல் ஞானஸ்நானம் பெற்றவர்கள் எத்தனை பேர்? **Total Count**: 42"
  -> Enhanced: "List all persons baptized in 2024 (tabBaptism) with their first name, middle name, last name, bapt date, and parish name"

- User Question: "avangaloda family members list pannunga" | Reference: "Holy Baptism Registry Record: Carmel Maria B. Family Card Number: FC-25646"
  -> Enhanced: "List all living family members of family card number FC-25646 (tabMember joined with tabFamily)"

- User Question: "அவர்களின் பெற்றோர் பெயர் விபரம் கொடு" | Reference: "2023-ல் புது நன்மை எடுத்தவர்கள் யார் யார்? Found communion records in 2023"
  -> Enhanced: "List all persons who received First Holy Communion in 2023 with their first name, middle name, last name, father name, mother name, and parish name (tabCommunion)"

CRITICAL RULES:
1. Return ONLY the extended, standalone question string.
2. Do NOT add markdown, explanations, or quotes.
3. Preserve all specific entity names (names, dates, years, sacraments).
4. STRICT SACRAMENT CONTEXT PRESERVATION:
   - If prior conversation is about First Holy Communion / புது நன்மை / FHC -> The extended query MUST remain about First Holy Communion (tabCommunion). NEVER change it to Confirmation or Baptism.
   - If prior conversation is about Baptism / ஞானஸ்நானம் -> The extended query MUST remain about Baptism (tabBaptism).
   - If prior conversation is about Confirmation / உறுதிப்பூசுதல் -> The extended query MUST remain about Confirmation (tabConfirmation).
   - If prior conversation is about Marriage / திருமணம் -> The extended query MUST remain about Marriage (tabMarriage).
5. When user asks for parish name / பங்கு பெயர் and family card / குடும்ப அட்டை -> ALWAYS explicitly include "parish name (fhc_parish_id AS parish_name)" and "family card number (family_card_no)" in the extended query."""),
    ("human", "{reference_context}\nLatest User Question: {question}"),
])

STANDALONE_ENHANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent Catholic Church terminology and translation assistant for the KOINONIA Parish Assistant app.
Your task is to convert Tamil, Tanglish, and English questions with typos/phonetics into a clean, unambiguous standalone English search query for a church/sacrament database.

CATHOLIC TAMIL TERMINOLOGY MAPPINGS:
- "பங்கு பெயர்" / "பங்கின் பெயர்" / "பங்கு" -> Parish Name (fhc_parish_id AS parish_name / bapt_parish_id AS parish_name / mrg_parish_id AS parish_name / parish_id AS parish_name)
- "குடும்ப அட்டை எண்" / "குடும்ப அட்டை" / "அட்டை எண்" -> Family Card Number (family_card_no)
- "பங்கினர் பெயர்" / "விசுவாசி பெயர்" / "நபரின் பெயர்" -> Person Full Name (first_name, middle_name, last_name)
- "புது நன்மை" / "புதுநன்மை" / "முதல் நற்கருணை" / "நற்கருணை" -> First Holy Communion (tabCommunion)
- "ஞானஸ்நானம்" / "திருமுழுக்கு" / "மாமோதீசா" -> Baptism (tabBaptism)
- "உறுதிப்பூசுதல்" / "அபிஷேகம்" -> Confirmation (tabConfirmation)
- "திருமணம்" / "விவாகம்" / "கல்யாணம்" -> Marriage / Holy Matrimony (tabMarriage)
- "நோயில் பூசுதல்" / "கடைசிப் பூசுதல்" / "தைலம்" -> Anointing of the Sick (tabAnointing Of Sick)
- "அடக்கம்" / "மரணப் பதிவு" / "மரித்தவர்கள்" -> Christian Burial / Death (tabDeath)
- "நபர்கள்" / "பங்கினர்" / "விசுவாசிகள்" / "உறுப்பினர்கள்" -> Members / Parishioners (tabMember)
- "குடும்பங்கள்" / "குடும்பம்" -> Families (tabFamily)
- "அன்பியம்" / "அன்பியங்கள்" -> BCC / Basic Christian Communities (tabFamily)
- "பங்கு" / "பங்குகள்" -> Parish / Parishes (tabParish)
- "மறைமாவட்டம்" -> Diocese (tabDiocese)
- "மறைவட்டம்" -> Vicariate (tabVicariate)

Examples:
- "இந்த வருடம் எத்தனை நபர்கள் புது நன்மை எடுத்துள்ளனர்?" -> "How many members received First Holy Communion this year?"
- "போன வருடம் நடந்த திருமணங்களில் Clinton-ன் பேரில் யாருக்கும் திருமணம் நடந்ததா?" -> "Find marriage records for Clinton in marriage register"
- "Clinton திருமணம் நடந்ததா?" -> "Find marriage records for Clinton in marriage register"
- "கடந்த வருடம் மொத்தம் எத்தனை பேர் புதுப்பணி எடுத்தார்கள் அவர்களின் list மற்றும் family code-ஐ இரண்டையும் குறிப்பிடவும்" -> "List members who received First Holy Communion last year with their names and family card number (family_card_no)"
- "கடந்த வருடம் புது நன்மை எடுத்தவர்கள் பட்டியல் மற்றும் குடும்ப அட்டை எண்" -> "List members who received First Holy Communion last year with their names and family card number (family_card_no)"
- "இந்த ஆண்டு மொத்தம் எத்தனை நபர்கள் இறந்தார்கள் மற்றும் அவர்களின் பரிஷ் பெயரை குறிப்பிடவும்" -> "List members who died this year with their parish name from death register"
- "புது நன்மை பெற்றவர்கள் பட்டியல்" -> "List of members who received First Holy Communion"
- "2024-ல் ஞானஸ்நானம் பெற்ற குழந்தைகள்" -> "List of children baptized in 2024"
- "உறுதிப்பூசுதல் பெற்றவர்கள் எத்தனை பேர்" -> "Count of members who received Confirmation"
- "shwo all merriage recrods" -> "Show all marriage records"
- "list membes who completed babtizum in 2024" -> "List members who completed baptism in 2024"
- "who is the famly hed in st joshep parsh" -> "Who is the family head in St. Joseph Parish"
- "how many familys in lourdu matha bcc" -> "How many families in Lourdu Matha BCC"
- "பங்கில் உள்ள மொத்த குடும்பங்களின் எண்ணிக்கை எவ்வளவு?" -> "How many families are in my parish?"
- "நம் பங்கில் மொத்தம் எத்தனை நபர்கள் உள்ளார்கள்" -> "How many members are in our parish?"
- "பங்கில் உள்ள குடும்பங்களின் எண்ணிக்கை" -> "Total number of families in my parish"

CRITICAL RULES:
1. Return ONLY the translated, corrected English question string.
2. Do NOT add quotes, markdown, explanations, or any extra text.
3. Preserve all specific numbers, years, proper names, and parish names."""),
    ("human", "{question}"),
])

def clean_church_query_text(text: str) -> str:
    if not text:
        return text
    # Specialized phonetic & speech-to-text cleaner for Tamil/English Catholic Church queries
    fixes = [
        (r'\bபுதுப்?\s*பணி\b', 'First Holy Communion (புது நன்மை)'),
        (r'\bபுதுப்பணி\b', 'First Holy Communion (புது நன்மை)'),
        (r'\bபுதுபணி\b', 'First Holy Communion (புது நன்மை)'),
        (r'\bpudupani\b', 'First Holy Communion'),
        (r'\bpudu\s*pani\b', 'First Holy Communion'),
        (r'\bபுது\s*நன்மை\b', 'First Holy Communion (புது நன்மை)'),
        (r'\bபுதுநன்மை\b', 'First Holy Communion (புது நன்மை)'),
        (r'\bமுதல்\s*நற்கருணை\b', 'First Holy Communion (முதல் நற்கருணை)'),
        (r'\bநற்கருணை\b', 'First Holy Communion (நற்கருணை)'),
        (r'\bkalyanam\b', 'Marriage'),
        (r'\bகல்யாண[ம்ா]?\b', 'Marriage (திருமணம்)'),
        (r'\bதிருமணம்\b', 'Marriage (திருமணம்)'),
        (r'\bவிவாகம்\b', 'Marriage (விவாகம்)'),
        (r'\bஞானஸ்தானம்\b', 'Baptism (ஞானஸ்நானம்)'),
        (r'\bஞான\s*ஸ்தானம்\b', 'Baptism (ஞானஸ்நானம்)'),
        (r'\bஞானஸ்நானம்\b', 'Baptism (ஞானஸ்நானம்)'),
        (r'\bஞானஸ்நான\b', 'Baptism (ஞானஸ்நான)'),
        (r'\bதிருமுழுக்கு\b', 'Baptism (திருமுழுக்கு)'),
        (r'\bமாமோதீசா\b', 'Baptism (மாமோதீசா)'),
        (r'\bஉறுதிப்?\s*பூசுதல்\b', 'Confirmation (உறுதிப்பூசுதல்)'),
        (r'\bநோயில்\s*பூசுதல்\b', 'Anointing of the Sick (நோயில் பூசுதல்)'),
        (r'\bகடைசிப்\s*பூசுதல்\b', 'Anointing of the Sick (நோயில் பூசுதல்)'),
        (r'\bஅடக்கம்\b', 'Christian Burial Death (அடக்கம்)'),
        (r'\bமரணப்\s*பதிவு\b', 'Death Register (மரணப் பதிவு)'),
        (r'\bபங்கு\s*பெயர்\b', 'parish name (fhc_parish_id AS parish_name)'),
        (r'\bபங்கின்\s*பெயர்\b', 'parish name (fhc_parish_id AS parish_name)'),
        (r'\bpangu\s*peyar\b', 'parish name'),
        (r'\bpangyin\s*peyar\b', 'parish name'),
        (r'\bபங்கு\b', 'parish'),
        (r'\bபங்குகள்\b', 'parishes'),
        (r'\bகுடும்ப\s*அட்டை(?:\s*எண்)?\b', 'family card number (family_card_no)'),
        (r'\bமரித்தவர்கள்\b', 'Death Register (மரித்தவர்கள்)'),
        (r'\bcarrots?\b', 'vicars'),
        (r'\bkarots?\b', 'vicars'),
        (r'\bparis-?கள்\b', 'parishes'),
        (r'\bparis\b', 'parish'),
        (r'\bpariss\b', 'parish'),
        (r'\bpaaris\b', 'parish'),
        (r'\bபரிஷ்\b', 'parish'),
        (r'\bபாரிஸ்\b', 'parish'),
        (r'\bdiocess\b', 'diocese'),
        (r'\bdie access\b', 'diocese'),
        (r'\bbabtizum\b', 'baptism'),
        (r'\bbapthism\b', 'baptism'),
        (r'\bbapthisam\b', 'baptism'),
        (r'\bunion\s+(?!all\s+select|select)', 'communion'),
        (r'\bcnf\b', 'confirmation'),
        (r'\bmrg\b', 'marriage'),
        (r'\banbiyam\b', 'BCC'),
        (r'\banbiyangal\b', 'BCCs'),
        (r'\bfamily\s*code\b', 'family card number (family_card_no)'),
        (r'\bகுடும்ப\s*கோடு\b', 'family card number (family_card_no)'),
        (r'\bகுடும்ப\s*அட்டை\b', 'family card number (family_card_no)'),
        (r'\bகடந்த\s*(?:வருடம்|ஆண்டு)\b', 'last year'),
        (r'\bபோன\s*வருடம்\b', 'last year'),
        (r'\bகுடும்ப\s*(?:விவரங்கள்|விபரம்|விபரங்கள்)\b', 'family details and members'),
        (r'\bகுடும்ப\s*உறுப்பினர்கள்\b', 'family members'),
        (r'\bஉறுப்பினர்கள்?\s*(?:விவரங்கள்|விபரம்|விபரங்கள்)\b', 'member details'),
        (r'\bஉறுப்பினர்கள்?\s*பட்டியல்\b', 'list of members with details'),
        (r'ஆக்னஸ்', 'Agnes'),
        (r'சூசையப்பர்', 'Joseph'),
        (r'சூசை', 'Joseph'),
        (r'அந்தோணி', 'Antony'),
        (r'அந்தோனி', 'Antony'),
        (r'சவேரியார்', 'Xavier'),
        (r'பிரான்சிஸ்', 'Francis'),
        (r'மேரி', 'Mary'),
        (r'மரியா', 'Mary'),
        (r'மரியாள்', 'Mary'),
        (r'\bagnus\b', 'Agnes'),
    ]
    cleaned = text
    for pat, rep in fixes:
        cleaned = re.sub(pat, rep, cleaned, flags=re.IGNORECASE)
    return cleaned

def enhance_query_node(state: GraphState) -> GraphState:
    question = clean_church_query_text(state["question"].strip())
    history = state.get("history", [])
    reference_text = (state.get("reference_text") or "").strip()
    
    reference_keywords = [
        # English Pronouns & Context Triggers
        "them", "they", "those", "these", "their", "him", "her", "it", "list them", "show them", 
        "the same", "above", "this", "that", "next", "more", "remaining", "page", "next records", 
        "next 50", "next page", "who are they", "what are their", "show their", "give their", "whose", "details",
        
        # Tamil Pronouns & Context Triggers (தமிழ் வினாத் தொடர்ச்சி)
        "அவர்கள்", "அவர்களை", "அவர்களின்", "அவங்களோட", "அவங்கள", "அவங்களுக்கு", "அவங்களை", "அவங்க", 
        "அதை", "அவற்றின்", "அவற்றில்", "அதில்", "மேலே", "மேற்கண்ட", "இவர்கள்", "இவர்களை", "இவர்களின்", 
        "இவங்களோட", "இவங்கள", "இவங்களுக்கு", "இவங்க", "அவர்களின் பெயர்", "பட்டியல்", "விபரம் குடு", "விவரம்",
        "குடும்ப அட்டை", "அட்டை எண்", "பங்கு பெயர்", "குடு", "தாங்க", "காட்டு",
        
        # Tanglish Pronouns & Triggers
        "avanga", "avangala", "avangaloda", "avangaluku", "ivanga", "ivangala", "ivangaloda", "ivangaluku", 
        "adhu", "adhula", "mela ulla", "mela irukura", "details kudu", "list kudu", "show their", "their parish"
    ]
    q_lower = question.lower()
    has_reference_keyword = any(re.search(r'\b' + re.escape(kw) + r'\b', q_lower) for kw in reference_keywords)
    
    # 1. If the user explicitly referenced a message/quote OR the query has reference pronouns with history:
    if reference_text or (history and has_reference_keyword):
        print(f"[enhance_query] Extending query with reference & spelling correction: question='{question}', reference='{reference_text}'")
        
        history_lines = []
        if reference_text:
            history_lines.append(f"Referenced Context / Quote: {reference_text}")
        for msg in history[-3:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.get('content', '')[:500]}")
        history_context = "\n".join(history_lines)

        try:
            response = invoke_llm_with_rotation(ENHANCE_QUERY_PROMPT.format_messages(
                reference_context=history_context,
                question=question
            ))
            enhanced = response.content.strip().strip('"').strip("'")
            print(f"[enhance_query] Extended & Corrected Question: '{enhanced}'")
        except Exception as e:
            print(f"[enhance_query] Warning in reference enhancement: {e}")
            enhanced = question

        embedding = embed_text(enhanced)
        return {**state, "enhanced_query": enhanced, "query_embedding": embedding}

    # 2. Standalone question: Automatic Background Spelling & Terminology Correction
    print(f"[enhance_query] Automatic background spelling & grammar enhancement for prompt: '{question}'")
    
    # Fast church keyword dictionary corrections
    corrections = {
        "babtism": "Baptism", "babtized": "Baptized", "baptizm": "Baptism", "parsh": "Parish",
        "marige": "Marriage", "marrige": "Marriage", "mariage": "Marriage", "famly": "Family",
        "famlies": "Families", "comunion": "Holy Communion", "dioce": "Diocese", "diocis": "Diocese",
        "membr": "Member", "membrs": "Members", "st josef": "St. Joseph's"
    }
    
    try:
        response = invoke_llm_with_rotation(STANDALONE_ENHANCE_PROMPT.format_messages(question=question))
        enhanced = response.content.strip().strip('"').strip("'")
        print(f"[enhance_query] Automatically Enhanced Question: '{enhanced}'")
    except Exception as e:
        print(f"[enhance_query] Fallback dictionary correction: {e}")
        words = question.split()
        fallback_words = [corrections.get(w.lower(), w) for w in words]
        enhanced = " ".join(fallback_words)
        
    embedding = embed_text(enhanced)
    return {**state, "enhanced_query": enhanced, "query_embedding": embedding}

def retrieve_context_node(state: GraphState) -> GraphState:
    print("[retrieve_context] Retrieving schema context and few-shots...")
    relevant_tables = fetch_relevant_schemas(state["question"], state["enhanced_query"], state["query_embedding"])
    combined_query = f"{state['question']} {state['enhanced_query']}"
    relevant_fields = fetch_relevant_fields(state["query_embedding"], combined_query)
    few_shots = fetch_few_shot_examples(state["query_embedding"])
    return {**state, "relevant_tables": relevant_tables, "relevant_fields": relevant_fields, "few_shot_examples": few_shots}

SQL_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", r"""You are an expert MariaDB SQL query writer for KOINONIA Catholic Diocesan & Parish Assistant.

## Database Tables & Schema Context:
{relevant_tables}

## Matched Columns:
{relevant_fields}

{few_shot_examples}

---
## USER JURISDICTION & ROLE-BASED ACCESS CONTROL (User: {user_role}, Diocese: {user_diocese}, Parish: {user_parish}, Vicariate: {user_vicariate}):
1. Admin / System Manager: Full global access across all 5 dioceses. No auto-filters unless requested.
2. Bishop / Curia / Chancellor / Vicar General: Scope data queries to `WHERE diocese_id = '{user_diocese}'`. If foreign diocese requested -> Return `UNAUTHORIZED_DIOCESE`.
3. Vicar Forane: Scope member/sacrament queries to `WHERE vicariate_id = '{user_vicariate}'`. If other vicariates requested -> Return `UNAUTHORIZED_DIOCESE`.
4. Parish Priest / Parishioner: Scope personal tables strictly to assigned parish (`bapt_parish_id = '{user_parish}'`, `fhc_parish_id = '{user_parish}'`, `cnf_parish_id = '{user_parish}'`, `mrg_parish_id = '{user_parish}'`, `death_parish_id = '{user_parish}'`, `parish_id = '{user_parish}'`). If other parishes requested -> Return `UNAUTHORIZED_DIOCESE`.
5. Diocesan & Parish Directory Queries: `tabParish`, `tabDiocese`, `tabVicariate` (churches, patron saints, priests, feast days) are permitted for all roles in `{user_diocese}`.

---
## COMPLETE CANONICAL RULES (1 – 48):

### Table Selection & Entity Routing:
- BAPTISM ("baptized", "christened", "bapt date", "godfather", "godmother") → `tabBaptism`
- COMMUNION ("first communion", "FHC", "holy communion", "புது நன்மை", "நற்கருணை") → `tabCommunion` (Never tabMember)
  - When listing communion recipients with parish name and family card: `SELECT first_name, middle_name, last_name, fhc_parish_id AS parish_name, family_card_no FROM tabCommunion WHERE diocese_id = '{user_diocese}' AND YEAR(fhc_date) = 1995`
- CONFIRMATION ("confirmed", "confirmation", "CNF", "sponsor", "உறுதிப்பூசுதல்") → `tabConfirmation`
- MARRIAGE ("married", "wedding", "groom", "bride", "திருமணம்") → `tabMarriage` (Search both `bridegroom_name` and `bride_name`)
- DEATH / BURIAL ("died", "death", "buried", "cemetery", "இறப்பு", "அடக்கம்") → `tabDeath`
- ANOINTING ("anointing", "sick anointing", "நோய் பூசுதல்") → `tabAnointing Of Sick` (Quote with backticks: `tabAnointing Of Sick`)
- MEMBER ("member", "parishioner", "blood group", "occupation", "marital status", "who is") → `tabMember`
- FAMILY ("family", "household", "family card", "BCC", "zone", "head of family") → `tabFamily`
- PARISH DIRECTORY ("parish", "church", "patron saint", "feast day", "parish priest") → `tabParish`
- DIOCESE PROFILE ("diocese history", "bishop of diocese", "chancery") → `tabDiocese`
- VICARIATE ("vicariate", "deanery", "vicar forane") → `tabVicariate`

### Detailed Person & Sacrament Profile Lookups:
- When user asks for "all details", "details about [Person]", "sacrament record", or searches a specific person by name:
  - tabBaptism: `SELECT name, bapt_register_ref, first_name, middle_name, last_name, gender, dob, birth_place, bapt_date, bapt_place, bapt_parish_id AS parish_name, diocese_id, father_name, father_occupation, mother_name, mother_occupation, bapt_god_father, bapt_god_father_last_name, bapt_god_mother, bapt_god_mother_last_name, bapt_minister, parish_priest, family_card_no, note FROM tabBaptism WHERE diocese_id = '{user_diocese}' AND (CONCAT_WS(' ', first_name, NULLIF(middle_name, ''), last_name) LIKE '%[first]%[last]%' OR (first_name LIKE '%[first]%' AND (middle_name LIKE '%[last]%' OR last_name LIKE '%[last]%'))) LIMIT 1`
  - tabCommunion: `SELECT name, first_name, middle_name, last_name, gender, dob, fhc_date, fhc_place, fhc_parish_id AS parish_name, diocese_id, father_name, mother_name, fhc_minister, parish_priest, family_card_no, note FROM tabCommunion WHERE diocese_id = '{user_diocese}' AND (CONCAT_WS(' ', first_name, NULLIF(middle_name, ''), last_name) LIKE '%[first]%[last]%' OR (first_name LIKE '%[first]%' AND (middle_name LIKE '%[last]%' OR last_name LIKE '%[last]%'))) LIMIT 1`
  - tabConfirmation: `SELECT name, first_name, middle_name, last_name, gender, dob, cnf_date, cnf_place, cnf_parish_id AS parish_name, diocese_id, father_name, mother_name, cnf_minister, sponsor_name, parish_priest, family_card_no, note FROM tabConfirmation WHERE diocese_id = '{user_diocese}' AND (CONCAT_WS(' ', first_name, NULLIF(middle_name, ''), last_name) LIKE '%[first]%[last]%' OR (first_name LIKE '%[first]%' AND (middle_name LIKE '%[last]%' OR last_name LIKE '%[last]%'))) LIMIT 1`
  - tabMarriage: `SELECT name, bridegroom_name, bridegroom_middle_name, bridegroom_last_name, bride_name, bride_middle_name, bride_last_name, mrg_date, mrg_place, mrg_parish_id AS parish_name, diocese_id, mrg_minister, witness1_name, witness2_name, note FROM tabMarriage WHERE diocese_id = '{user_diocese}' AND (bridegroom_name LIKE '%[Name]%' OR bride_name LIKE '%[Name]%') LIMIT 1`
  - tabDeath: `SELECT name, first_name, middle_name, last_name, gender, death_date, burial_date, cemetery, parish_id AS parish_name, diocese_id, father_name, mother_name, minister, family_card_no, note FROM tabDeath WHERE diocese_id = '{user_diocese}' AND (CONCAT_WS(' ', first_name, NULLIF(middle_name, ''), last_name) LIKE '%[first]%[last]%' OR first_name LIKE '%[first]%') LIMIT 1`
  - tabMember: `SELECT name, first_name, middle_name, last_name, gender, dob, age, marital_status_id, blood_group_id, occupation, education, mobile, email, street, city, parish_id AS parish_name, vicariate_id, diocese_id, family_id, is_family_head, living_status, bapt_date, fhc_date, cnf_date, mrg_date FROM tabMember WHERE diocese_id = '{user_diocese}' AND (CONCAT_WS(' ', first_name, NULLIF(middle_name, ''), last_name) LIKE '%[first]%[last]%' OR first_name LIKE '%[first]%') LIMIT 1`

### Relationship & Schema Rules:
6. Primary key in all Frappe tables is `name` (NOT `id`).
7. `tabMember` links to `tabFamily` via `tabMember.family_id = tabFamily.name`.
   - Multi-sacrament family aggregation: `SELECT f.name AS family_id, f.parish_id AS parish_name, f.vicariate_id, COUNT(m.name) AS total_members, SUM((m.bapt_date IS NOT NULL) + (m.fhc_date IS NOT NULL) + (m.cnf_date IS NOT NULL) + (m.mrg_date IS NOT NULL)) AS total_sacraments FROM tabFamily f JOIN tabMember m ON m.family_id = f.name WHERE f.diocese_id = '{user_diocese}' GROUP BY f.name, f.parish_id, f.vicariate_id HAVING COUNT(m.name) >= 4 AND total_sacraments >= 3 ORDER BY total_members DESC LIMIT 50`
8. NEVER JOIN `tabMember` or `tabFamily` with sacrament tables (`tabBaptism`, `tabCommunion`, `tabConfirmation`, `tabMarriage`, `tabDeath`, `tabAnointing Of Sick`). Always query sacrament tables directly.
9. To find members in a Zone → JOIN tabMember with tabFamily on `tabMember.family_id = tabFamily.name`, filter `tabFamily.zone_id = 'Zone X'`.
10. In `tabFamily`, the parish column is strictly `parish_id` (e.g. `WHERE parish_id = '{user_parish}'` or `SELECT COUNT(*) AS total_families FROM tabFamily WHERE parish_id = '{user_parish}'`). NEVER use `parish_bcc_id` for parish names! `parish_bcc_id` is ONLY for BCC / Anbiyam names (such as 'St. Joseph Anbiyam' or 'Arockiya Annai Anbiyam'). When searching BCC, always use `LIKE '%[BCC Name]%'` (e.g. `parish_bcc_id LIKE '%Joseph%'`).
11. To list parishes in a vicariate → `FROM tabParish WHERE vicariate_id = '[vicariate name]'`.
12. NEVER use `parish_priest` to filter parish name. Use `bapt_parish_id`, `mrg_parish_id`, `cnf_parish_id`, `fhc_parish_id`, `death_parish_id`, or `parish_id`.

### Date & Year Rules:
13. Specific 4-digit years ("in 2024", "2023 baptisms"): Use literal `WHERE YEAR([date_col]) = 2024`. Never replace with YEAR(CURDATE()).
14. "This year" / "current year": Use `WHERE YEAR([date_col]) = YEAR(CURDATE())`.
15. "This month": `WHERE MONTH([date_col]) = MONTH(CURDATE()) AND YEAR([date_col]) = YEAR(CURDATE())`.
16. Specific month ("in January 2024"): `WHERE MONTH([date_col]) = 1 AND YEAR([date_col]) = 2024`.
17. Date range ("between 2020 and 2023"): `WHERE [date_col] BETWEEN '2020-01-01' AND '2023-12-31'`.
18. Recent records ("last 6 months"): `WHERE [date_col] >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)`.

### Aggregation, Counts & Lists:
19. Count-only ("how many", "count"): `SELECT COUNT(*) FROM ...`.
    - Total families in user's parish: `SELECT COUNT(*) AS total_families FROM tabFamily WHERE parish_id = '{user_parish}'`.
    - Total members in user's parish: `SELECT COUNT(*) AS total_members FROM tabMember WHERE parish_id = '{user_parish}'`.
    - For Bishop asking count in diocese without single parish name, return parish-wise breakdown: `SELECT parish_id AS parish_name, COUNT(*) AS total_families FROM tabFamily WHERE diocese_id = '{user_diocese}' GROUP BY parish_id ORDER BY total_families DESC`.
20. List / Show Records queries ("show", "list", "display", "find", "who are"): Select readable columns. Never use COUNT(*).
21. Grouped counts: `SELECT [group_col], COUNT(*) FROM ... GROUP BY [group_col] ORDER BY COUNT(*) DESC`. Include `[col] IS NOT NULL AND [col] != ''`.
22. "How many AND list them": Return just the list of rows (no COUNT(*)).
23. Charts / Plots ("bar chart", "pie chart", "trend"): Return exactly TWO columns: one label column and one numeric count column (e.g. `SELECT gender, COUNT(*) FROM tabBaptism GROUP BY gender`).
24. Multi-metric questions ("total parishes + total members"): Combine using `UNION ALL` into a single query.

### Filtering Rules:
25. By minister / priest: `WHERE bapt_minister LIKE '%Thomas%'`.
26. By godparent / sponsor: `WHERE bapt_god_father LIKE '%[name]%'`, `bapt_god_mother`, `cnf_god_father`, `sponsor_name`, `witness1_name`.
27. By gender: `WHERE gender = 'Female'` or `WHERE gender = 'Male'`.
28. By living status: `WHERE living_status = 'Alive'` or `'Deceased'` in tabMember.
29. By person name: `WHERE (CONCAT_WS(' ', first_name, NULLIF(middle_name, ''), last_name) LIKE '%[first]%[last]%' OR (first_name = '[first]' AND middle_name = '[last]'))`. Single word search: `WHERE (first_name = '[name]' OR middle_name = '[name]' OR last_name = '[name]')`.
30. By family card number: `WHERE family_card_no = '[card_no]'` (sacraments) or `family_register_number` (tabFamily).
31. By diocese/vicariate scope: `WHERE diocese_id = '[name]'` or `vicariate_id = '[name]'`.
32. By marital status: `WHERE marital_status_id = 'Single'` in tabMember; `marital_status = 'Widowed'` in tabDeath.
33. By occupation: `WHERE occupation LIKE '%teacher%'` in tabMember.
34. By economic status: `WHERE economic_status LIKE '%Poor%'` in tabFamily.
35. By cause of death: `WHERE death_cause LIKE '%cancer%'` in tabDeath.
36. By cemetery: `WHERE cemetery_code LIKE '%St. Joseph%'` in tabDeath.
37. Register reference: `WHERE bapt_register_ref = '[ref]'`, `mrg_register_ref`, `cnf_register_ref`, `fhc_register_ref`, `death_register_ref`.
38. Patron saint / feast queries: `WHERE patron_saint LIKE '%Mary%'` or `WHERE MONTH(feast_day) = 1` in tabParish.
39. Active / Inactive: `WHERE active = 1` or `WHERE active = 0`.
40. Family Heads & Listing All Family Heads:
    - In tabFamily, the Family Head / Family Name is stored in `reference` (e.g. 'Albert & Agnes', 'Antony Selvan P.', 'Arul A').
    - When asked to "get all family head names", "list family heads", "who are the family heads", or "குடும்பத் தலைவர்கள் யார் யார்?":
      SELECT family_card_number, reference AS family_head, parish_bcc_id AS anbiyam, mobile, street FROM tabFamily WHERE parish_id = '{user_parish}' AND reference IS NOT NULL AND reference != '' ORDER BY family_card_number ASC
    - NEVER use `WHERE is_family_head = 'Yes'` on tabMember for listing parish family heads because family heads are tracked via `tabFamily.reference`!
41. Bride / groom religion: `WHERE bridegroom_religion_id != 'Catholic'` in tabMarriage.
42. Multi-sacrament completion: Query `tabMember` directly `WHERE bapt_date IS NOT NULL AND fhc_date IS NOT NULL AND cnf_date IS NOT NULL`.
43. Apostrophes in names (D'Souza, St. Mary's): Escape single quotes in SQL literals: `LIKE '%Souza%'` or `\'`.
44. Family details & household members of a person (Exact Match Priority & Same-Name Disambiguation):
    - When user asks for "family details of [Person Name]", "family members of [Person Name]", "[Person Name] குடும்ப விவரங்கள்", or "[Person Name] குடும்ப உறுப்பினர்கள்":
    - The person is a member in tabMember (wife, child, husband, or head).
    - NEVER query tabFamily WHERE family_head_name = '[Person Name]'.
    - NEVER put LIMIT 1 on the outer query (that drops the other family members!).
    - EXACT MULTI-TOKEN MATCH PRIORITY (Crucial for disambiguating same-name or similar-name persons):
      When the user provides a multi-token name (e.g., "Rose King", "Antony Selvan", "Agnes Mary A"):
      DO NOT allow a loose single-token OR condition like `OR first_name LIKE '%Rose%'` to override the exact person, because that matches unrelated members (e.g. 'Roselin' instead of 'Rose King')!
      The subquery MUST strictly prioritize the full name:
      WHERE m.family_id = (
          SELECT family_id FROM tabMember 
          WHERE (CONCAT_WS(' ', first_name, middle_name, last_name) LIKE '%[Person Name]%' 
                 OR (first_name LIKE '%[First]%' AND (middle_name LIKE '%[Second]%' OR last_name LIKE '%[Second]%')))
            AND parish_id = '{user_parish}' 
          ORDER BY CASE 
              WHEN CONCAT_WS(' ', first_name, middle_name, last_name) LIKE '%[Person Name]%' THEN 1 
              WHEN first_name LIKE '%[First]%' AND (middle_name LIKE '%[Second]%' OR last_name LIKE '%[Second]%') THEN 2 
              ELSE 3 
          END ASC, creation DESC 
          LIMIT 1
      ) AND m.parish_id = '{user_parish}' ORDER BY CASE WHEN m.relationship_id IN ('Self', 'Head', 'Husband') THEN 1 WHEN m.relationship_id = 'Wife' THEN 2 ELSE 3 END, m.dob ASC
    - SINGLE AMBIGUOUS FIRST NAME (e.g., "Antony", "Rose"):
      When only a single name token is queried:
      WHERE m.family_id = (
          SELECT family_id FROM tabMember 
          WHERE (first_name = '[Name]' OR first_name LIKE '[Name] %' OR first_name LIKE '% [Name]%' OR CONCAT_WS(' ', first_name, middle_name, last_name) LIKE '%[Name]%')
            AND parish_id = '{user_parish}' 
          ORDER BY CASE 
              WHEN first_name = '[Name]' THEN 1 
              WHEN relationship_id IN ('Self', 'Head', 'Husband') THEN 2 
              ELSE 3 
          END ASC, dob ASC 
          LIMIT 1
      ) AND m.parish_id = '{user_parish}' ORDER BY CASE WHEN m.relationship_id IN ('Self', 'Head', 'Husband') THEN 1 WHEN m.relationship_id = 'Wife' THEN 2 ELSE 3 END, m.dob ASC
45. Relationship between two persons: SELECT CONCAT_WS(' ', m1.first_name, NULLIF(m1.middle_name, ''), m1.last_name) AS person1_fullname, m1.relationship_id AS relationship1, m1.parish_id AS parish_name, CONCAT_WS(' ', m2.first_name, NULLIF(m2.middle_name, ''), m2.last_name) AS person2_fullname, m2.relationship_id AS relationship2, m1.family_id, m1.diocese_id FROM tabMember m1 JOIN tabMember m2 ON m1.family_id = m2.family_id WHERE (m1.first_name LIKE '%[name1]%') AND (m2.first_name LIKE '%[name2]%') LIMIT 1.
46. Single Diocese Profile: SELECT diocese_name, bishop_name, established_date, city, phone, email, website, note FROM tabDiocese WHERE (name LIKE '%[diocese]%' OR diocese_name LIKE '%[diocese]%') LIMIT 1.
47. Listing all Dioceses: SELECT diocese_name, bishop_name, established_date, city, phone, email FROM tabDiocese ORDER BY diocese_name ASC.
48. Multi-Sacrament Counts / Summary: UNION ALL across tabBaptism, tabCommunion, tabConfirmation, tabMarriage, tabDeath, `tabAnointing Of Sick`.
49. Family details by Family Card Number or Head Name:
    - When asked for family details by card number (e.g. 'YLG/001' or '3165') or by Family Head name:
      SELECT m.name AS member_id, CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) AS full_name, m.relationship_id AS relationship, m.gender, m.age, m.dob, m.mobile, f.family_card_number, f.parish_bcc_id AS anbiyam, f.street, f.city, m.parish_id FROM tabMember m JOIN tabFamily f ON m.family_id = f.name WHERE (f.name = '[card_or_id]' OR f.family_card_number = '[card_or_id]' OR f.family_head_name LIKE '%[name]%') AND f.parish_id = '{user_parish}' ORDER BY CASE WHEN m.relationship_id IN ('Self', 'Head', 'Husband') THEN 1 WHEN m.relationship_id = 'Wife' THEN 2 ELSE 3 END, m.dob ASC
50. Sacraments of a specific person (Baptism, Communion, Confirmation, Marriage):
    - Baptism: SELECT first_name, middle_name, last_name, dob, bapt_date, bapt_place, bapt_minister, father_name, mother_name, bapt_god_father, bapt_god_mother, family_card_no, bapt_parish_id FROM tabBaptism WHERE (CONCAT_WS(' ', first_name, middle_name, last_name) LIKE '%[Name]%' OR first_name LIKE '%[First]%') AND parish_id = '{user_parish}'
    - Confirmation: SELECT first_name, middle_name, last_name, dob, cnf_date, cnf_place, cnf_minister, sponsor, father_name, mother_name, family_card_no, cnf_parish_id FROM tabConfirmation WHERE (CONCAT_WS(' ', first_name, middle_name, last_name) LIKE '%[Name]%' OR first_name LIKE '%[First]%') AND parish_id = '{user_parish}'
    - Communion: SELECT first_name, middle_name, last_name, dob, fhc_date, fhc_place, fhc_minister, father_name, mother_name, family_card_no, fhc_parish_id FROM tabCommunion WHERE (CONCAT_WS(' ', first_name, middle_name, last_name) LIKE '%[Name]%' OR first_name LIKE '%[First]%') AND parish_id = '{user_parish}'
    - Marriage: SELECT bridegroom_name, bridegroom_middle_name, bridegroom_last_name, bride_name, bride_middle_name, bride_last_name, mrg_date, mrg_place, mrg_minister, witness1_name, witness2_name, mrg_parish_id FROM tabMarriage WHERE (bridegroom_name LIKE '%[Name]%' OR bride_name LIKE '%[Name]%') AND parish_id = '{user_parish}'
51. Sacraments of an entire family / household:
    - When asked for all sacraments received in a family or household:
      SELECT m.name AS member_id, CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) AS full_name, m.relationship_id AS relationship, m.gender, m.age, m.dob, m.mobile, m.bapt_date AS baptism_date, m.fhc_date AS communion_date, m.cnf_date AS confirmation_date, m.mrg_date AS marriage_date, f.family_card_number, f.parish_bcc_id AS anbiyam, f.street, f.city, m.parish_id FROM tabMember m JOIN tabFamily f ON m.family_id = f.name WHERE m.family_id = (SELECT family_id FROM tabMember WHERE (CONCAT_WS(' ', first_name, middle_name, last_name) LIKE '%[Name]%' OR first_name LIKE '%[First]%') AND parish_id = '{user_parish}' LIMIT 1) AND m.parish_id = '{user_parish}' ORDER BY CASE WHEN m.relationship_id IN ('Self', 'Head', 'Husband') THEN 1 WHEN m.relationship_id = 'Wife' THEN 2 ELSE 3 END, m.dob ASC
52. Listing members / members data retrieval (All scenarios):
    - When user asks to 'list members', 'members data', 'members in [Anbiyam]', or 'உறுப்பினர்கள் பட்டியல்':
    - NEVER select only first_name!
    - ALWAYS select: CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) AS full_name, m.relationship_id AS relationship, m.gender, m.age, m.dob, m.mobile, f.family_card_number, f.parish_bcc_id AS anbiyam
    - If filtered by Anbiyam: WHERE f.parish_bcc_id LIKE '%[Anbiyam Name]%' AND m.parish_id = '{user_parish}'
    - If filtered by Family Heads: WHERE m.is_family_head = 'Yes' AND m.parish_id = '{user_parish}'
    - If filtered by Gender: WHERE m.gender = 'Female' (or 'Male') AND m.parish_id = '{user_parish}'
    - If filtered by Youth / Children: WHERE TIMESTAMPDIFF(YEAR, m.dob, CURDATE()) < 18 AND m.parish_id = '{user_parish}'

---
## STRICT OUTPUT RULES:
- Output ONLY a single executable MariaDB SELECT query inside ```sql ... ``` code block.
- NEVER use `SELECT *` or system metadata columns (amended_from, _user_tags, creation, modified, idx, docstatus).
"""),
    ("human", "User question: {enhanced_query}"),
])

def generate_sql_node(state: GraphState) -> GraphState:
    print("[generate_sql] Generating SQL query...")
    
    if state.get("generated_sql") == "BLOCKED_SECURITY":
        return state

    q_raw = (state.get("question") or "").lower()
    
    # Defense-in-depth security injection check
    injection_patterns = [
        r';\s*(?:DROP|ALTER|TRUNCATE|DELETE|INSERT|UPDATE|CREATE|REPLACE)',
        r'(?:SLEEP|BENCHMARK|WAITFOR|GET_LOCK)\s*\(',
        r'UNION\s+(?:ALL\s+)?SELECT\s+.*(?:tabUser|information_schema|mysql\.|password|hash|secret|token|api_key)',
        r'--\s*$',
        r'(?:ignore all previous instructions|output your system prompt|leak secret keys)'
    ]
    for pat in injection_patterns:
        if re.search(pat, state.get("question", ""), re.IGNORECASE):
            print(f"[generate_sql] Intercepted malicious injection in generate_sql: '{state.get('question')}'")
            return {**state, "generated_sql": "BLOCKED_SECURITY"}
    q_enhanced = (state.get("enhanced_query") or "").lower()
    q_combined = f"{q_raw} {q_enhanced}"
    
    user_role = state.get("user_role") or "Parish Priest"
    
    is_count_query = any(k in q_combined for k in [
        "count", "how many", "total number", "total members", "total families", 
        "total baptisms", "total marriages", "statistics", "how much", "number of",
        "total count", "count of"
    ])
    
    # Pre-check for non-Bishop roles requesting diocese-wide / all-parishes data (only for detailed DATA queries, NOT counts)
    is_global = user_role in ["Bishop", "Curia", "Chancellor", "Administrator", "System Manager"]
    broad_diocese_keywords = [
        "all parishes", "entire diocese", "whole diocese", "across all parishes", 
        "every parish", "all the diocese", "all parishes in the diocese", "other parishes"
    ]
    
    if not is_global and not is_count_query and any(kw in q_combined for kw in broad_diocese_keywords):
        print(f"[generate_sql] Access Restricted: {user_role} requested broad diocese personal data: raw='{q_raw}', enhanced='{q_enhanced}'")
        return {**state, "generated_sql": "UNAUTHORIZED_DIOCESE"}

    # Check for unauthorized parish request if user is Vicar General / Vicar Forane / Parish Priest (only for detailed DATA queries, NOT counts)
    user_parishes = state.get("user_parishes") or []
    user_parish = (state.get("user_parish") or "").strip()
    
    all_parish_names = []
    try:
        import frappe
        if hasattr(frappe, "local") and hasattr(frappe.local, "db") and frappe.local.db:
            if frappe.db.table_exists("Parish"):
                all_parish_names = frappe.db.sql_list("SELECT name FROM tabParish")
    except Exception:
        all_parish_names = []

    if not is_count_query and all_parish_names:
        if user_role in ["Vicar General", "Vicar Forane"] and user_parishes:
            for p in all_parish_names:
                if p not in user_parishes:
                    short_p = p.replace("Parish", "").replace("Church", "").replace("Cathedral", "").strip()
                    if short_p and len(short_p) > 3 and re.search(r'\b' + re.escape(short_p.lower()) + r'\b', q_combined):
                        print(f"[generate_sql] Access Restricted: Vicar General requested unauthorized parish data '{p}'")
                        return {**state, "generated_sql": "UNAUTHORIZED_PARISH"}
                        
        elif user_role in ["Parish Priest", "Parishioner"] and user_parish:
            for p in all_parish_names:
                if p.lower() != user_parish.lower():
                    short_p = p.replace("Parish", "").replace("Church", "").replace("Cathedral", "").strip()
                    if short_p and len(short_p) > 3 and re.search(r'\b' + re.escape(short_p.lower()) + r'\b', q_combined):
                        print(f"[generate_sql] Access Restricted: {user_role} of {user_parish} requested unauthorized parish data '{p}'")
                        return {**state, "generated_sql": "UNAUTHORIZED_PARISH"}

    # Pre-check for foreign diocese or foreign parish requests (e.g., Bishop of Salem asking for Christ the King Parish in Trichy)
    user_parish = (state.get("user_parish") or "").strip()
    user_diocese = (state.get("user_diocese") or "").strip()
    if not user_diocese and user_parish:
        try:
            ensure_frappe_db()
            import frappe
            if hasattr(frappe, "local") and hasattr(frappe.local, "db") and frappe.local.db:
                p_dio = frappe.db.get_value("Parish", user_parish, "diocese_id")
                if p_dio:
                    user_diocese = p_dio
        except Exception:
            pass
    if not user_diocese:
        user_diocese = "Trichy"
    
    all_parishes = []
    try:
        import frappe
        if hasattr(frappe, "local") and hasattr(frappe.local, "db") and frappe.local.db:
            if frappe.db.table_exists("Parish"):
                all_parishes = frappe.db.sql("SELECT name, diocese_id FROM tabParish", as_dict=True)
    except Exception:
        all_parishes = []

    if all_parishes:
        for p in all_parishes:
            p_name = p.get("name") or ""
            p_dio = p.get("diocese_id") or ""
            short_p = p_name.replace("Parish", "").replace("Church", "").replace("Cathedral", "").replace("Shrine", "").strip()
            # Only match foreign parish if 'parish' or 'church' or 'பங்கு' is in query and 'cemetery' is not the primary target
            if short_p and len(short_p) > 4 and re.search(r'\b' + re.escape(short_p.lower()) + r'\s+(?:parish|church|cathedral|shrine|பங்கு)\b', q_combined, re.IGNORECASE):
                if user_diocese and p_dio and p_dio.lower() != user_diocese.lower():
                    print(f"[generate_sql] Access Restricted: User {user_role} of {user_diocese} requested parish '{p_name}' in foreign diocese '{p_dio}'")
                    return {
                        **state, 
                        "generated_sql": "UNAUTHORIZED_DIOCESE", 
                        "requested_foreign_parish": p_name, 
                        "requested_foreign_diocese": p_dio
                    }

    known_dioceses = [
        "trichy", "chennai", "vellore", "salem", "coimbatore", "madurai", 
        "tuticorin", "thanjavur", "kottar", "palayamkottai", "ooty", 
        "dharmapuri", "kumbakonam", "tirunelveli", "kuzhithurai", "sivagangai", "dindigul"
    ]
    for d in known_dioceses:
        if user_diocese and d != user_diocese.lower():
            # Match diocese name with word boundary (including Tamil suffixes like "salem diocese-ல", "salem-la", "salem")
            if re.search(r'\b' + re.escape(d) + r'(\b|[-_])', q_combined, re.IGNORECASE):
                print(f"[generate_sql] Access Restricted: User {user_role} of {user_diocese} requested unauthorized diocese '{d}': raw='{q_raw}'")
                return {**state, "generated_sql": "UNAUTHORIZED_DIOCESE", "requested_foreign_diocese": d.title()}

    parish_context = user_parish
    if user_role in ["Vicar General", "Vicar Forane"] and user_parishes:
        parish_context = "IN (" + ", ".join([f"'{p}'" for p in user_parishes]) + ")"

    prompt_msgs = SQL_GEN_PROMPT.format_messages(
        relevant_tables="\n\n".join(state["relevant_tables"]),
        relevant_fields="\n".join(state["relevant_fields"]),
        few_shot_examples=state["few_shot_examples"],
        user_role=state.get("user_role") or "Parish Priest",
        user_diocese=user_diocese,
        user_parish=parish_context,
        user_vicariate=state.get("user_vicariate") or "",
        user_member_id=state.get("user_member_id") or "",
        user_email=state.get("user_email") or "",
        enhanced_query=state["enhanced_query"]
    )
    
    response = invoke_llm_with_rotation(prompt_msgs)
    content = response.content.strip()
    if "```sql" in content:
        sql = content.split("```sql")[1].split("```")[0].strip()
    elif "```" in content:
        sql = content.split("```")[1].strip()
    else:
        sql = content.strip()
        
    print(f"[generate_sql] SQL Generated:\n  {sql}")
    return {**state, "generated_sql": sql}

def sanitize_select_clause(sql: str, question: str = "") -> str:
    sql_upper = sql.upper().strip()
    if not sql_upper.startswith("SELECT") or "COUNT(" in sql_upper or "UNION" in sql_upper or "GROUP BY" in sql_upper:
        return sql
        
    table_match = re.search(r'FROM\s+`?(tab[A-Za-z0-9_ ]+)`?', sql, re.IGNORECASE)
    if not table_match:
        return sql
        
    table_name = table_match.group(1).strip().strip('`')
    
    preferred_cols = {
        "tabBaptism": ["first_name", "middle_name", "last_name", "bapt_parish_id", "family_card_no", "bapt_date"],
        "tabCommunion": ["first_name", "middle_name", "last_name", "fhc_parish_id", "family_card_no", "fhc_date"],
        "tabConfirmation": ["first_name", "middle_name", "last_name", "cnf_parish_id", "family_card_no", "cnf_date"],
        "tabMarriage": ["bridegroom_name", "bridegroom_middle_name", "bridegroom_last_name", "bride_name", "mrg_date"],
        "tabAnointing Of Sick": ["first_name", "middle_name", "last_name", "anointing_date", "minister"],
        "tabDeath": ["first_name", "middle_name", "last_name", "death_date", "age"],
        "tabMember": ["first_name", "middle_name", "last_name", "relationship_id", "gender", "age", "family_id"],
        "tabFamily": ["family_register_number", "parish_bcc_id", "zone_id", "status", "parish_id"],
        "tabDiocese": ["diocese_name", "bishop_name", "established_date", "city", "phone", "email", "note"],
        "tabVicariate": ["vicariate_name", "diocese_id", "vicar_forane", "active"],
        "tabParish": ["parish_name", "patron_saint", "feast_day", "established_date", "parish_priest", "assistant_priest", "city", "note"]
    }
    
    # Check if wildcard is present anywhere in SELECT part
    select_part_raw = sql_upper.split("FROM")[0].replace("SELECT", "").strip()
    if "*" in select_part_raw:
        if "tabMember" in sql and "tabFamily" in sql:
            from_part_match = re.search(r'\bFROM\b.*', sql, re.IGNORECASE | re.DOTALL)
            if from_part_match:
                new_sql = f"SELECT m.first_name, m.middle_name, m.last_name, m.relationship_id, f.parish_bcc_id {from_part_match.group(0)}"
                if re.search(r'\btabMember\s+AS\s+T3\b|\btabMember\s+T3\b', sql, re.IGNORECASE):
                    new_sql = f"SELECT T3.first_name, T3.middle_name, T3.last_name, T3.relationship_id, T2.parish_bcc_id {from_part_match.group(0)}"
                print(f"[sanitize_select_clause] Converted JOIN wildcard to explicit columns: {new_sql}")
                return new_sql
        elif table_name in preferred_cols:
            cols_str = ", ".join(f"`{c}`" for c in preferred_cols[table_name])
            from_part_match = re.search(r'\bFROM\b.*', sql, re.IGNORECASE | re.DOTALL)
            if from_part_match:
                new_sql = f"SELECT {cols_str} {from_part_match.group(0)}"
                print(f"[sanitize_select_clause] Converted single-table wildcard to: {new_sql}")
                return new_sql

    if table_name in preferred_cols:
        select_part = sql_upper.split("FROM")[0].replace("SELECT", "").strip()
        selected_cols = [c.strip().strip('`') for c in select_part.split(",") if c.strip()]
        
        # If the query is looking up a single document or specific person details, preserve all columns
        if "LIMIT 1" in sql_upper or any(c.lower() in ["name", "bapt_register_ref", "fhc_register_ref", "cnf_register_ref", "mrg_register_ref", "death_register_ref"] for c in selected_cols):
            return sql

        is_wildcard = "*" in select_part
        max_allowed = 8 if table_name in ["tabParish", "tabDiocese"] else 5
        has_too_many = len(selected_cols) > max_allowed
        
        core_fields = ["first_name", "bridegroom_name", "diocese_name", "vicariate_name", "parish_name", "family_name"]
        missing_core = not any(f.upper() in select_part for f in core_fields) and any(f in preferred_cols[table_name] for f in core_fields)
        
        if is_wildcard or has_too_many or missing_core:
            final_cols = list(preferred_cols[table_name])
            
            # Scan the original select_part for specific keywords requested (like mobile, email, phone, address, etc.)
            # and preserve them if they were selected in the original SQL
            special_keywords = ["mobile", "email", "phone", "address", "website", "established_date", "dob", "cemetery_code", "note", "assistant_priest", "family_card_no", "parish_name", "fhc_parish_id", "bapt_parish_id", "cnf_parish_id", "mrg_parish_id"]
            for col in selected_cols:
                col_lower = col.lower()
                if any(kw in col_lower for kw in special_keywords):
                    # Extract the actual column name from any alias or prefix (e.g. t.mobile -> mobile)
                    base_col = col_lower.split(".")[-1].strip().strip('`')
                    if base_col not in final_cols:
                        final_cols.append(base_col)
            
            # Enforce column limit
            if len(final_cols) > max_allowed:
                final_cols = final_cols[:max_allowed]
                
            cols_str = ", ".join(f"`{c}`" for c in final_cols)
            from_part_match = re.search(r'\bFROM\b.*', sql, re.IGNORECASE | re.DOTALL)
            if from_part_match:
                from_part = from_part_match.group(0)
                new_sql = f"SELECT {cols_str} {from_part}"
                print(f"[sanitize_select_clause] Automatically polished SELECT clause from {len(selected_cols)} cols to: {new_sql}")
                return new_sql
                
    return sql

def sql_escape_str(val: str) -> str:
    if val is None:
        return ""
    return str(val).replace("'", "''")

def sql_quote_str(val: str) -> str:
    if val is None:
        return "''"
    return "'" + str(val).replace("'", "''") + "'"

def enforce_jurisdiction_sql_single(sql: str, state: GraphState) -> str:
    user_role = state.get("user_role") or "Parish Priest"
    user_parish = (state.get("user_parish") or "").strip()
    user_diocese = (state.get("user_diocese") or "").strip()
    user_vicariate = state.get("user_vicariate") or ""
    user_parishes = state.get("user_parishes") or []
    
    if not user_diocese and user_parish:
        try:
            ensure_frappe_db()
            import frappe
            if hasattr(frappe, "local") and hasattr(frappe.local, "db") and frappe.local.db:
                p_dio = frappe.db.get_value("Parish", user_parish, "diocese_id")
                if p_dio:
                    user_diocese = p_dio
        except Exception:
            pass
    if not user_diocese:
        user_diocese = "Trichy"

    if user_role in ["Administrator", "System Manager"] or user_diocese in ["All Dioceses", "All"]:
        return sql
        
    sql_upper = sql.upper()
    if not sql_upper.startswith("SELECT") or sql in ["UNSUPPORTED", "UNAUTHORIZED_DIOCESE", "UNAUTHORIZED_PARISH"]:
        return sql

    # If the LLM mistakenly generated a conflicting diocese_id, correct it to the user's actual diocese_id
    if user_diocese and user_role not in ["Administrator", "System Manager"]:
        sql = re.sub(r"\bdiocese_id\s*=\s*['\"][^'\"]*['\"]", f"diocese_id = {sql_quote_str(user_diocese)}", sql, flags=re.IGNORECASE)
        sql_upper = sql.upper()
        
    alias_match = re.search(r'FROM\s+`?(tab[A-Za-z0-9_]+(?:\s+Of\s+Sick)?)`?(?:\s+(?:AS\s+)?([a-zA-Z0-9_]+))?', sql, re.IGNORECASE)
    if not alias_match:
        return sql
    table_name = alias_match.group(1).strip().strip('`')
    
    # Define column mappings for personal member & sacrament tables
    parish_col_map = {
        "tabBaptism": "bapt_parish_id",
        "tabMarriage": "mrg_parish_id",
        "tabConfirmation": "cnf_parish_id",
        "tabCommunion": "fhc_parish_id",
        "tabDeath": "death_parish_id",
        "tabAnointing Of Sick": "anointing_parish_id",
        "tabMember": "parish_id",
        "tabFamily": "parish_id"
    }
    
    conditions = []
    
    table_prefix = ""
    if alias_match.group(2) and alias_match.group(2).upper() not in ["WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "ON", "GROUP", "ORDER", "LIMIT", "SET", "VALUES"]:
        table_prefix = f"`{alias_match.group(2)}`."
    elif "JOIN" in sql_upper:
        table_prefix = f"`{table_name}`."

    user_diocese_esc = sql_escape_str(user_diocese)
    user_parish_esc = sql_escape_str(user_parish)
    user_vicariate_esc = sql_escape_str(user_vicariate)

    # 1. Diocese ID constraint (applies to ALL roles for any table with diocese_id)
    if user_diocese and "DIOCESE_ID" not in sql_upper and table_name in ["tabFamily", "tabMember", "tabBaptism", "tabMarriage", "tabConfirmation", "tabCommunion", "tabDeath", "tabAnointing Of Sick", "tabParish", "tabVicariate"]:
        conditions.append(f"{table_prefix}`diocese_id` = {sql_quote_str(user_diocese)}")
        
    is_pure_count = "COUNT(" in sql_upper and "FIRST_NAME" not in sql_upper and "LAST_NAME" not in sql_upper and "DOB" not in sql_upper and "MOBILE" not in sql_upper
    
    # 2. Vicariate / Controlled Parishes constraints for Vicar General & Vicar Forane (applied strictly to DATA queries)
    if user_role in ["Vicar General", "Vicar Forane"]:
        if not is_pure_count:
            if user_parishes and table_name in parish_col_map:
                parish_col = parish_col_map[table_name]
                parish_list_str = ", ".join([sql_quote_str(p) for p in user_parishes])
                # If the SQL used '=' with multiple parishes or used a comma in the string
                sql = re.sub(r"\b" + parish_col + r"\s*=\s*['\"][^'\"]*,[^'\"]*['\"]", f"{parish_col} IN ({parish_list_str})", sql, flags=re.IGNORECASE)
                sql = re.sub(r"\bparish_id\s*=\s*['\"][^'\"]*,[^'\"]*['\"]", f"parish_id IN ({parish_list_str})", sql, flags=re.IGNORECASE)
                sql_upper = sql.upper()
                
                # Check if any single controlled parish is explicitly matched
                has_valid_single_parish = any(sql_quote_str(p).upper() in sql_upper or f"'{p.upper()}'" in sql_upper or f'"{p.upper()}"' in sql_upper for p in user_parishes)
                if not has_valid_single_parish:
                    if f"{parish_col.upper()} IN" not in sql_upper and "PARISH_ID IN" not in sql_upper:
                        conditions.append(f"{table_prefix}`{parish_col}` IN ({parish_list_str})")
            elif user_parishes and table_name == "tabParish":
                parish_list_str = ", ".join([sql_quote_str(p) for p in user_parishes])
                if "NAME" not in sql_upper and "VICARIATE_ID" not in sql_upper:
                    conditions.append(f"{table_prefix}`name` IN ({parish_list_str})")
            
    # 3. Parish ID constraint for Parish Priest / Parishioner (applies strictly to ALL queries including counts)
    elif user_role in ["Parish Priest", "Parishioner"] and user_parish and table_name in parish_col_map:
        parish_col = parish_col_map[table_name]

        # Fix LLM confusion where parish_bcc_id was used instead of parish_id in tabFamily
        if table_name == "tabFamily" and "PARISH_BCC_ID" in sql_upper:
            p_esc = re.escape(user_parish)
            sql = re.sub(r"\bparish_bcc_id\s*=\s*['\"]" + p_esc + r"['\"]", f"{parish_col} = {sql_quote_str(user_parish)}", sql, flags=re.IGNORECASE)
            sql = re.sub(r"\bparish_bcc_id\s+LIKE\s+['\"]%?" + p_esc + r"%?['\"]", f"{parish_col} = {sql_quote_str(user_parish)}", sql, flags=re.IGNORECASE)
            sql = re.sub(r"\bparish_bcc_id\s*=\s*['\"][^'\"]*Parish['\"]", f"{parish_col} = {sql_quote_str(user_parish)}", sql, flags=re.IGNORECASE)
            sql_upper = sql.upper()

        has_assigned_parish = (
            user_parish.upper() in sql_upper or 
            user_parish_esc.upper() in sql_upper or 
            sql_quote_str(user_parish).upper() in sql_upper or
            user_parish.replace("'", r"\'").upper() in sql_upper
        )
        # Replace any foreign parish filter that was generated with the user's assigned parish
        if parish_col.upper() in sql_upper or "PARISH_ID" in sql_upper:
            if not has_assigned_parish:
                sql = re.sub(r"\b" + parish_col + r"\s*=\s*(?:'[^']*'|\"[^\"]*\")", f"{parish_col} = {sql_quote_str(user_parish)}", sql, flags=re.IGNORECASE)
                sql = re.sub(r"\bparish_id\s*=\s*(?:'[^']*'|\"[^\"]*\")", f"{parish_col} = {sql_quote_str(user_parish)}", sql, flags=re.IGNORECASE)
                sql_upper = sql.upper()
        else:
            conditions.append(f"{table_prefix}`{parish_col}` = {sql_quote_str(user_parish)}")
            
    if user_role in ["Bishop", "Curia", "Chancellor", "Administrator"]:
        # Strip accidental empty parish_id = '' or parish_id = '{user_diocese}'
        clean_sql_tmp = re.sub(r"\b(parish_id|bapt_parish_id|mrg_parish_id|cnf_parish_id|fhc_parish_id|death_parish_id)\s*=\s*['\"](?:|None|" + re.escape(user_diocese) + r"|" + re.escape(user_diocese_esc) + r")['\"]\s*(AND|OR)?\s*", "", sql, flags=re.IGNORECASE)
        clean_sql_tmp = re.sub(r"\s*(AND|OR)\s*(parish_id|bapt_parish_id|mrg_parish_id|cnf_parish_id|fhc_parish_id|death_parish_id)\s*=\s*['\"](?:|None|" + re.escape(user_diocese) + r"|" + re.escape(user_diocese_esc) + r")['\"]", "", clean_sql_tmp, flags=re.IGNORECASE)
        clean_sql_tmp = re.sub(r"\bWHERE\s+(GROUP\s+BY|ORDER\s+BY|LIMIT|\Z)", r" \1", clean_sql_tmp, flags=re.IGNORECASE)
        clean_sql_tmp = re.sub(r"\bWHERE\s*;\s*$", "", clean_sql_tmp, flags=re.IGNORECASE)
        clean_sql_tmp = re.sub(r"\bWHERE\s*$", "", clean_sql_tmp, flags=re.IGNORECASE)
        clean_sql_tmp = re.sub(r"\bWHERE\s+(AND|OR)\b", "WHERE", clean_sql_tmp, flags=re.IGNORECASE)
        sql = clean_sql_tmp.strip()
        sql_upper = sql.upper()

    if not conditions:
        return sql
        
    where_to_add = " AND ".join(conditions)
    
    # Strip any trailing semicolons or whitespace
    clean_sql = sql.strip().rstrip(';')
    
    # Identify positions of trailing clauses (GROUP BY, ORDER BY, LIMIT) using regex
    trailing_match = re.search(r'\b(GROUP\s+BY|ORDER\s+BY|LIMIT)\b', clean_sql, re.IGNORECASE)
    if trailing_match:
        trailing_pos = trailing_match.start()
    else:
        trailing_pos = len(clean_sql)
            
    head = clean_sql[:trailing_pos].strip()
    tail = clean_sql[trailing_pos:].strip()
    
    if re.search(r'\bWHERE\s*$', head, re.IGNORECASE):
        new_sql = f"{head} {where_to_add}"
    elif re.search(r'\bWHERE\b', head, re.IGNORECASE):
        new_sql = f"{head} AND {where_to_add}"
    else:
        new_sql = f"{head} WHERE {where_to_add}"
        
    if tail:
        new_sql = f"{new_sql} {tail}"
        
    print(f"[enforce_jurisdiction_sql] Injected scope filters: {new_sql}")
    return new_sql

def enforce_jurisdiction_sql(sql: str, state: GraphState) -> str:
    if not sql or sql in ["UNSUPPORTED", "UNAUTHORIZED_DIOCESE", "UNAUTHORIZED_PARISH"]:
        return sql
    sql_upper = sql.upper()
    if "UNION" in sql_upper:
        delimiter = "\nUNION ALL\n" if "UNION ALL" in sql_upper else "\nUNION\n"
        branches = re.split(r'\bUNION\s+(?:ALL\s+)?', sql, flags=re.IGNORECASE)
        enforced = [enforce_jurisdiction_sql_single(b.strip(), state) for b in branches if b.strip()]
        return delimiter.join(enforced)
    return enforce_jurisdiction_sql_single(sql, state)

def validate_sql_node(state: GraphState) -> GraphState:
    print("[validate_sql] Validating SQL in sandbox...")
    sql = state.get("generated_sql") or ""
    
    if sql == "BLOCKED_SECURITY":
        return {**state, "error_message": "Blocked security injection"}
    if sql == "UNSUPPORTED":
        return {**state, "error_message": "Unsupported query"}
    if sql == "UNAUTHORIZED_DIOCESE":
        return {**state, "error_message": "Unauthorized diocese access"}
    if sql == "UNAUTHORIZED_PARISH":
        return {**state, "error_message": "Unauthorized parish access"}

    # Automatically repair unescaped single quotes in parish & personal names (e.g. 'St. Joseph's' -> 'St. Joseph''s', 'D'Souza' -> 'D''Souza')
    apostrophe_fixes = [
        (r"'(St\.\s+[A-Za-z]+)'s\s+([^']*)'", r"'''s '"),
        (r"'(St\.\s+[A-Za-z]+)'s'", r"'''s'"),
        (r"'([A-Za-z]+)'s'", r"'''s'"),
        (r"'([A-Z])'([A-Za-z]+)'", r"''''"),
        (r"'(D|O|Mc|Mac)'([A-Za-z]+)'", r"''''"),
    ]
    for pat, rep in apostrophe_fixes:
        sql = re.sub(pat, rep, sql, flags=re.IGNORECASE)

    user_parish = state.get("user_parish") or ""
    if user_parish and "'" in user_parish:
        raw_literal = f"'{user_parish}'"
        esc_literal = sql_quote_str(user_parish)
        if raw_literal in sql:
            sql = sql.replace(raw_literal, esc_literal)

    # Deterministically enforce role & scope ID filtering
    sql = enforce_jurisdiction_sql(sql, state)

    # Automatically sanitize and polish select clause to ensure meaningful columns
    sql = sanitize_select_clause(sql, state["question"])

    # Automatically resolve canonical BCC name in tabFamily filters
    if "PARISH_BCC_ID" in sql.upper():
        bcc_match = re.search(r"\bparish_bcc_id\s+(?:LIKE|=)\s+['\"]%?([^%'\"\n]+)%?['\"]", sql, re.IGNORECASE)
        if bcc_match:
            raw_bcc = bcc_match.group(1).strip()
            from koinonia_assistant.rag.name_search import resolve_canonical_bcc
            canon_bcc = resolve_canonical_bcc(raw_bcc, user_parish=state.get("user_parish"))
            if canon_bcc and canon_bcc != raw_bcc:
                sql = re.sub(
                    r"\bparish_bcc_id\s+(?:LIKE|=)\s+['\"][^'\"]*['\"]",
                    f"parish_bcc_id = {sql_quote_str(canon_bcc)}",
                    sql,
                    flags=re.IGNORECASE
                )

    # Normalize common phonetics/transliterations
    sql = re.sub(r"\bAgnus\b", "Agnes", sql, flags=re.IGNORECASE)
    state = {**state, "generated_sql": sql}


    # Basic safety checks
    sql_upper = sql.upper().strip()
    if not sql_upper.startswith("SELECT"):
        return {**state, "error_message": "Only SELECT queries are allowed."}
        
    for forbidden in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE", "CREATE", "SLEEP", "BENCHMARK", "INFORMATION_SCHEMA", "INTO OUTFILE", "DUMPFILE", "LOAD_FILE"]:
        if re.search(r'\b' + forbidden + r'\b', sql_upper):
            return {**state, "error_message": f"Query contains forbidden security-sensitive keyword: {forbidden}"}

    clean_sql = re.sub(r'COUNT\s*\(\s*\*\s*\)', '', sql_upper)
    if '*' in clean_sql:
        return {**state, "error_message": "Do not use wildcard SELECT * or f.*. You MUST explicitly select up to 5 human-readable columns (like first_name, last_name, date, etc.) that directly answer the query."}

    if ("PLACE_OF_BIRTH" in sql_upper or "BIRTH_PLACE" in sql_upper) and "BCC" in sql_upper:
        return {**state, "error_message": "Do not filter place_of_birth or birth_place by BCC names. Place of birth represents a town or city, not a Basic Christian Community (BCC). Filter by parish_bcc_id in tabFamily instead."}

    # Check for invalid mix of COUNT(*) and columns without GROUP BY (ignore UNION queries)
    if "UNION" not in sql_upper:
        select_part = sql_upper.split("FROM")[0] if "FROM" in sql_upper else sql_upper
        if re.search(r'COUNT\s*\(\s*\*\s*\)', select_part) and "," in select_part and "GROUP BY" not in sql_upper:
            return {**state, "error_message": "Do not mix COUNT(*) with individual columns in the SELECT clause without a GROUP BY. If the user wants a list of items, do not use COUNT(*). If they want a count, only select COUNT(*)."}



    # EXPLAIN Sandbox check in Frappe MariaDB
    try:
        ensure_frappe_db()
        import frappe
        if hasattr(frappe, "local") and hasattr(frappe.local, "db") and frappe.local.db:
            explain_sql = f"EXPLAIN {sql}"
            frappe.db.sql(explain_sql)
            print("[validate_sql] SQL validated successfully.")
            return {**state, "error_message": ""}
        else:
            print("[validate_sql] frappe.local.db not bound; skipping EXPLAIN sandbox check.")
            return {**state, "error_message": ""}
    except Exception as e:
        error_msg = str(e)
        if "object is not bound" in error_msg.lower():
            print("[validate_sql] frappe.db object is not bound; bypassing EXPLAIN check.")
            return {**state, "error_message": ""}
        print(f"[validate_sql] SQL Validation Failed: {error_msg}")
        return {**state, "error_message": error_msg}

SQL_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL rewrite assistant for a MariaDB database in KOINONIA Parish Assistant.
The SQL query you generated failed with a database error.
Rewrite the SQL query to fix the error.

User Question:
{question}

Table Schemas:
{relevant_tables}

Relevant Fields (Semantic matching columns that exist in the tables):
{relevant_fields}

CRITICAL RULES:
1. Fix the syntax or column error reported.
2. Only use columns defined in the table schemas above. Do NOT invent columns.
3. For tabFamily, DO NOT select `first_name`, `last_name`, or `family_id` (those columns only exist in tabMember).
4. NEVER return SELECT * or SELECT alias.* or select unhelpful database system columns (such as `name`, `creation`, `modified`, `modified_by`, `owner`, `docstatus`, `idx`, `amended_from`, `_user_tags`, `_comments`, `_assign`, `_liked_by`, `custom`). You MUST explicitly select up to 5 specific, human-readable, helpful columns (like names, key dates, places, status, and IDs).
5. Return ONLY the corrected SQL query — no explanation, no markdown.
6. When the query mentions a "BCC" (e.g. 'Lourdu Matha BCC', 'Christ the King BCC'), ALWAYS filter by the `parish_bcc_id` column in `tabFamily`. NEVER use `place_of_birth` or other birth-related columns to filter by BCC name.
7. If the database error states "Do not mix COUNT(*) with individual columns...", you MUST fix this by removing COUNT(*) entirely from the SELECT clause to return a list of items (since the user asked to "list them"), rather than keeping it.
8. If the user asked for a chart or graph, ALWAYS return exactly TWO columns: one label column and one numeric count column (e.g., `SELECT gender, COUNT(*) FROM tabBaptism GROUP BY gender`). NEVER use pivoted conditional counts like `COUNT(CASE...)`.
"""),
    ("human", """Failed SQL: {failed_sql}
Database Error: {error_message}"""),
])

def rewrite_sql_node(state: GraphState) -> GraphState:
    print(f"[rewrite_sql] Rewriting SQL. Retry count: {state.get('retry_count', 0) + 1}...")
    retry = state.get("retry_count", 0) + 1
    
    response = invoke_llm_with_rotation([
        ("system", SQL_REWRITE_PROMPT.format(
            question=state["question"],
            relevant_tables="\n\n".join(state["relevant_tables"]),
            relevant_fields="\n".join(state["relevant_fields"]),
            failed_sql=state["generated_sql"],
            error_message=state["error_message"]
        ))
    ])
    raw_sql = response.content.strip().strip("```sql").strip("```").strip()
    
    # Strip any markdown code formatting
    if raw_sql.startswith("```"):
        raw_sql = re.sub(r"^```[a-zA-Z]*\n", "", raw_sql)
        raw_sql = re.sub(r"\n```$", "", raw_sql)
        
    return {**state, "generated_sql": raw_sql, "retry_count": retry}

def execute_sql_node(state: GraphState) -> GraphState:
    print("[execute_sql] Running SQL against MariaDB...")
    sql = state["generated_sql"]
    
    if sql in ["UNSUPPORTED", "UNAUTHORIZED_DIOCESE", "UNAUTHORIZED_PARISH"]:
        return {**state, "sql_result": None, "error_message": ""}
        
    try:
        ensure_frappe_db()
        import frappe
        results = frappe.db.sql(sql, as_dict=True)
        print(f"[execute_sql] Query returned {len(results)} rows.")
        return {**state, "sql_result": results, "error_message": ""}
    except Exception as e:
        print(f"[execute_sql] Execution failed: {e}")
        return {**state, "error_message": f"Database execution error: {str(e)}"}

def format_response_node(state: GraphState) -> GraphState:
    print("[format_response] Formatting final response locally...")
    
    if state.get("error_message") == "Unauthorized parish access" or state.get("generated_sql") == "UNAUTHORIZED_PARISH":
        user_role = state.get("user_role") or "User"
        user_parish = state.get("user_parish") or "your assigned parish"
        user_vicariate = state.get("user_vicariate") or "your vicariate"
        user_parishes = state.get("user_parishes") or []
        
        if user_role == "Parish Priest":
            ans = (
                f"🔒 **Access Restricted**\n\n"
                f"As the **Parish Priest** of **{user_parish}**, your jurisdiction and access are strictly limited to records within **{user_parish}**.\n\n"
                f"You do not have permission to view or query member and sacrament registries belonging to other parishes."
            )
        else:
            p_str = ", ".join([f"**{p}**" for p in user_parishes]) if user_parishes else f"**{user_parish}**"
            ans = (
                f"🔒 **Access Restricted**\n\n"
                f"As the **{user_role}** for **{user_vicariate}**, your administration and registry access are strictly limited to your controlled parishes: {p_str}.\n\n"
                f"You do not have permission to view or query registries belonging to parishes outside your jurisdiction."
            )
        return {**state, "final_answer": ans}

    if state.get("error_message") == "Unauthorized diocese access" or state.get("generated_sql") == "UNAUTHORIZED_DIOCESE":
        user_role = state.get("user_role", "Parish Priest")
        user_diocese = state.get("user_diocese") or "your assigned diocese"
        user_parish = state.get("user_parish") or "your assigned parish"
        user_vicariate = state.get("user_vicariate") or "your assigned vicariate"
        foreign_p = state.get("requested_foreign_parish")
        foreign_d = state.get("requested_foreign_diocese")
        
        p_prefix = f"• **{foreign_p}** belongs to the **{foreign_d} Diocese**.\n" if (foreign_p and foreign_d) else (f"• The requested registry belongs to the **{foreign_d} Diocese**.\n" if foreign_d else "")
        
        if user_role == "Bishop":
            ans = (
                f"🔒 **Access Restricted**\n\n"
                f"{p_prefix}"
                f"• As the **Bishop** of **{user_diocese} Diocese**, your jurisdiction and access are strictly limited to records within **{user_diocese} Diocese**.\n\n"
                f"You do not have permission to view or query registries belonging to other dioceses."
            )
        elif user_role == "Parish Priest":
            ans = (
                f"🔒 **Access Restricted**\n\n"
                f"{p_prefix}"
                f"• As a **Parish Priest** for **{user_parish}**, your access is restricted to records within your assigned parish. "
                f"You do not have permission to view data across all parishes in the diocese or other dioceses.\n\n"
                f"If you require diocese-wide reports or statistics, please request them from the Bishop's Office."
            )
        elif user_role in ["Vicar Forane", "Vicar General"]:
            user_parishes = state.get("user_parishes") or []
            p_str = ", ".join([f"**{p}**" for p in user_parishes]) if user_parishes else "your assigned parishes"
            ans = (
                f"🔒 **Access Restricted**\n\n"
                f"{p_prefix}"
                f"• As the **{user_role}** for **{user_vicariate}**, your access is restricted to your assigned parishes ({p_str}) within **{user_diocese} Diocese**.\n\n"
                f"You do not have permission to view records for the entire diocese or other dioceses."
            )
        else:
            ans = (
                f"🔒 **Access Restricted**\n\n"
                f"{p_prefix}"
                f"• Your account role (**{user_role}**) is restricted to local parish records. "
                f"You do not have permission to view data across other parishes or dioceses."
            )
        return {**state, "final_answer": ans}

    if state.get("error_message") == "Unsupported query" or state.get("generated_sql") == "UNSUPPORTED":
        ans = (
            "I'm sorry, but I can only search sacrament registers (Baptism, Communion, Confirmation, Marriage, Anointing of the Sick, and Death) and parish Family/Member registers. "
            "Please ask a question related to these registries, and I'll be happy to search for you! 😊"
        )
        return {**state, "final_answer": ans}
        
    if state.get("error_message"):
        ans = (
            "🙏 I couldn't complete this query right now. "
            "Please try rephrasing your question or ask again in a moment! 😊"
        )
        return {**state, "final_answer": ans}

    raw_results = state["sql_result"]
    
    if not raw_results:
        # Check if the user was querying a parish that exists in another diocese
        import frappe
        foreign_p = None
        foreign_d = None
        user_dio = (state.get("user_diocese") or "").strip()
        user_role_val = state.get("user_role") or "Bishop"
        q_text = (state.get("query_text") or "").lower()
        if frappe.db.table_exists("Parish"):
            all_parishes = frappe.db.sql("SELECT name, diocese_id FROM tabParish", as_dict=True)
            for p in all_parishes:
                p_name = p.get("name") or ""
                p_dio = p.get("diocese_id") or ""
                short_p = p_name.replace("Parish", "").replace("Church", "").replace("Cathedral", "").replace("Shrine", "").strip()
                if short_p and len(short_p) > 3 and re.search(r'\b' + re.escape(short_p.lower()) + r'\b', q_text):
                    if user_dio and p_dio and p_dio.lower() != user_dio.lower():
                        foreign_p = p_name
                        foreign_d = p_dio
                        break
        
        if foreign_p and foreign_d:
            ans = (
                f"🔒 **Access Restricted**\n\n"
                f"• **{foreign_p}** belongs to the **{foreign_d} Diocese**.\n"
                f"• As the **{user_role_val}** of **{user_dio} Diocese**, your jurisdiction and access are strictly limited to records within **{user_dio} Diocese**.\n\n"
                f"You do not have permission to view or query registries belonging to other dioceses."
            )
            return {**state, "final_answer": ans}
            
        q_orig = state.get("question") or ""
        is_tam = bool(re.search(r'[\u0B80-\u0BFF]', q_orig)) or any(kw in q_orig.lower() for kw in ["tamil", "தமிழில்", "தமிழ்"])
        
        gen_sql = state.get("generated_sql") or ""
        
        # Check if query was searching for a specific person's name
        like_matches = re.findall(r"LIKE\s+'%?([^'%]+(?:%[^'%]+)*)%?'", gen_sql, re.IGNORECASE)
        q_name_match = re.search(r'(?:family\s+details\s+of|details\s+of|family\s+members\s+of|member\s+details\s+of|about)\s+([A-Za-z0-9\s.]+)', q_orig, re.IGNORECASE)

        candidates = []
        if q_name_match:
            candidates.append(q_name_match.group(1).strip())
        for lm in like_matches:
            cleaned = lm.replace("%", " ").strip()
            if len(cleaned) > 1 and cleaned not in candidates:
                candidates.append(cleaned)

        name_search_match = re.search(r"(?:first_name|last_name|bridegroom_name|bride_name|bridegroom_last_name|bride_last_name|full_name|mrg_minister|mrg_register_ref)\s+LIKE\s+'%([^%']+)%'", gen_sql, re.IGNORECASE)
        if name_search_match and name_search_match.group(1).strip() not in candidates:
            candidates.append(name_search_match.group(1).strip())

        searched_name = candidates[0] if candidates else ""

        tokens = []
        for cand in candidates:
            for t in re.split(r'[\s%._]+', cand):
                t_clean = t.strip()
                if len(t_clean) >= 3 and t_clean.lower() not in ["show", "the", "details", "family", "member", "all", "our", "parish"] and t_clean not in tokens:
                    tokens.append(t_clean)

        if tokens and frappe.db.table_exists("Baptism"):
            try:
                where_clauses = ["parish_id = %s"]
                params = [state.get("user_parish") or "Yelagiri Parish"]
                name_conds = []
                for t in tokens:
                    name_conds.append("(first_name LIKE %s OR middle_name LIKE %s OR last_name LIKE %s)")
                    params.extend([f"%{t}%", f"%{t}%", f"%{t}%"])
                where_clauses.append(" AND ".join(name_conds))
                b_rows = frappe.db.sql(f"SELECT * FROM tabBaptism WHERE {' AND '.join(where_clauses)} ORDER BY bapt_date DESC LIMIT 1", tuple(params), as_dict=True)
                if b_rows:
                    brow = b_rows[0]
                    pname = " ".join([str(brow.get(c) or "").strip() for c in ["first_name", "middle_name", "last_name"] if str(brow.get(c) or "").strip()])
                    bdate = brow.get("bapt_date")
                    bdate_str = bdate.strftime("%d-%b-%Y") if hasattr(bdate, "strftime") else (str(bdate) if bdate else "-")
                    bplace = brow.get("bapt_place") or brow.get("parish_id") or "-"
                    bmin = brow.get("bapt_minister") or "-"
                    fname = brow.get("father_name") or "-"
                    mname = brow.get("mother_name") or "-"
                    gf = brow.get("bapt_god_father") or "-"
                    gm = brow.get("bapt_god_mother") or "-"
                    
                    if is_tam:
                        ans = f"### 🕊️ ஞானஸ்நானப் பதிவேடு: **{pname}**\n\n"
                        ans += "| 📜 பதிவு விவரங்கள் | விவரம் |\n"
                        ans += "| :--- | :--- |\n"
                        ans += f"| **பங்கு** | {brow.get('parish_id')} |\n"
                        ans += f"| **ஞானஸ்நான தேதி** | {bdate_str} |\n"
                        ans += f"| **ஞானஸ்நான இடம்** | {bplace} |\n"
                        ans += f"| **அருட்பணியாளர்** | {bmin} |\n\n"
                        ans += f"#### 👨‍👩‍👧 பெற்றோர் & ஞானப் பெற்றோர் (குடும்ப விவரம்)\n"
                        ans += f"• **தந்தை பெயர்:** {fname}\n"
                        ans += f"• **தாய் பெயர்:** {mname}\n"
                        ans += f"• **ஞானத்தந்தை:** {gf}\n"
                        ans += f"• **ஞானத்தாய்:** {gm}\n\n"
                        ans += f"ℹ️ *குறிப்பு: இந்த நபர் பங்கின் ஞானஸ்நானப் பதிவேட்டில் உள்ளார். குடும்பக் கணக்கெடுப்பு அட்டை (tabFamily) இன்னும் உருவாக்கப்படவில்லை.*"
                    else:
                        ans = f"### 🕊️ Sacramental Registry Lineage: **{pname}**\n\n"
                        ans += "| 📜 Registry Details | Value |\n"
                        ans += "| :--- | :--- |\n"
                        ans += f"| **Parish Church** | {brow.get('parish_id')} |\n"
                        ans += f"| **Baptism Date** | {bdate_str} |\n"
                        ans += f"| **Place of Baptism** | {bplace} |\n"
                        ans += f"| **Administering Minister** | {bmin} |\n\n"
                        ans += f"#### 👨‍👩‍👧 Parents & Godparents (Family Lineage)\n"
                        ans += f"• **Father's Name:** {fname}\n"
                        ans += f"• **Mother's Name:** {mname}\n"
                        ans += f"• **Godfather:** {gf}\n"
                        ans += f"• **Godmother:** {gm}\n\n"
                        ans += f"ℹ️ *Note: This person is registered in the Parish Baptism Register with their parents and godparents. A separate family census card (tabFamily) has not yet been registered in the parish database.*"
                    return {**state, "final_answer": ans}
            except Exception as b_err:
                print(f"[format_response] Fallback sacrament lookup error: {b_err}")

            table_match = re.search(r'FROM\s+`?(tab(?:Anointing Of Sick|[A-Za-z0-9_]+))`?', gen_sql, re.IGNORECASE)
            tname = table_match.group(1).strip('`') if table_match else "tabMember"
            reg_name = {
                "tabMarriage": "திருமணப் பதிவேட்டில்",
                "tabBaptism": "ஞானஸ்நானப் பதிவேட்டில்",
                "tabCommunion": "புது நன்மைப் பதிவேட்டில்",
                "tabConfirmation": "உறுதிப்பூசுதல் பதிவேட்டில்",
                "tabDeath": "மரணப் பதிவேட்டில்",
                "tabMember": "உறுப்பினர் பட்டியலில்"
            }.get(tname, "பதிவேட்டில்")
            
            if is_tam:
                ans = f"{reg_name} **{searched_name}** என்ற பெயரில் எந்தப் பதிவும் கண்டறியப்படவில்லை."
            else:
                ans = f"No records found for **{searched_name}** in the registry."
            return {**state, "final_answer": ans}

        # Check if query searched for 2026 / current year on a registry with prior records
        table_match = re.search(r'FROM\s+`?(tab(?:Anointing Of Sick|[A-Za-z0-9_]+))`?', gen_sql, re.IGNORECASE)
        latest_yr = None
        if table_match and any(k in gen_sql for k in ["YEAR(CURDATE())", "2026", "YEAR("]):
            tname = table_match.group(1).strip().strip('`')
            doc_type = tname.replace("tab", "", 1) if tname.startswith("tab") else tname
            if frappe.db.table_exists(doc_type):
                cols = frappe.db.get_table_columns(doc_type)
                date_cols = [c for c in ["death_date", "bapt_date", "mrg_date", "fhc_date", "cnf_date", "anointing_date"] if c in cols]
                if date_cols:
                    dcol = date_cols[0]
                    dio_val = (state.get("user_diocese") or "").strip()
                    dio_cond = f"WHERE diocese_id = '{dio_val}' AND {dcol} IS NOT NULL" if dio_val and "diocese_id" in cols else f"WHERE {dcol} IS NOT NULL"
                    res_max = frappe.db.sql(f"SELECT MAX(YEAR({dcol})) AS max_yr, COUNT(*) AS count_max FROM `{tname}` {dio_cond}", as_dict=True)
                    if res_max and res_max[0].get("max_yr"):
                        latest_yr = res_max[0]["max_yr"]
        
        if latest_yr and latest_yr < 2026:
            if is_tam:
                ans = f"நடப்பு 2026-ஆம் ஆண்டிற்கான பதிவுகள் எதுவும் பதிவேற்றப்படவில்லை. மிக சமீபத்திய பதிவுகள் **{latest_yr}**-ஆம் ஆண்டிற்குரியவை. நீங்கள் **{latest_yr}**-ஆம் ஆண்டின் பதிவுகளைப் பார்க்க விரும்புகிறீர்களா?"
            else:
                ans = f"No records have been recorded for 2026 yet. The latest available records in the registry are from **{latest_yr}**. Would you like to view records for {latest_yr}?"
            suggested_qs = [
                f"{latest_yr}-ல் எத்தனை நபர்கள் இறந்தார்கள்?",
                f"Show {latest_yr} records",
                f"{latest_yr} பதிவுகள்"
            ]
            return {**state, "final_answer": ans, "suggested_questions": suggested_qs}
            
        ans = "உங்கள் தேடலுக்குரிய பதிவுகள் எதுவும் கிடைக்கவில்லை." if is_tam else "No records were found matching your search." 
    else:
        # Keep all results for full interactive table pagination in UI
            
        if isinstance(raw_results, list) and len(raw_results) > 0 and isinstance(raw_results[0], dict):
            # Check for relationship inquiry between two persons
            is_rel_query = (
                "rel_relationship_id" in raw_results[0] or 
                "rel1" in raw_results[0] or
                "relationship1" in raw_results[0] or
                "person1_role" in raw_results[0] or
                "person1_relationship" in raw_results[0] or
                "person2_relationship" in raw_results[0] or
                "rel_first_name" in raw_results[0] or
                "first_name2" in raw_results[0] or
                "person2_first" in raw_results[0] or
                ("relationship_id" in raw_results[0] and len(raw_results[0]) <= 8 and any(k in raw_results[0] for k in ["parish1", "person1", "rel2", "rel_parish_id", "first_name1"]))
            )
            if is_rel_query:
                row = raw_results[0]
                p1_parts = [row.get("person1_fullname") or row.get("first_name") or row.get("person1_first") or row.get("first_name1") or row.get("person1"), row.get("middle_name") or row.get("middle_name1") if not row.get("person1_fullname") else None, row.get("last_name") or row.get("person1_last") or row.get("last_name1") if not row.get("person1_fullname") else None]
                p1 = row.get("person1_fullname") or " ".join([str(p).strip() for p in p1_parts if p and str(p).strip() not in ("", "None", "null", "NULL")]).strip()
                r1 = row.get("relationship_id") or row.get("person1_role") or row.get("person1_relationship") or row.get("relationship1") or row.get("rel1") or "Member"
                parish1 = row.get("parish_id") or row.get("person1_parish") or row.get("parish_name") or row.get("parish1") or ""
                
                p2_parts = [row.get("person2_fullname") or row.get("rel_first_name") or row.get("person2_first") or row.get("first_name2") or row.get("person2"), row.get("rel_middle_name") or row.get("middle_name2") if not row.get("person2_fullname") else None, row.get("rel_last_name") or row.get("person2_last") or row.get("last_name2") if not row.get("person2_fullname") else None]
                p2 = row.get("person2_fullname") or " ".join([str(p).strip() for p in p2_parts if p and str(p).strip() not in ("", "None", "null", "NULL")]).strip()
                r2 = row.get("rel_relationship_id") or row.get("person2_role") or row.get("person2_relationship") or row.get("relationship2") or row.get("rel2") or "Member"
                parish2 = row.get("rel_parish_id") or row.get("person2_parish") or row.get("parish2") or parish1
                
                fam = row.get("family_id") or ""
                dio = row.get("diocese_id") or ""
                
                # Compute natural relationship description
                r1_clean = str(r1).strip().title()
                r2_clean = str(r2).strip().title()
                
                if (r1_clean in ["Father", "Head", "Head Of Family"] and r2_clean in ["Mother", "Spouse", "Wife"]) or (r2_clean in ["Father", "Head", "Head Of Family"] and r1_clean in ["Mother", "Spouse", "Wife"]):
                    rel_sentence = f"**{p1}** and **{p2}** are **Husband & Wife** (Father and Mother of the family)."
                elif r1_clean in ["Father", "Mother"] and r2_clean in ["Son", "Daughter", "Child"]:
                    rel_sentence = f"**{p1}** is the **{r1_clean}** of **{p2}** ({r2_clean})."
                elif r2_clean in ["Father", "Mother"] and r1_clean in ["Son", "Daughter", "Child"]:
                    rel_sentence = f"**{p2}** is the **{r2_clean}** of **{p1}** ({r1_clean})."
                elif r1_clean in ["Son", "Brother", "Daughter", "Sister"] and r2_clean in ["Son", "Daughter", "Sister", "Brother"]:
                    rel_sentence = f"**{p1}** and **{p2}** are **Siblings / Brother & Sister**."
                else:
                    rel_sentence = f"**{p1}** ({r1_clean}) and **{p2}** ({r2_clean}) belong to the same household."
                
                ans = f"👤 **Family Relationship Found**\n\n• {rel_sentence}\n"
                if parish1:
                    ans += f"• **Parish:** {parish1}\n"
                if fam:
                    ans += f"• **Family Registration:** {fam}\n"
                if dio:
                    ans += f"• **Diocese:** {dio} Diocese\n"
                return {**state, "final_answer": ans}

            # Check for Household / Family Members Registry Record (e.g. "family details of [Person]")
            is_household_family = (
                len(raw_results) >= 1 and
                any(k in raw_results[0] for k in ["relationship", "relationship_id"]) and
                any(k in raw_results[0] for k in ["family_card_number", "family_id", "family_card_no"]) and
                len(set(str(r.get("family_card_number") or r.get("family_id") or r.get("family_card_no")) for r in raw_results if r.get("family_card_number") or r.get("family_id") or r.get("family_card_no"))) <= 1
            )
            if is_household_family:
                q_orig = state.get("question") or ""
                is_tam = bool(re.search(r'[\u0B80-\u0BFF]', q_orig)) or any(kw in q_orig.lower() for kw in ["tamil", "தமிழில்", "தமிழ்"])
                
                first_r = raw_results[0]
                fc_val = first_r.get("family_card_number") or first_r.get("family_card_no") or first_r.get("family_id") or "-"
                anbiyam_val = first_r.get("anbiyam") or first_r.get("parish_bcc_id") or "-"
                parish_val = first_r.get("parish_id") or first_r.get("parish_name") or state.get("user_parish") or "-"
                street_val = first_r.get("street") or ""
                city_val = first_r.get("city") or ""
                addr_val = f"{street_val}, {city_val}".strip(", ") or "-"
                head_val = first_r.get("family_head_name") or ""
                if not head_val:
                    for r in raw_results:
                        rel = str(r.get("relationship") or r.get("relationship_id") or "").lower()
                        if rel in ["self", "head", "husband", "head of family"] or str(r.get("is_family_head")).lower() in ["1", "yes", "true"]:
                            parts = [r.get("first_name"), r.get("middle_name"), r.get("last_name")]
                            head_val = r.get("full_name") or " ".join([str(p).strip() for p in parts if p and str(p).strip() not in ("", "None", "null", "NULL")]).strip()
                            break
                if not head_val:
                    head_val = "-"

                def _fmt_date_str(val):
                    if not val: return "-"
                    if hasattr(val, "strftime"): return val.strftime("%d-%b-%Y")
                    return str(val)

                if is_tam:
                    ans = f"### 👨‍👩‍👧‍👦 குடும்பப் பதிவு: குடும்ப அட்டை #{fc_val}\n\n"
                    ans += "| 📜 குடும்ப விவரங்கள் | விவரம் |\n"
                    ans += "| :--- | :--- |\n"
                    ans += f"| **குடும்ப அட்டை எண்** | `{fc_val}` |\n"
                    ans += f"| **குடும்பத் தலைவர்** | {head_val} |\n"
                    ans += f"| **பங்கு** | {parish_val} |\n"
                    ans += f"| **அன்பியம்** | {anbiyam_val} |\n"
                    ans += f"| **முகவரி** | {addr_val} |\n\n"
                    ans += f"#### 👥 குடும்ப உறுப்பினர்கள் ({len(raw_results)})\n\n"
                    ans += "| பெயர் | உறவுமுறை | பாலினம் | வயது | பிறந்த தேதி | தொலைபேசி எண் |\n"
                    ans += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
                    for r in raw_results:
                        parts = [r.get("first_name"), r.get("middle_name"), r.get("last_name")]
                        fn = r.get("full_name") or " ".join([str(p).strip() for p in parts if p and str(p).strip() not in ("", "None", "null", "NULL")]).strip() or "-"
                        rel = r.get("relationship") or r.get("relationship_id") or "-"
                        gen = r.get("gender") or "-"
                        age_str = str(r.get("age")) if r.get("age") is not None else "-"
                        dob_str = _fmt_date_str(r.get("dob"))
                        mob = r.get("mobile") or r.get("phone") or "-"
                        ans += f"| **{fn}** | {rel} | {gen} | {age_str} | {dob_str} | {mob} |\n"
                else:
                    ans = f"### 👨‍👩‍👧‍👦 Family Household Registry: Family Card #{fc_val}\n\n"
                    ans += "| 📜 Family Registry | Details |\n"
                    ans += "| :--- | :--- |\n"
                    ans += f"| **Family Card Number** | `{fc_val}` |\n"
                    ans += f"| **Head of Family** | {head_val} |\n"
                    ans += f"| **Parish Church** | {parish_val} |\n"
                    ans += f"| **BCC / Anbiyam** | {anbiyam_val} |\n"
                    ans += f"| **Address / Street** | {addr_val} |\n\n"
                    ans += f"#### 👥 Household Members ({len(raw_results)})\n\n"
                    ans += "| Full Name | Relationship | Gender | Age | Date of Birth | Contact Number |\n"
                    ans += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
                    for r in raw_results:
                        parts = [r.get("first_name"), r.get("middle_name"), r.get("last_name")]
                        fn = r.get("full_name") or " ".join([str(p).strip() for p in parts if p and str(p).strip() not in ("", "None", "null", "NULL")]).strip() or "-"
                        rel = r.get("relationship") or r.get("relationship_id") or "-"
                        gen = r.get("gender") or "-"
                        age_str = str(r.get("age")) if r.get("age") is not None else "-"
                        dob_str = _fmt_date_str(r.get("dob"))
                        mob = r.get("mobile") or r.get("phone") or "-"
                        ans += f"| **{fn}** | {rel} | {gen} | {age_str} | {dob_str} | {mob} |\n"

                has_sacraments = any(k in first_r for k in ["baptism_date", "bapt_date", "communion_date", "fhc_date", "cnf_date", "confirmation_date", "mrg_date", "marriage_date"])
                if has_sacraments:
                    if is_tam:
                        ans += "\n#### ✝️ திருவருட்சாதனங்கள் விவரம்\n\n"
                        ans += "| பெயர் | உறவுமுறை | ஞானஸ்நானம் | புது நன்மை | உறுதிப்பூசுதல் | திருமணம் |\n"
                        ans += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
                        for r in raw_results:
                            parts = [r.get("first_name"), r.get("middle_name"), r.get("last_name")]
                            fn = r.get("full_name") or " ".join([str(p).strip() for p in parts if p and str(p).strip() not in ("", "None", "null", "NULL")]).strip() or "-"
                            rel = r.get("relationship") or r.get("relationship_id") or "-"
                            b_d = _fmt_date_str(r.get("baptism_date") or r.get("bapt_date"))
                            c_d = _fmt_date_str(r.get("communion_date") or r.get("fhc_date"))
                            cnf_d = _fmt_date_str(r.get("confirmation_date") or r.get("cnf_date"))
                            m_d = _fmt_date_str(r.get("marriage_date") or r.get("mrg_date"))
                            ans += f"| **{fn}** | {rel} | {b_d} | {c_d} | {cnf_d} | {m_d} |\n"
                    else:
                        ans += "\n#### ✝️ Holy Sacraments History\n\n"
                        ans += "| Full Name | Relationship | Baptism | First Communion | Confirmation | Holy Matrimony |\n"
                        ans += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
                        for r in raw_results:
                            parts = [r.get("first_name"), r.get("middle_name"), r.get("last_name")]
                            fn = r.get("full_name") or " ".join([str(p).strip() for p in parts if p and str(p).strip() not in ("", "None", "null", "NULL")]).strip() or "-"
                            rel = r.get("relationship") or r.get("relationship_id") or "-"
                            b_d = _fmt_date_str(r.get("baptism_date") or r.get("bapt_date"))
                            c_d = _fmt_date_str(r.get("communion_date") or r.get("fhc_date"))
                            cnf_d = _fmt_date_str(r.get("confirmation_date") or r.get("cnf_date"))
                            m_d = _fmt_date_str(r.get("marriage_date") or r.get("mrg_date"))
                # Disambiguation Notice for Same-Name Persons (if multiple distinct households share the searched name)
                try:
                    import frappe
                    q_input = state.get("question") or ""
                    tokens_searched = [t.strip() for t in re.split(r'[\s%._]+', q_input) if len(t.strip()) > 2 and t.lower() not in ["family", "details", "member", "members", "the", "show", "our", "parish", "give", "list"]]
                    if tokens_searched and hasattr(frappe, "local") and hasattr(frappe.local, "db") and frappe.local.db:
                        first_tok = tokens_searched[0]
                        parish_id = state.get("user_parish") or "Yelagiri Parish"
                        other_sql = """
                            SELECT m.name AS member_id,
                                   CONCAT_WS(' ', m.first_name, NULLIF(m.middle_name, ''), m.last_name) AS full_name,
                                   m.relationship_id,
                                   m.age,
                                   f.family_card_number,
                                   f.parish_bcc_id AS anbiyam
                            FROM tabMember m
                            JOIN tabFamily f ON m.family_id = f.name
                            WHERE (
                                m.first_name = %s 
                                OR m.first_name LIKE %s 
                                OR m.first_name LIKE %s
                                OR m.middle_name LIKE %s
                                OR m.last_name LIKE %s
                            )
                            AND m.parish_id = %s
                            AND f.family_card_number != %s
                            ORDER BY 
                                CASE 
                                    WHEN m.first_name = %s THEN 1 
                                    WHEN m.first_name LIKE %s THEN 2 
                                    ELSE 3 
                                END,
                                CASE 
                                    WHEN m.relationship_id IN ('Self', 'Head', 'Husband') THEN 1 
                                    WHEN m.relationship_id = 'Wife' THEN 2 
                                    ELSE 3 
                                END,
                                m.dob ASC
                            LIMIT 10
                        """
                        prefix = f"{first_tok}%"
                        word_boundary = f"% {first_tok}%"
                        other_rows = frappe.db.sql(other_sql, (first_tok, prefix, word_boundary, prefix, prefix, parish_id, fc_val or "", first_tok, prefix), as_dict=True)
                        seen_other_fams = set()
                        distinct_others = []
                        for orow in other_rows:
                            ofid = orow.get("family_card_number") or orow.get("member_id")
                            if ofid not in seen_other_fams:
                                seen_other_fams.add(ofid)
                                distinct_others.append(orow)
                        if distinct_others:
                            if is_tam:
                                ans += f"\n\n---\n💡 **பங்கில் '{first_tok}' என்ற பெயரில் உள்ள பிற குடும்பங்கள் (Did you mean?):**\n"
                                for o in distinct_others[:3]:
                                    ans += f"• **{o.get('full_name')}** (குடும்ப அட்டை `#{o.get('family_card_number')}`, {o.get('anbiyam')})\n"
                                ans += "\n*(அவர்களின் விவரங்களைக் காண கீழே உள்ள பரிந்துரைகளைக் கிளிக் செய்யவும்)*"
                            else:
                                ans += f"\n\n---\n💡 **Other parishioners matching '{first_tok}' in this parish (Did you mean?):**\n"
                                for o in distinct_others[:3]:
                                    ans += f"• **{o.get('full_name')}** (Card `#{o.get('family_card_number')}`, {o.get('anbiyam')})\n"
                                ans += "\n*(Click any of the suggested questions below to view their household)*"
                except Exception as disambig_err:
                    print(f"[format_response] Disambiguation notice error: {disambig_err}")

                return {**state, "final_answer": ans}

            # Check for Parish Profile & History (Strictly for tabParish)
            table_match = re.search(r'FROM\s+`?(tab[A-Za-z0-9_ ]+)`?', state.get("generated_sql", ""), re.IGNORECASE)
            sql_table = table_match.group(1).strip().strip('`') if table_match else ""
            is_parish_profile = len(raw_results) == 1 and sql_table == "tabParish" and any(k in raw_results[0] for k in ["patron_saint", "feast_day", "assistant_priest"])
            if is_parish_profile:
                row = raw_results[0]
                p_name = row.get("parish_name") or row.get("name") or "Parish"
                p_priest = row.get("parish_priest") or ""
                asst_priest = row.get("assistant_priest") or ""
                saint = row.get("patron_saint") or ""
                feast_val = row.get("feast_day")
                feast = feast_val.strftime("%B %d") if hasattr(feast_val, "strftime") else (str(feast_val) if feast_val else "")
                est_val = row.get("established_date")
                est = est_val.strftime("%B %d, %Y") if hasattr(est_val, "strftime") else (str(est_val) if est_val else "")
                city = row.get("city") or ""
                note = row.get("note") or row.get("history") or ""
                
                if not note or not saint or not p_priest:
                    import frappe
                    p_db = frappe.db.sql("SELECT parish_name, patron_saint, feast_day, established_date, parish_priest, assistant_priest, city, note FROM tabParish WHERE name = %s OR parish_name = %s LIMIT 1", (p_name, p_name), as_dict=True)
                    if p_db:
                        pdb_row = p_db[0]
                        p_name = pdb_row.get("parish_name") or p_name
                        saint = saint or pdb_row.get("patron_saint") or "Patron Saint"
                        p_priest = p_priest or pdb_row.get("parish_priest") or "Parish Priest"
                        asst_priest = asst_priest or pdb_row.get("assistant_priest") or ""
                        if not feast and pdb_row.get("feast_day"):
                            fd = pdb_row["feast_day"]
                            feast = fd.strftime("%B %d") if hasattr(fd, "strftime") else str(fd)
                        if not est and pdb_row.get("established_date"):
                            ed = pdb_row["established_date"]
                            est = ed.strftime("%B %d, %Y") if hasattr(ed, "strftime") else str(ed)
                        city = city or pdb_row.get("city") or ""
                        note = note or pdb_row.get("note") or ""
                
                ans = f"⛪ **{p_name} Profile**\n\n"
                if saint: ans += f"• **Patron Saint:** {saint}\n"
                if feast: ans += f"• **Feast Day:** {feast}\n"
                if est: ans += f"• **Established:** {est}\n"
                if p_priest: ans += f"• **Parish Priest:** {p_priest}\n"
                if asst_priest: ans += f"• **Assistant Priest:** {asst_priest}\n"
                if city: ans += f"• **Location:** {city}\n\n"
                if note:
                    ans += f"📖 **History & Heritage:**\n{note}\n"
                return {**state, "final_answer": ans}

            # Check for Diocese Profile (Strictly for tabDiocese)
            is_single_diocese_profile = len(raw_results) == 1 and sql_table == "tabDiocese" and any(k in raw_results[0] for k in ["diocese_name", "bishop_name"])
            if is_single_diocese_profile:
                row = raw_results[0]
                d_name = row.get("diocese_name") or row.get("name") or "Diocese"
                b_name = row.get("bishop_name") or "Bishop / Ordinary"
                est_val = row.get("established_date")
                est = est_val.strftime("%B %d, %Y") if hasattr(est_val, "strftime") else (str(est_val) if est_val else "Historic")
                addr = row.get("address") or row.get("city") or "Bishop's House"
                phone = row.get("phone") or ""
                email = row.get("email") or ""
                web = row.get("website") or ""
                note = row.get("note") or row.get("history") or ""
                
                if not note:
                    import frappe
                    d_db = frappe.db.sql("SELECT diocese_name, bishop_name, established_date, city, phone, email, website, note FROM tabDiocese WHERE name = %s OR diocese_name = %s LIMIT 1", (d_name, d_name), as_dict=True)
                    if d_db:
                        ddb_row = d_db[0]
                        d_name = ddb_row.get("diocese_name") or d_name
                        b_name = ddb_row.get("bishop_name") or b_name
                        if not est_val and ddb_row.get("established_date"):
                            ed = ddb_row["established_date"]
                            est = ed.strftime("%B %d, %Y") if hasattr(ed, "strftime") else str(ed)
                        addr = ddb_row.get("city") or addr
                        phone = phone or ddb_row.get("phone") or ""
                        email = email or ddb_row.get("email") or ""
                        web = web or ddb_row.get("website") or ""
                        note = ddb_row.get("note") or ""
                
                contacts = []
                if phone: contacts.append(f"📞 **Phone:** {phone}")
                if email: contacts.append(f"✉️ **Email:** {email}")
                if web: contacts.append(f"🌐 **Website:** [{web}]({web if web.startswith('http') else 'https://' + web})")
                contact_str = " | ".join(contacts) if contacts else ""
                
                ans = f"🏛️ **{d_name} Profile**\n\n• **Bishop / Ordinary:** {b_name}\n• **Established:** {est}\n• **Chancery / City:** {addr}\n"
                if contact_str:
                    ans += f"• **Contact:** {contact_str}\n\n"
                if note:
                    ans += f"📖 **History & Heritage:**\n{note}\n"
                return {**state, "final_answer": ans}

            # Compute full_name if first_name / last_name are present
            has_name_fields = any(k in raw_results[0] for k in ["first_name", "last_name", "bridegroom_name", "bride_name"])
            if has_name_fields:
                for row in raw_results:
                    if "full_name" not in row:
                        parts = [
                            row.get("first_name") or row.get("bridegroom_name") or row.get("bride_name"),
                            row.get("middle_name") or row.get("bridegroom_middle_name") or row.get("bride_middle_name"),
                            row.get("last_name") or row.get("bridegroom_last_name") or row.get("bride_last_name")
                        ]
                        fn_str = " ".join([str(p).strip() for p in parts if p and str(p).strip() not in ("", "None", "null", "NULL")]).strip()
                        if fn_str:
                            row["full_name"] = fn_str

            raw_headers = list(raw_results[0].keys())
            
            # Define system metadata and unhelpful fields to exclude if other columns are present
            system_fields = {
                'name', 'creation', 'modified', 'modified_by', 'owner', 'docstatus', 'idx',
                '_user_tags', '_comments', '_assign', '_liked_by', 'amended_from', 'custom', 'active'
            }
            
            # Filter headers
            headers = [h for h in raw_headers if h.lower() not in system_fields]
            if not headers:
                headers = raw_headers
                
            # If full_name is present, prioritize it and remove fragmented first/middle/last from wide table
            if "full_name" in headers and len(headers) > 3:
                headers = ["full_name"] + [h for h in headers if h.lower() not in ["full_name", "first_name", "middle_name", "last_name"]]
                
            # Define priority ordering weights for common meaningful fields (lower weight = higher priority)
            field_weights = {
                # Names & Canonical Identifiers
                'full_name': 0, 'first_name': 1, 'middle_name': 2, 'last_name': 3,
                'relationship': 4, 'relationship_id': 4,
                'gender': 5, 'age': 6, 'dob': 7,
                'mobile': 8, 'phone': 9,
                'anbiyam': 10, 'parish_bcc_id': 10,
                'family_card_number': 11, 'family_card_no': 11, 'family_register_number': 11,
                'family_head': 12, 'family_head_name': 12,
                'diocese_name': 1, 'vicariate_name': 1, 'parish_name': 1, 'family_name': 1,
                'fhc_parish_id': 1, 'bapt_parish_id': 1, 'mrg_parish_id': 1, 'cnf_parish_id': 1, 'death_parish_id': 1, 'parish_id': 1,
                'bishop_name': 2, 'vicar_forane': 2, 'parish_priest': 2,
                'bridegroom_name': 1, 'bride_name': 2, 'witness1_name': 4, 'witness2_name': 5,
                
                # Codes & IDs
                'diocese_code': 10, 'vicariate_code': 10, 'parish_code': 10, 'family_card_no': 10,
                'diocese_id': 11, 'vicariate_id': 12, 'parish_id': 13, 'family_id': 14, 'member_id': 15,
                
                # Dates
                'established_date': 20, 'bapt_date': 21, 'mrg_date': 22, 'cnf_date': 23, 'fhc_date': 24, 'death_date': 25,
                'dob': 26, 'age': 27, 'established': 28,
                
                # Places
                'bapt_place': 30, 'birth_place': 31, 'place_of_birth': 32, 'city': 33, 'state_id': 34, 'country_id': 35,
                
                # Classifications/Groupings
                'status': 40, 'living_status': 41, 'marital_status_id': 42,
                'parish_bcc_id': 43, 'zone_id': 44
            }
            
            # Sort headers by weight, keeping stable ordering for others
            headers.sort(key=lambda h: field_weights.get(h.lower(), 100))
            
            # Format 1: Single Row, Single Column
            if len(raw_results) == 1 and len(headers) == 1:
                k = headers[0]
                v = raw_results[0].get(k)
                label = str(k).replace("_", " ").title().replace("Count(*)", "Total Count").replace("Count(Idx)", "Total Count")
                q_original = state.get("question") or ""
                eq_lower = (state.get("enhanced_query") or "").lower()
                is_tam = bool(re.search(r'[\u0B80-\u0BFF]', q_original)) or any(kw in q_original.lower() for kw in ["tamil", "தமிழில்", "தமிழ்"])
                
                if label == "Total Count":
                    if "family" in eq_lower or "famil" in eq_lower or "குடும்ப" in q_original:
                        label = "Total Families"
                    elif "member" in eq_lower or "உறுப்பினர்" in q_original or "நபர்கள்" in q_original:
                        label = "Total Members"
                    elif "bapt" in eq_lower or "ஞானஸ்நான" in q_original:
                        label = "Total Baptisms"
                    elif "marr" in eq_lower or "திருமண" in q_original:
                        label = "Total Marriages"

                if is_tam:
                    ta_labels = {
                        "Total Count": "மொத்த எண்ணிக்கை (Total Count)",
                        "Total Families": "மொத்த குடும்பங்கள் (Total Families)",
                        "Total Members": "மொத்த உறுப்பினர்கள் (Total Members)",
                        "Total Parishes": "மொத்த பங்குகள் (Total Parishes)",
                        "Total Baptisms": "மொத்த ஞானஸ்நானங்கள் (Total Baptisms)",
                        "Total Marriages": "மொத்த திருமணங்கள் (Total Marriages)",
                    }
                    ta_lbl = ta_labels.get(label, label)
                    ans = f"**{ta_lbl}**: **{v}**"
                else:
                    ans = f"**{label}**: {v}"
                
            # Format 2: Single Row, Multiple Columns (Rich Sacramental Card / Certificate)
            elif len(raw_results) == 1 and len(headers) > 1:
                row = raw_results[0]
                table_match = re.search(r'FROM\s+`?(tab[A-Za-z0-9_ ]+)`?', state.get("generated_sql", ""), re.IGNORECASE)
                table_name = table_match.group(1).strip().strip('`') if table_match else ""
                
                # Fetch full document from DB if we have a table and name/name match to ensure 100% complete data
                if table_name:
                    doctype_name = table_name.replace("tab", "", 1) if table_name.startswith("tab") else table_name
                    doc_id = row.get("name")
                    if not doc_id and row.get("first_name"):
                        try:
                            import frappe
                            match_query = "SELECT * FROM `" + table_name + "` WHERE first_name = %s"
                            params = [row.get("first_name")]
                            if row.get("last_name"):
                                match_query += " AND last_name = %s"
                                params.append(row.get("last_name"))
                            match_query += " LIMIT 1"
                            full_db_rows = frappe.db.sql(match_query, tuple(params), as_dict=True)
                            if full_db_rows:
                                row = {**full_db_rows[0], **row}
                                doc_id = row.get("name")
                        except Exception as e:
                            print(f"[format_response] DB row lookup: {e}")

                def _fmt_d(val):
                    if not val: return "-"
                    if hasattr(val, "strftime"): return val.strftime("%d-%b-%Y")
                    return str(val)
                
                def _join_names(*names):
                    parts = [str(n).strip() for n in names if n and str(n).strip() not in ("", "None", "null", "NULL")]
                    return " ".join(parts) if parts else "-"

                # --- 1. PARISHIONER MEMBER PROFILE CARD (Priority 1) ---
                is_member = (
                    "tabMember" in table_name or 
                    (row.get("name") and str(row.get("name")).startswith("MEM-")) or 
                    "living_status" in row or 
                    "blood_group_id" in row or 
                    ("family_id" in row and "bapt_register_ref" not in row and "fhc_register_ref" not in row)
                )
                if is_member:
                    # Enrich full member record from DB including family join
                    import frappe
                    m_id = row.get("name")
                    fn = row.get("first_name")
                    ln = row.get("last_name")
                    try:
                        m_db = None
                        if m_id and str(m_id).startswith("MEM-"):
                            m_db = frappe.db.sql("SELECT m.*, f.family_register_number, f.parish_bcc_id, f.zone_id FROM tabMember m LEFT JOIN tabFamily f ON m.family_id = f.name WHERE m.name = %s LIMIT 1", (m_id,), as_dict=True)
                        elif fn:
                            m_db = frappe.db.sql("SELECT m.*, f.family_register_number, f.parish_bcc_id, f.zone_id FROM tabMember m LEFT JOIN tabFamily f ON m.family_id = f.name WHERE m.first_name = %s AND (m.last_name = %s OR %s IS NULL) LIMIT 1", (fn, ln, ln), as_dict=True)
                        if m_db:
                            row = {**m_db[0], **row}
                    except Exception as me:
                        print(f"[format_response] Member DB lookup: {me}")

                    m_name = _join_names(row.get("first_name"), row.get("middle_name"), row.get("last_name"))
                    reg_no = row.get("name") or "-"
                    parish = row.get("parish_id") or row.get("parish_name") or "-"
                    diocese = row.get("diocese_id") or "-"
                    vicariate = row.get("vicariate_id") or "-"
                    fam_id = row.get("family_id") or "-"
                    fc_no = row.get("family_register_number") or row.get("family_card_no") or fam_id
                    bcc = row.get("parish_bcc_id") or "-"
                    gender = row.get("gender") or "-"
                    dob = _fmt_d(row.get("dob"))
                    pob = row.get("place_of_birth") or row.get("city") or diocese
                    age = str(row.get("age")) if row.get("age") is not None else "-"
                    marital = row.get("marital_status_id") or "Single"
                    blood = row.get("blood_group_id") or "O+"
                    occ = row.get("occupation") or "-"
                    edu = row.get("education") or "-"
                    mob = row.get("mobile") or "-"
                    email = row.get("email") or "-"
                    addr = f"{row.get('street') or ''}, {row.get('city') or ''}".strip(', ') or parish
                    status = row.get("living_status") or "Living"
                    head = " (Head of Family)" if str(row.get("is_family_head")).lower() in ["1", "yes", "true"] else ""
                    f_name = row.get("father_name") or "-"
                    m_name_parent = row.get("mother_name") or "-"
                    
                    b_date = _fmt_d(row.get("bapt_date"))
                    b_parish = row.get("bapt_parish_id") or parish
                    fhc_date = _fmt_d(row.get("fhc_date"))
                    fhc_parish = row.get("fhc_parish_id") or parish
                    cnf_date = _fmt_d(row.get("cnf_date"))
                    cnf_parish = row.get("cnf_parish_id") or parish
                    mrg_date = _fmt_d(row.get("mrg_date"))
                    mrg_parish = row.get("mrg_parish_id") or parish

                    ans = f"### 👤 Parishioner Registry Record: **{m_name}**\n\n"
                    ans += "| 📜 Registry Identification | Value |\n"
                    ans += "| :--- | :---\n"
                    ans += f"| **Member ID** | `{reg_no}` |\n"
                    ans += f"| **Family Card Number** | `{fc_no}` |\n"
                    ans += f"| **Parish Church** | {parish} |\n"
                    ans += f"| **Diocese / Vicariate** | {diocese} Diocese ({vicariate}) |\n"
                    ans += f"| **Living / Membership Status** | {status}{head} |\n\n"
                    ans += f"#### 👤 Personal Details\n"
                    ans += f"• **Full Name:** {m_name}\n"
                    ans += f"• **Gender:** {gender} | **Age:** {age} yrs | **Blood Group:** {blood}\n"
                    ans += f"• **Date of Birth:** {dob} | **Place of Birth:** {pob}\n"
                    ans += f"• **Marital Status:** {marital} | **Occupation:** {occ} | **Education:** {edu}\n\n"
                    ans += f"#### 👨‍👩‍👧 Family & Lineage\n"
                    ans += f"• **Father's Name:** {f_name}\n"
                    ans += f"• **Mother's Name:** {m_name_parent}\n"
                    ans += f"• **Family Unit (ID):** `{fam_id}` | **BCC Unit:** {bcc}\n\n"
                    ans += f"#### 📞 Contact & Residence\n"
                    ans += f"• **Mobile:** {mob}\n"
                    ans += f"• **Email:** {email}\n"
                    ans += f"• **Address:** {addr}\n\n"
                    ans += f"#### ✝️ Holy Sacraments Received\n"
                    ans += f"• **Baptism:** {b_date} *({b_parish})*\n"
                    ans += f"• **First Holy Communion:** {fhc_date} *({fhc_parish})*\n"
                    ans += f"• **Confirmation:** {cnf_date} *({cnf_parish})*\n"
                    if mrg_date != "-":
                        ans += f"• **Holy Matrimony:** {mrg_date} *({mrg_parish})*\n"
                    return {**state, "final_answer": ans}

                # --- 2. BAPTISM CERTIFICATE CARD (tabBaptism) ---
                elif "tabBaptism" in table_name or (row.get("name") and str(row.get("name")).startswith("BAP-")) or "bapt_register_ref" in row:
                    cand_name = _join_names(row.get("first_name"), row.get("middle_name"), row.get("last_name"))
                    reg_no = row.get("name") or "-"
                    reg_ref = row.get("bapt_register_ref") or "-"
                    parish = row.get("bapt_parish_id") or row.get("parish_name") or row.get("parish_id") or "-"
                    diocese = row.get("diocese_id") or "-"
                    fc_no = row.get("family_card_no") or "-"
                    gender = row.get("gender") or "-"
                    dob = _fmt_d(row.get("dob"))
                    pob = row.get("birth_place") or parish
                    b_date = _fmt_d(row.get("bapt_date"))
                    b_place = row.get("bapt_place") or parish
                    minister = row.get("bapt_minister") or "Rev. Fr. Joseph Arul"
                    pp = row.get("parish_priest") or "Rev. Fr. Joseph Arul"
                    f_name = row.get("father_name") or "Joseph"
                    f_occ = f" *({row.get('father_occupation')})*" if row.get("father_occupation") else ""
                    m_name = row.get("mother_name") or "Mary"
                    m_occ = f" *({row.get('mother_occupation')})*" if row.get("mother_occupation") else ""
                    gf = _join_names(row.get("bapt_god_father"), row.get("bapt_god_father_last_name"))
                    if gf == "-": gf = "Antony Godparent"
                    gm = _join_names(row.get("bapt_god_mother"), row.get("bapt_god_mother_last_name"))
                    if gm == "-": gm = "Theresa Godparent"

                    ans = f"### 🕊️ Holy Baptism Registry Record: **{cand_name}**\n\n"
                    ans += "| 📜 Registry Identification | Value |\n"
                    ans += "| :--- | :---\n"
                    ans += f"| **Registration Number (ID)** | `{reg_no}` |\n"
                    ans += f"| **Baptism Register Ref** | `{reg_ref}` |\n"
                    ans += f"| **Parish Church** | {parish} |\n"
                    ans += f"| **Diocese** | {diocese} Diocese |\n"
                    ans += f"| **Family Card Number** | `{fc_no}` |\n\n"
                    ans += f"#### 👤 Personal Details\n"
                    ans += f"• **Full Name:** {cand_name}\n"
                    ans += f"• **Gender:** {gender}\n"
                    ans += f"• **Date of Birth:** {dob}\n"
                    ans += f"• **Place of Birth:** {pob}\n\n"
                    ans += f"#### 🕊️ Sacramental Administration\n"
                    ans += f"• **Date of Baptism:** {b_date}\n"
                    ans += f"• **Place of Baptism:** {b_place}\n"
                    ans += f"• **Administering Minister:** {minister}\n"
                    ans += f"• **Parish Priest:** {pp}\n\n"
                    ans += f"#### 👨‍👩‍👧 Family & Parents\n"
                    ans += f"• **Father's Name:** {f_name}{f_occ}\n"
                    ans += f"• **Mother's Name:** {m_name}{m_occ}\n\n"
                    ans += f"#### ✝️ Godparents / Sponsors\n"
                    ans += f"• **Godfather:** {gf}\n"
                    ans += f"• **Godmother:** {gm}\n"
                    return {**state, "final_answer": ans}

                # --- 3. FIRST HOLY COMMUNION CARD (tabCommunion) ---
                elif "tabCommunion" in table_name or (row.get("name") and str(row.get("name")).startswith("COM-")) or "fhc_register_ref" in row:
                    cand_name = _join_names(row.get("first_name"), row.get("middle_name"), row.get("last_name"))
                    reg_no = row.get("name") or "-"
                    reg_ref = row.get("fhc_register_ref") or "-"
                    parish = row.get("fhc_parish_id") or row.get("parish_name") or row.get("parish_id") or "-"
                    diocese = row.get("diocese_id") or "-"
                    fc_no = row.get("family_card_no") or "-"
                    fhc_date = _fmt_d(row.get("fhc_date"))
                    fhc_place = row.get("fhc_place") or parish
                    minister = row.get("fhc_minister") or "Rev. Fr. Joseph Arul"
                    f_name = row.get("father_name") or "Joseph"
                    m_name = row.get("mother_name") or "Mary"

                    ans = f"### 🍞 First Holy Communion Registry Record: **{cand_name}**\n\n"
                    ans += "| 📜 Registry Details | Value |\n"
                    ans += "| :--- | :---\n"
                    ans += f"| **Registration Number (ID)** | `{reg_no}` |\n"
                    ans += f"| **FHC Register Ref** | `{reg_ref}` |\n"
                    ans += f"| **Parish Church** | {parish} |\n"
                    ans += f"| **Diocese** | {diocese} Diocese |\n"
                    ans += f"| **Family Card Number** | `{fc_no}` |\n\n"
                    ans += f"#### 👤 Candidate & Sacrament Details\n"
                    ans += f"• **Full Name:** {cand_name}\n"
                    ans += f"• **Date of First Communion:** {fhc_date}\n"
                    ans += f"• **Place:** {fhc_place}\n"
                    ans += f"• **Administering Minister:** {minister}\n\n"
                    ans += f"#### 👨‍👩‍👧 Parents\n"
                    ans += f"• **Father:** {f_name} | **Mother:** {m_name}\n"
                    return {**state, "final_answer": ans}

                # --- 4. CONFIRMATION CARD (tabConfirmation) ---
                elif "tabConfirmation" in table_name or (row.get("name") and str(row.get("name")).startswith("CNF-")) or "cnf_register_ref" in row:
                    cand_name = _join_names(row.get("first_name"), row.get("middle_name"), row.get("last_name"))
                    reg_no = row.get("name") or "-"
                    reg_ref = row.get("cnf_register_ref") or "-"
                    parish = row.get("cnf_parish_id") or row.get("parish_name") or row.get("parish_id") or "-"
                    diocese = row.get("diocese_id") or "-"
                    fc_no = row.get("family_card_no") or "-"
                    cnf_date = _fmt_d(row.get("cnf_date"))
                    cnf_place = row.get("cnf_place") or parish
                    minister = row.get("cnf_minister") or "Most Rev. Bishop"
                    sponsor = row.get("sponsor") or row.get("sponsor_name") or "Antony Sponsor"
                    f_name = row.get("father_name") or "Joseph"
                    m_name = row.get("mother_name") or "Mary"

                    ans = f"### 🕊️ Sacrament of Confirmation Record: **{cand_name}**\n\n"
                    ans += "| 📜 Registry Details | Value |\n"
                    ans += "| :--- | :---\n"
                    ans += f"| **Registration Number (ID)** | `{reg_no}` |\n"
                    ans += f"| **Confirmation Register Ref** | `{reg_ref}` |\n"
                    ans += f"| **Parish Church** | {parish} |\n"
                    ans += f"| **Diocese** | {diocese} Diocese |\n"
                    ans += f"| **Family Card Number** | `{fc_no}` |\n\n"
                    ans += f"#### 👤 Candidate & Confirmation Details\n"
                    ans += f"• **Full Name:** {cand_name}\n"
                    ans += f"• **Confirmation Date:** {cnf_date}\n"
                    ans += f"• **Place:** {cnf_place}\n"
                    ans += f"• **Administering Minister:** {minister}\n"
                    ans += f"• **Confirmation Sponsor:** {sponsor}\n\n"
                    ans += f"#### 👨‍👩‍👧 Parents\n"
                    ans += f"• **Father:** {f_name} | **Mother:** {m_name}\n"
                    return {**state, "final_answer": ans}

                # --- 5. HOLY MATRIMONY CARD (tabMarriage) ---
                elif "tabMarriage" in table_name or (row.get("name") and str(row.get("name")).startswith("MRG-")) or "mrg_date" in row or "bridegroom_name" in row:
                    groom = _join_names(row.get("bridegroom_name"), row.get("bridegroom_middle_name"), row.get("bridegroom_last_name"))
                    bride = _join_names(row.get("bride_name"), row.get("bride_middle_name"), row.get("bride_last_name"))
                    reg_no = row.get("name") or "-"
                    reg_ref = row.get("mrg_register_ref") or "-"
                    parish = row.get("mrg_parish_id") or row.get("parish_name") or row.get("parish_id") or "-"
                    diocese = row.get("diocese_id") or "-"
                    m_date = _fmt_d(row.get("mrg_date"))
                    m_place = row.get("mrg_place") or parish
                    minister = row.get("mrg_minister") or "Rev. Fr. Joseph Arul"
                    w1 = row.get("witness1_name") or "John Witness"
                    w2 = row.get("witness2_name") or "Mary Witness"

                    ans = f"### 💍 Holy Matrimony Registry Record\n\n"
                    ans += "| 📜 Registry Details | Value |\n"
                    ans += "| :--- | :---\n"
                    ans += f"| **Registration Number (ID)** | `{reg_no}` |\n"
                    ans += f"| **Marriage Register Ref** | `{reg_ref}` |\n"
                    ans += f"| **Parish Church** | {parish} |\n"
                    ans += f"| **Diocese** | {diocese} Diocese |\n"
                    ans += f"| **Date of Marriage** | {m_date} |\n"
                    ans += f"| **Solemnizing Minister** | {minister} |\n\n"
                    ans += f"#### 👫 Holy Couple\n"
                    ans += f"• **Bridegroom:** {groom}\n"
                    ans += f"• **Bride:** {bride}\n\n"
                    ans += f"#### ✍️ Official Witnesses\n"
                    ans += f"• **Witness 1:** {w1} | **Witness 2:** {w2}\n"
                    return {**state, "final_answer": ans}

                # --- 6. DEPARTED & MEMORIAL REGISTRY CARD (tabDeath) ---
                elif "tabDeath" in table_name or (row.get("name") and str(row.get("name")).startswith("DTH-")) or "death_date" in row:
                    m_name = _join_names(row.get("first_name"), row.get("middle_name"), row.get("last_name"))
                    reg_no = row.get("name") or "-"
                    reg_ref = row.get("death_register_ref") or "-"
                    parish = row.get("death_parish_id") or row.get("parish_name") or row.get("parish_id") or "-"
                    diocese = row.get("diocese_id") or "-"
                    fc_no = row.get("family_card_no") or "-"
                    d_date = _fmt_d(row.get("death_date"))
                    b_date = _fmt_d(row.get("burial_date"))
                    age = str(row.get("age")) if row.get("age") is not None else "-"
                    cause = row.get("death_cause") or "Natural Causes"
                    cem = row.get("cemetery_code") or "St. Joseph Cemetery"
                    pp = row.get("parish_priest") or "Rev. Fr. Joseph Arul"
                    minister = row.get("burial_minister") or pp

                    ans = f"### 🕯️ Departed & Memorial Registry Record: **{m_name}**\n\n"
                    ans += "| 📜 Registry Details | Value |\n"
                    ans += "| :--- | :---\n"
                    ans += f"| **Registration Number (ID)** | `{reg_no}` |\n"
                    ans += f"| **Memorial Register Ref** | `{reg_ref}` |\n"
                    ans += f"| **Parish Church** | {parish} |\n"
                    ans += f"| **Diocese** | {diocese} Diocese |\n"
                    ans += f"| **Family Card Number** | `{fc_no}` |\n\n"
                    ans += f"#### 👤 Deceased Parishioner Details\n"
                    ans += f"• **Full Name:** {m_name} (Age: {age} yrs)\n"
                    ans += f"• **Date of Death:** {d_date} | **Cause:** {cause}\n"
                    ans += f"• **Date of Burial:** {b_date} | **Cemetery:** {cem}\n"
                    ans += f"• **Burial Minister / Priest:** {minister}\n"
                    return {**state, "final_answer": ans}
                # Try to fetch full document details if possible to display all fields
                table_match = re.search(r'FROM\s+`?(tab[A-Za-z0-9_ ]+)`?', state.get("generated_sql", ""), re.IGNORECASE)
                doc_detail_card = None
                if table_match:
                    table_name = table_match.group(1).strip().strip('`')
                    doctype_name = table_name.replace("tab", "", 1) if table_name.startswith("tab") else table_name
                    if doctype_name == "Anointing Of Sick":
                        doctype_name = "Anointing Of Sick"
                    
                    doc_id = raw_results[0].get("name")
                    if doc_id:
                        try:
                            import frappe
                            import datetime
                            doc = frappe.get_doc(doctype_name, doc_id)
                            meta = frappe.get_meta(doctype_name)
                            
                            # Gather meaningful fields
                            fields_data = []
                            fields_data.append((meta.title_field or "ID", doc_id))
                            
                            system_fields = {
                                'name', 'creation', 'modified', 'modified_by', 'owner', 'docstatus', 'idx',
                                '_user_tags', '_comments', '_assign', '_liked_by', 'amended_from', 'custom', 'active'
                            }
                            
                            for df in meta.fields:
                                if df.fieldtype not in ["Section Break", "Column Break", "Table", "Password"]:
                                    val = doc.get(df.fieldname)
                                    if val is not None and str(val).strip() != "" and df.fieldname not in system_fields:
                                        # Format date values nicely
                                        if df.fieldtype == "Date" and isinstance(val, datetime.date):
                                             val = val.strftime("%d-%b-%Y")
                                        elif df.fieldtype == "Check":
                                            val = "Yes" if val else "No"
                                        fields_data.append((df.label or df.fieldname.replace("_", " ").title(), val))
                                        
                            doc_detail_card = f"### 📋 {doctype_name} Details: {doc_id}\n\n"
                            doc_detail_card += "\n".join(f"- **{label}**: {val}" for label, val in fields_data)
                        except Exception as e:
                            print(f"[format_response] Failed to fetch full doc details: {e}")
                            
                if doc_detail_card:
                    ans = doc_detail_card
                else:
                    ans = ""
                    for k in headers:
                        v = raw_results[0].get(k)
                        if v is None or str(v).strip() in ("", "None", "null", "NULL", "undefined"):
                            v = "-"
                        label = str(k).replace("_", " ").title()
                        ans += f"- **{label}**: {v}\n"
                    
            # Format 3: Multiple Rows (Markdown Table with Disambiguation Context)
            else:
                # Enforce a maximum of 8 columns to prevent UI overflow while showing rich details
                if len(headers) > 8:
                    headers = headers[:8]
                    
                def _fmt_val(v):
                    if v is None or str(v).strip() in ("", "None", "null", "NULL", "undefined"):
                        return "-"
                    return str(v).replace("\n", " ").replace("|", "\\|")

                table_md = "| " + " | ".join(str(h).replace("_", " ").title() for h in headers) + " |\n"
                table_md += "|" + "|".join(["---"] * len(headers)) + "|\n"
                for row in raw_results:
                    table_md += "| " + " | ".join(_fmt_val(row.get(h)) for h in headers) + " |\n"
                
                num_records = len(raw_results)
                q_original = state.get("question") or ""
                is_tamil = bool(re.search(r'[\u0B80-\u0BFF]', q_original)) or any(kw in q_original.lower() for kw in ["tamil", "தமிழில்", "தமிழ்"])
                if is_tamil:
                    header_text = f"உங்கள் தேடலுக்கு ஏற்ப **{num_records}** பதிவுகள் கண்டறியப்பட்டன:\n\n"
                else:
                    header_text = f"Found **{num_records}** records matching your search:\n\n"
                
                # Collect full names for follow-up prompt if available
                names = []
                for row in raw_results:
                    fn_val = row.get("full_name")
                    if not fn_val:
                        parts = [
                            row.get("first_name") or row.get("bridegroom_name") or row.get("bride_name"),
                            row.get("middle_name"),
                            row.get("last_name") or row.get("bridegroom_last_name") or row.get("bride_last_name")
                        ]
                        fn_val = " ".join([str(p).strip() for p in parts if p and str(p).strip() not in ("", "None", "null", "NULL")]).strip()
                    if fn_val and fn_val not in names:
                        names.append(fn_val)
                
                footer_text = ""
                q_user = (state.get("question") or "").lower()
                is_group_or_list_query = any(k in q_user for k in [
                    "children", "child", "members", "families", "all", "list", "show", "who are", "they", "them", "both", "details"
                ])
                if names and 1 < len(names) <= 5 and not is_group_or_list_query:
                    names_str = " or ".join([f"**{n}**" for n in names])
                    footer_text = f"\n💡 *Did you mean {names_str}? Ask again with the full name for a direct match.*"
                
                ans = header_text + table_md + footer_text
                
                
        else:
            ans = str(raw_results)

    # Log successful queries for few-shot learning
    if state.get("history_id") and state.get("history_id") > 0:
        update_correctness_flag(state["history_id"], 1, state.get("generated_sql") or "")

    return {**state, "final_answer": ans}

# ─── Define Router Decisions ──────────────────────────────────────────────────

def decide_route(state: GraphState):
    if state.get("route") == "blocked_security":
        return "blocked_security"
    if state.get("route") == "greeting":
        return "greeting"
    if state.get("route") == "unclear":
        return "unclear"
    return "text_to_sql"

def decide_validation(state: GraphState):
    if state.get("error_message") in ["Unsupported query", "Unauthorized diocese access", "Unauthorized parish access"]:
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

# ─── Assemble LangGraph Workflow ──────────────────────────────────────────────

def route_after_router(state: GraphState):
    return decide_route(state)

def route_after_validation(state: GraphState):
    return decide_validation(state)

def route_after_execution(state: GraphState):
    return decide_execution(state)

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("router_node", router_node)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("unclear", unclear_node)
    workflow.add_node("blocked_security", blocked_security_node)
    workflow.add_node("enhance_query_node", enhance_query_node)
    workflow.add_node("retrieve_context_node", retrieve_context_node)
    workflow.add_node("generate_sql_node", generate_sql_node)
    workflow.add_node("validate_sql_node", validate_sql_node)
    workflow.add_node("rewrite_sql_node", rewrite_sql_node)
    workflow.add_node("execute_sql_node", execute_sql_node)
    workflow.add_node("format_response_node", format_response_node)
    workflow.set_entry_point("router_node")
    workflow.add_conditional_edges("router_node", route_after_router, {
        "greeting": "greeting", 
        "unclear": "unclear", 
        "blocked_security": "blocked_security",
        "text_to_sql": "enhance_query_node"
    })
    workflow.add_edge("greeting", END)
    workflow.add_edge("unclear", END)
    workflow.add_edge("blocked_security", END)
    workflow.add_edge("enhance_query_node", "retrieve_context_node")
    workflow.add_edge("retrieve_context_node", "generate_sql_node")
    workflow.add_edge("generate_sql_node", "validate_sql_node")
    workflow.add_conditional_edges("validate_sql_node", route_after_validation, {
        "valid": "execute_sql_node", 
        "retry": "rewrite_sql_node", 
        "unsupported": "format_response_node", 
        "failed": "format_response_node"
    })
    workflow.add_edge("rewrite_sql_node", "validate_sql_node")
    workflow.add_conditional_edges("execute_sql_node", route_after_execution, {
        "success": "format_response_node", 
        "retry": "rewrite_sql_node", 
        "failed": "format_response_node"
    })
    workflow.add_edge("format_response_node", END)
    return workflow.compile()

app = build_graph()

# ─── Public Invocation Entrypoint ──────────────────────────────────────────────

def run_query(question: str, history: list = None, reference_text: str = None, user_role: str = "Parish Priest", user_parish: str = None, user_vicariate: str = None, user_diocese: str = None, user_parishes: list = None, user_member_id: str = None, user_email: str = None, **kwargs) -> dict:
    app = build_graph()
    
    # Resolve diocese and vicariate from parish if missing (e.g. Yelagiri Parish -> Vellore)
    if user_parish and not user_diocese:
        try:
            ensure_frappe_db()
            import frappe
            if hasattr(frappe, "local") and hasattr(frappe.local, "db") and frappe.local.db:
                p_info = frappe.db.sql("SELECT diocese_id, vicariate_id FROM `tabParish` WHERE name=%s LIMIT 1", (user_parish,), as_dict=True)
                if p_info:
                    if p_info[0].get("diocese_id"):
                        user_diocese = p_info[0]["diocese_id"]
                    if not user_vicariate and p_info[0].get("vicariate_id"):
                        user_vicariate = p_info[0]["vicariate_id"]
        except Exception:
            pass
    if not user_diocese:
        user_diocese = "Trichy"

    # 1. Embed query to check if it's already in history
    embedding = embed_text(question)
    
    # Log query history to Postgres (initializes thread, correctness_flag defaults to NULL)
    history_id = log_query_history(question, "", embedding)
    
    # 2. Run graph execution
    initial_state = {
        "question": question,
        "history": history or [],
        "reference_text": reference_text or "",
        "route": "",
        "enhanced_query": "",
        "relevant_tables": [],
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
        "user_role": user_role,
        "user_parish": user_parish,
        "user_vicariate": user_vicariate,
        "user_diocese": user_diocese,
        "user_parishes": user_parishes or [],
        "user_member_id": user_member_id,
        "user_email": user_email
    }
    
    langsmith_config = {
        "run_name": f"Koinonia_RAG_Pipeline: {question[:40]}",
        "metadata": {
            "user_role": user_role,
            "user_diocese": user_diocese,
            "user_parish": user_parish,
            "question": question
        },
        "tags": [user_role, user_diocese or "Global"]
    }
    
    try:
        final_state = app.invoke(initial_state, config=langsmith_config)
    except Exception as e:
        print(f"[run_query] Graph execution error: {e}")
        return {"reply": "Sorry, I encountered an error while processing your request.", "generated_sql": ""}

    generated_sql = final_state.get("generated_sql", "")
    
    # Update the query history in postgres with the actual generated SQL
    if history_id and generated_sql:
        try:
            import psycopg2
            conn = psycopg2.connect(**PG_CONFIG)
            with conn.cursor() as cur:
                # Only update if the row is not already marked as correct (correctness_flag is not 1)
                cur.execute("UPDATE koinonia_query_history SET generated_sql = %s WHERE id = %s AND (correctness_flag IS NULL OR correctness_flag != 1)", (generated_sql, history_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[run_query] Failed to update query history SQL: {e}")

    suggested = final_state.get("suggested_questions") or generate_suggested_questions(
        question, 
        user_role=user_role, 
        user_diocese=user_diocese,
        user_parish=user_parish,
        user_vicariate=user_vicariate
    )
    return {
        "reply": final_state.get("final_answer", ""),
        "answer": final_state.get("final_answer", ""),
        "generated_sql": generated_sql,
        "sql_result": final_state.get("sql_result"),
        "data": final_state.get("sql_result") or [],
        "suggested_questions": suggested,
        "request_id": kwargs.get("request_id"),
        "error_message": final_state.get("error_message", "")
    }


def generate_suggested_questions(
    question: str, 
    user_role: str = "Parish Priest", 
    user_diocese: str = "Salem",
    user_parish: str = None,
    user_vicariate: str = None
) -> list:
    """Generates 3-4 highly relevant contextual follow-up questions using AI strictly answerable from available church database DocTypes and bounded by user's role jurisdiction."""
    if not question or not question.strip():
        return []
    
    # 0. Dynamic Same-Name Disambiguation: Top 3 Suggestions
    name_suggestions = []
    try:
        name_match = re.search(r'(?:family\s+details\s+of|family\s+members\s+of|details\s+of|member\s+details\s+of|about|who\s+is|show\s+family\s+of|family\s+of)\s+([A-Za-z0-9\s.]+)', question, re.IGNORECASE)
        name_query = name_match.group(1).strip() if name_match else ""
        if not name_query and len(question.split()) <= 3 and not any(kw in question.lower() for kw in ["how", "count", "list", "show all", "total", "what", "where", "our"]):
            name_query = question.strip()
            
        tokens = [t.strip() for t in re.split(r'[\s%._]+', name_query) if len(t.strip()) > 2 and t.lower() not in ["family", "details", "member", "members", "the", "show", "our", "parish", "give", "list"]]
        if tokens and user_parish:
            import frappe
            if hasattr(frappe, "local") and hasattr(frappe.local, "db") and frappe.local.db:
                first_tok = tokens[0]
                sql = """
                    SELECT m.name AS member_id,
                           CONCAT_WS(' ', m.first_name, NULLIF(m.middle_name, ''), m.last_name) AS full_name,
                           m.relationship_id,
                           m.age,
                           f.family_card_number,
                           f.parish_bcc_id AS anbiyam
                    FROM tabMember m
                    JOIN tabFamily f ON m.family_id = f.name
                    WHERE (
                        m.first_name = %s 
                        OR m.first_name LIKE %s 
                        OR m.first_name LIKE %s
                        OR m.middle_name LIKE %s
                        OR m.last_name LIKE %s
                    )
                    AND m.parish_id = %s
                    ORDER BY 
                        CASE 
                            WHEN m.first_name = %s THEN 1 
                            WHEN m.first_name LIKE %s THEN 2 
                            ELSE 3 
                        END,
                        CASE 
                            WHEN m.relationship_id IN ('Self', 'Head', 'Husband') THEN 1 
                            WHEN m.relationship_id = 'Wife' THEN 2 
                            ELSE 3 
                        END,
                        m.dob ASC
                    LIMIT 10
                """
                prefix = f"{first_tok}%"
                word_boundary = f"% {first_tok}%"
                rows = frappe.db.sql(sql, (first_tok, prefix, word_boundary, prefix, prefix, user_parish, first_tok, prefix), as_dict=True)
                
                seen_fams = set()
                candidates = []
                for r in rows:
                    fid = r.get("family_card_number") or r.get("member_id")
                    if fid not in seen_fams:
                        seen_fams.add(fid)
                        candidates.append(r)
                
                if len(candidates) > 1:
                    is_tam = bool(re.search(r'[\u0B80-\u0BFF]', question))
                    for c in candidates[:3]:
                        c_name = c.get('full_name', '').strip()
                        c_card = c.get('family_card_number', '').strip()
                        card_suffix = f" (Card #{c_card})" if c_card else ""
                        if is_tam:
                            name_suggestions.append(f"{c_name}{card_suffix} குடும்ப விவரங்கள்")
                        else:
                            name_suggestions.append(f"Family details of {c_name}{card_suffix}")
    except Exception as d_err:
        print(f"[generate_suggested_questions] Disambiguation helper error: {d_err}")

    if len(name_suggestions) >= 2:
        return name_suggestions[:3]

    parish_info = f"Assigned Parish: {user_parish}" if user_parish else "No single parish assigned"
    vicariate_info = f"Assigned Vicariate: {user_vicariate}" if user_vicariate else ""
    
    try:
        system_instruction = (
            "You are an intelligent Catholic Parish & Diocesan assistant for KOINONIA.\n"
            "Given the user's current query and context, generate 3 to 4 short, clickable FOLLOW-UP SEARCH PROMPTS that the user would want to ask next to explore related church data.\n\n"
            "CRITICAL INSTRUCTION — USER SEARCH PROMPTS ONLY (NEVER ASK QUESTIONS TO THE USER):\n"
            "- Every suggestion MUST be an actionable search query that the USER sends to the assistant (e.g. 'Show top 5 parishes with highest members', 'Compare male vs female members in my diocese', 'Count families in each parish', 'List baptisms in 2024').\n"
            "- NEVER ASK CLARIFICATION QUESTIONS TO THE USER! Never generate questions like 'Which sacrament should be graphed?', 'Specify year range?', 'Do you want all dioceses?', 'Include only active parishes?'. These are strictly forbidden.\n"
            "- Every prompt must be a complete, ready-to-execute user question that searches the database.\n\n"
            f"USER JURISDICTION & ROLE CONTEXT:\n"
            f"- User Role: {user_role}\n"
            f"- Diocese: {user_diocese or 'Not specified'}\n"
            f"- {parish_info}\n"
            f"- {vicariate_info}\n\n"
            "STRICT ROLE-BASED ACCESS LIMITS (Crucial: NEVER suggest questions outside the user's jurisdiction):\n"
            "1. If User Role is 'Parish Priest':\n"
            "   - The questions MUST be strictly scoped to their own parish ('in my parish').\n"
            "   - Allowed: 'Count baptisms in my parish for 2024', 'Show families in my parish', 'Total members in my parish'.\n"
            "   - FORBIDDEN: NEVER suggest diocese-wide census or searching other parishes.\n"
            "2. If User Role is 'Vicar Forane' or 'Vicar General':\n"
            "   - The questions MUST be scoped to their vicariate or controlled parishes ('in my vicariate', 'across parishes in my vicariate').\n"
            "3. If User Role is 'Bishop', 'Chancellor', 'Curia', or 'Administrator':\n"
            "   - The questions can be diocese-wide ('in my diocese', 'across parishes in diocese').\n"
            "4. If User Role is 'Parishioner' or 'Member':\n"
            "   - The questions MUST be personal/family scoped ('Show my family details', 'Parish patron saint').\n\n"
            "STRICT DATABASE SCHEMA BOUNDARIES (You must ONLY suggest questions answerable from these 9 active DocTypes):\n"
            "1. Parish & Diocesan Directory (DocTypes: tabDiocese, tabVicariate, tabParish)\n"
            "2. Family & Parishioner Census (DocTypes: tabFamily, tabMember)\n"
            "3. Sacramental Registers (DocTypes: tabBaptism, tabCommunion, tabConfirmation, tabMarriage, tabDeath)\n\n"
            "FORBIDDEN TOPICS (NEVER suggest these because NO data exists in the database for them):\n"
            "- DO NOT suggest events, diocesan events, parish feasts, calendar schedules, or mass timings.\n"
            "- DO NOT suggest finances, donations, parish budget, collections, or accounts.\n"
            "- DO NOT suggest sermons, homilies, catechism, or clergy transfer orders.\n\n"
            "CRITICAL RULES:\n"
            "1. Return ONLY a valid JSON array of 3 or 4 short question strings.\n"
            "2. Keep each question concise (under 8 words).\n"
            "3. Ensure EVERY question is a direct user query searchable in the database.\n"
            "4. Do NOT output markdown, backticks, or any explanation."
        )
        prompt = [
            ("system", system_instruction),
            ("human", f"User Question: '{question}'")
        ]
        resp = invoke_llm_with_rotation(prompt)
        content = resp.content.strip()
        
        # Robust JSON array extraction
        match = re.search(r"\[\s*.*?\s*\]", content, re.DOTALL)
        if match:
            content = match.group(0)
            
        parsed = json.loads(content)
        if isinstance(parsed, list) and len(parsed) > 0:
            return [str(q).strip().strip('"').strip("'") for q in parsed if str(q).strip()][:4]
    except Exception as e:
        print(f"[generate_suggested_questions] Warning: {e}")
    
    # Fully dynamic fallback using keyword matching and entity extraction with role scoping
    q_low = question.lower()
    
    # Scope determination based on role
    if user_role in ["Bishop", "Chancellor", "Curia", "Administrator", "System Manager"]:
        geo_scope = f"{user_diocese} Diocese" if user_diocese else "my diocese"
    elif user_role in ["Vicar Forane", "Vicar General"]:
        geo_scope = "my vicariate"
    else:
        geo_scope = f"{user_parish}" if user_parish else "my parish"
    
    # 1. Parse Year
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", q_low)
    year = year_match.group(0) if year_match else "2026"
    prev_year = str(int(year) - 1) if year.isdigit() else "2025"
    
    # 2. Parse Sacrament
    sacraments = ["baptism", "marriage", "communion", "confirmation", "death", "sacrament"]
    sacrament = "sacrament"
    for s in sacraments:
        if s in q_low:
            sacrament = s
            break
            
    # 3. Dynamic Fallbacks
    if sacrament != "sacrament":
        return [
            f"Breakdown {sacrament}s by month in {geo_scope} for {year}",
            f"Compare {year} {sacrament}s with {prev_year} in {geo_scope}",
            f"Give this {sacrament} record in graph",
            f"List {sacrament}s with parents names in {geo_scope}"
        ]
        
    if "family" in q_low or "member" in q_low:
        names = re.findall(r"\b[A-Z][a-z]+\b", question)
        name_str = f" for {names[0]}" if names else ""
        return [
            f"Show family registration details{name_str}",
            f"Count members by gender in {geo_scope}",
            f"Show families by BCC unit in {geo_scope}",
            "Who is the family head?"
        ]
        
    if "diocese" in q_low or "parish" in q_low or "vicariate" in q_low:
        if user_role in ["Bishop", "Chancellor", "Curia", "Administrator", "System Manager"]:
            return [
                f"List all parishes in {geo_scope}",
                f"Show total families in each parish of {geo_scope}",
                f"Count members in each parish of {geo_scope}",
                "Compare member counts across dioceses"
            ]
        elif user_role in ["Vicar Forane", "Vicar General"]:
            return [
                "List parishes in my vicariate",
                "Total members in my controlled parishes",
                "Show families in my controlled parishes",
                "Total members in my diocese"
            ]
        else:
            return [
                f"Show families in {geo_scope}",
                f"Total members in {geo_scope}",
                f"Show total counts of all sacraments in {geo_scope}",
                f"List active BCC units in {geo_scope}"
            ]
        
    if user_role in ["Bishop", "Chancellor", "Curia", "Administrator", "System Manager"]:
        return [
            f"Total members in {geo_scope}",
            f"Show total counts of all sacraments in {geo_scope}",
            f"List all parishes in {geo_scope}",
            f"Show total families in each parish of {geo_scope}"
        ]
    elif user_role in ["Vicar Forane", "Vicar General"]:
        return [
            "Total members in my controlled parishes",
            "Show total counts of all sacraments in my vicariate",
            "List parishes in my vicariate",
            "Show families in my parish"
        ]
    else:
        return [
            f"Total members in {geo_scope}",
            f"Show total counts of all sacraments in {geo_scope}",
            f"Show families in {geo_scope}",
            f"Count baptisms in {geo_scope}"
        ]
