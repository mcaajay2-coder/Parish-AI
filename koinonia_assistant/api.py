
def resolve_user_parish(user_email):
    if not user_email or user_email in ['Administrator', 'admin@example.com']:
        return None
        
    # 1. Direct match on Member.email -> parish_id
    member_parish = frappe.db.get_value('Member', {'email': user_email}, 'parish_id')
    if member_parish:
        return member_parish
        
    # 2. Direct match on Parish.email -> name
    parish_by_email = frappe.db.get_value('Parish', {'email': user_email}, 'name')
    if parish_by_email:
        return parish_by_email
        
    # 3. Match Parish by parish_priest name
    try:
        user_doc = frappe.get_doc('User', user_email)
        full_name = user_doc.full_name or ''
        clean_name = full_name.replace('Fr.', '').replace('Fr', '').replace('Rev.', '').replace('SDB', '').strip()
        if clean_name and len(clean_name) > 2:
            match = frappe.db.sql("""SELECT name FROM `tabParish` WHERE parish_priest LIKE %s LIMIT 1""", (f'%{clean_name}%',), as_dict=True)
            if match:
                return match[0]['name']
    except Exception:
        pass
        
    # 4. Email prefix match on tabParish
    try:
        prefix = user_email.split('@')[0].split('_')[0].replace('priest', '').replace('parish', '').strip()
        if len(prefix) > 2:
            match = frappe.db.sql("""SELECT name FROM `tabParish` WHERE LOWER(name) LIKE %s OR LOWER(parish_name) LIKE %s LIMIT 1""", (f'%{prefix}%', f'%{prefix}%'), as_dict=True)
            if match:
                return match[0]['name']
    except Exception:
        pass
        
    return None

import frappe
from koinonia_assistant.rag.sacrament_normalizer import normalize_sacrament_query
import requests
import json
import os
import psycopg2

def _ensure_langsmith_env():
    """Ensure LangSmith tracing environment variables are loaded in gunicorn worker."""
    if not os.environ.get("LANGCHAIN_API_KEY"):
        key = None
        try:
            key = frappe.conf.get("langchain_api_key")
        except Exception:
            pass
        if not key:
            key = os.environ.get("LANGCHAIN_API_KEY", "")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = key
        os.environ["LANGCHAIN_PROJECT"] = "koinonia_assistant"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

_ensure_langsmith_env()


PG_CONFIG = {
    "host":     os.getenv("PG_HOST",     "postgres-vector"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB",      "parish_vectordb"),
    "user":     os.getenv("PG_USER",    "postgres"),
    "password": os.getenv("PG_PASS",    "password"),
}

@frappe.whitelist()
def transcribe_audio():
    """Transcribes uploaded audio using SarvamAI saaras:v3 model with differentiated errors."""
    has_audio_file = False
    try:
        if hasattr(frappe, "request") and frappe.request and hasattr(frappe.request, "files"):
            has_audio_file = "audio" in frappe.request.files
    except Exception:
        pass
    if has_audio_file:
        audio_file = frappe.request.files["audio"]
        if audio_file and audio_file.filename:
            print(f"[Koinonia STT] Transcribing voice input: {audio_file.filename} ({audio_file.mimetype})...")
            sarvam_key = frappe.conf.get("sarvam_api_key")
            if not sarvam_key:
                return {"error": "SarvamAI API Key is missing from site configuration."}
            
            url = "https://api.sarvam.ai/speech-to-text"
            headers = {"api-subscription-key": sarvam_key}
            audio_bytes = audio_file.read() if hasattr(audio_file, "read") else audio_file.stream.read()
            if not audio_bytes or len(audio_bytes) < 100:
                return {"error": "No speech was detected. Please try again."}

            files = {"file": (audio_file.filename or "voice_input.webm", audio_bytes, audio_file.mimetype or "audio/webm")}
            data = {"model": "saaras:v3", "mode": "codemix", "language_code": "unknown"}
            
            try:
                response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
                response.raise_for_status()
                res_json = response.json()
                transcript = (res_json.get("transcript") or "").strip()
                print(f"[Koinonia STT] Transcript received: '{transcript}'")
                if not transcript:
                    return {"error": "No speech was detected. Please try again."}

                try:
                    from koinonia_assistant.rag.tamil_utils import is_tamil, normalize_tamil_query
                    if is_tamil(transcript):
                        transcript = normalize_tamil_query(transcript)
                except Exception as te:
                    print(f"[Koinonia STT] Tamil norm error: {te}")

                try:
                    norm_transcript, _ = normalize_sacrament_query(transcript)
                except Exception:
                    norm_transcript = transcript
                return {"transcript": norm_transcript, "raw_transcript": transcript}
            except requests.exceptions.RequestException as e:
                print(f"[Koinonia STT] Upload / Network Error: {e}")
                return {"error": "Audio upload failed. Check your connection."}
            except Exception as e:
                print(f"[Koinonia STT] Transcription Error: {e}")
                return {"error": "Could not understand the recording."}
    return {"error": "No audio file provided."}

@frappe.whitelist()
def process_message(text=None, query_text=None, message=None, history=None, reference_text=None, workspace_id=None, **kwargs):
    """Processes sacrament and parish questions using the LangGraph RAG pipeline."""
    resolved_query = query_text or text or message or kwargs.get("message")
    if not resolved_query and hasattr(frappe, "form_dict") and frappe.form_dict:
        resolved_query = frappe.form_dict.get("query_text") or frappe.form_dict.get("text") or frappe.form_dict.get("message")
    query_text = resolved_query
    
    # 1. Handle Audio input if uploaded via multipart/form-data
    has_audio_file = False
    try:
        if hasattr(frappe, "request") and frappe.request and hasattr(frappe.request, "files"):
            has_audio_file = "audio" in frappe.request.files
    except Exception:
        pass
    if has_audio_file:
        audio_file = frappe.request.files["audio"]
        if audio_file and audio_file.filename:
            print("[Koinonia Chat] Received audio upload. Transcribing...")
            sarvam_key = frappe.conf.get("sarvam_api_key")
            if not sarvam_key:
                return {"error": "SarvamAI API Key is missing from site configuration."}
            
            url = "https://api.sarvam.ai/speech-to-text"
            headers = {"api-subscription-key": sarvam_key}
            files = {"file": (audio_file.filename, audio_file.stream, audio_file.mimetype)}
            data = {"model": "saaras:v3", "mode": "codemix", "language_code": "unknown"}
            
            try:
                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                res_json = response.json()
                query_text = res_json.get("transcript")
                print(f"[Koinonia Chat] Audio transcribed to: '{query_text}'")
            except Exception as e:
                print(f"[Koinonia Chat] Audio transcription error: {e}")
                return {"error": f"Failed to transcribe audio: {str(e)}"}

    if not query_text:
        return {"error": "No text or audio query provided."}

    # Preserve original_query as immutable source of truth (Sections 24–27, 48)
    original_query = query_text.strip()
    raw_stt_text = original_query
    from koinonia_assistant.rag.tamil_utils import is_tamil, detect_query_language
    detected_lang = detect_query_language(original_query)

    # Only run English typo normalization when input is pure English (NEVER on Tamil input)
    if not is_tamil(original_query):
        try:
            normalized_query, norm_info = normalize_sacrament_query(original_query)
        except Exception as norm_err:
            print(f"[Sacrament Normalizer] Warning in process_message: {norm_err}")

    # Parse history if provided
    parsed_history = []
    if history:
        try:
            parsed_history = json.loads(history)
        except Exception as he:
            print(f"[Koinonia Chat] Warning: Failed to parse history: {he}")

    # 2. Resolve Role-Based Jurisdiction Boundaries (Sections 50–53)
    user_email = frappe.session.user
    user_roles = frappe.get_roles(user_email)
    
    if "Bishop" in user_roles or user_email in ["Administrator", "admin@example.com"]:
        user_role = "Bishop"
        user_parish = None
    elif "Parish Priest" in user_roles:
        user_role = "Parish Priest"
        user_parish = resolve_user_parish(user_email)
    else:
        user_role = "Parishioner"
        user_parish = resolve_user_parish(user_email)

    _ensure_langsmith_env()
    import uuid
    req_id = kwargs.get("request_id") or (frappe.form_dict.get("request_id") if hasattr(frappe, "form_dict") and frappe.form_dict else None) or str(uuid.uuid4())
    print(f"[BACKEND API] REQUEST RECEIVED | request_id={req_id} | lang={detected_lang} | User={user_email} | Role={user_role} | Parish={user_parish} | original_query='{original_query}'")

    # 3. Invoke LangGraph RAG pipeline with the EXACT original_query
    try:
        from koinonia_assistant.rag.rag_engine import run_query
        res = run_query(
            original_query,
            history=parsed_history,
            user_role=user_role,
            user_parish=user_parish,
            request_id=req_id,
            user_id=user_email,
        )

        if isinstance(res, str):
            res = {
                "request_id": req_id,
                "reply": res,
                "answer": res,
                "generated_sql": "",
                "data": [],
                "disambiguation": None,
                "suggested_questions": [],
                "query_id": -1
            }

        reply = res.get("reply", "")
        generated_sql = res.get("generated_sql", "")
        data_rows = res.get("data") or []
        disambig = res.get("disambiguation")
        suggested_qs = res.get("suggested_questions") or []
        query_id = res.get("query_id", -1)

        if query_id == -1:
            try:
                conn = psycopg2.connect(**PG_CONFIG)
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT id FROM koinonia_query_history WHERE user_question = %s ORDER BY id DESC LIMIT 1;",
                            (query_text,)
                        )
                        row = cur.fetchone()
                        if row:
                            query_id = row[0]
                finally:
                    conn.close()
            except Exception:
                pass

        is_direct = reply.startswith("?") or reply.startswith("??") or reply.startswith("Multiple records found")

        return {
            "request_id": res.get("request_id", req_id),
            "trace_id": res.get("trace_id", f"lg-trace-{req_id[:12]}"),
            "original_query": res.get("original_query", original_query),
            "detected_language": res.get("detected_language", detected_lang),
            "normalized_query": res.get("normalized_query", ""),
            "canonical_terms": res.get("canonical_terms", {}),
            "final_response_language": res.get("final_response_language", "ta" if detected_lang == "ta" else "en"),
            "intent": res.get("intent", "GENERAL_DATABASE_QUERY"),
            "speed_tier": res.get("speed_tier", "FAST"),
            "status": res.get("status", "success"),
            "authorization_context": res.get("authorization_context"),
            "authorization_check": res.get("authorization_check"),
            "authorization_result": res.get("authorization_result"),
            "sql_validation_result": res.get("sql_validation_result"),
            "authorized_record_count": res.get("authorized_record_count", 0),
            "scope_type": res.get("scope_type"),
            "scope_id": res.get("scope_id"),
            "scope_name": res.get("scope_name"),
            "user_message": original_query,
            "reply": reply,
            "answer": reply,
            "generated_sql": "" if is_direct else generated_sql,
            "data": data_rows,
            "chart": res.get("chart"),
            "disambiguation": disambig,
            "suggested_questions": suggested_qs,
            "analysis_type": res.get("analysis_type", "operational"),
            "historical_period": res.get("historical_period"),
            "forecast_period": res.get("forecast_period"),
            "method": res.get("method"),
            "processing_time": res.get("processing_time", 0.0),
            "query_id": query_id
        }

    except Exception as e:
        print(f"[Koinonia Chat] Error (request_id={req_id}): {e}")
        import traceback
        traceback.print_exc()
        return {
            "request_id": req_id,
            "status": "error",
            "user_message": query_text,
            "error": "I couldn't complete the database request. Please try again."
        }

@frappe.whitelist()
def log_query_feedback(query_id=None, is_correct=None):
    """Updates correctness flag for human feedback in koinonia_query_history."""
    if query_id is None or is_correct is None:
        frappe.throw("query_id and is_correct are required.")

    try:
        query_id = int(query_id)
        flag = 1 if str(is_correct).lower() == "true" else 0

        conn = psycopg2.connect(**PG_CONFIG)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE koinonia_query_history SET correctness_flag = %s WHERE id = %s;",
                (flag, query_id)
            )
        conn.commit()
        conn.close()

        print(f"[Koinonia Feedback] Query {query_id} marked as {flag}")
        return {"success": True, "query_id": query_id, "correctness_flag": flag}

    except Exception as e:
        print(f"[Koinonia Feedback] Error: {e}")
        return {"success": False, "error": str(e)}

# ─── pgvector Hook Triggers ───────────────────────────────────────────────────

def sync_doctype_schema(doc, method=None):
    """Real-time sync of table schemas and fields to pgvector, restricted to custom sacrament DocTypes."""
    custom_doctypes = {
        "Family", "Member", "Baptism", "Communion", "Confirmation", "Marriage", 
        "Anointing Of Sick", "Death"
    }
    if doc.name not in custom_doctypes:
        return

    try:
        table_name = f"tab{doc.name}"
        print(f"[Koinonia Hook] Syncing schema for {doc.name} (table: {table_name}) to pgvector...")
        
        columns = frappe.db.sql("""
            SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COLUMN_KEY
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION;
        """, (frappe.conf.db_name, table_name), as_dict=True)
        
        if not columns:
            return

        lines = [
            f"Table Name: `{table_name}`",
            f"Entity/Concept: {doc.name}",
            "Columns:"
        ]
        for col in columns:
            nullable_str = "nullable" if col["IS_NULLABLE"] == "YES" else "NOT NULL"
            key_str      = f" [{col['COLUMN_KEY']}]" if col["COLUMN_KEY"] else ""
            lines.append(f"  - `{col['COLUMN_NAME']}` ({col['COLUMN_TYPE']}, {nullable_str}){key_str}")
        description = "\n".join(lines)

        from koinonia_assistant.rag.ingest import embed_text, build_field_description, upsert_pg_field
        embedding = embed_text(description)

        conn = psycopg2.connect(**PG_CONFIG)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO koinonia_table_schemas (table_name, schema_ddl, embedding)
                VALUES (%s, %s, %s::vector)
                ON CONFLICT (table_name) DO UPDATE
                    SET schema_ddl = EXCLUDED.schema_ddl,
                        embedding  = EXCLUDED.embedding;
            """, (table_name, description, embedding))
        conn.commit()
        conn.close()
        print(f"[Koinonia Hook] Successfully synced table schema for {table_name} to pgvector.")

        # Sync field-level schemas in real-time
        print(f"[Koinonia Hook] Syncing fields for {table_name} to pgvector...")
        for col in columns:
            col_name = col["COLUMN_NAME"]
            col_type = col["COLUMN_TYPE"]
            
            # Skip standard meta fields
            if col_name in ["name", "owner", "creation", "modified", "modified_by", "docstatus", "idx", "_user_tags", "_comments", "_assign", "_liked_by"]:
                continue
                
            label, desc = build_field_description(table_name, col_name, col_type)
            emb = embed_text(desc)
            upsert_pg_field(table_name, col_name, col_type, label, desc, emb)
        print(f"[Koinonia Hook] Successfully synced all fields for {table_name} to pgvector.")

    except Exception as e:
        print(f"[Koinonia Hook] Error syncing schema: {e}")

def delete_doctype_schema(doc, method=None):
    """Deletes custom table and field schemas from pgvector when custom DocType is deleted."""
    custom_doctypes = {
        "Family", "Member", "Baptism", "Communion", "Confirmation", "Marriage", 
        "Anointing Of Sick", "Death"
    }
    if doc.name not in custom_doctypes:
        return

    try:
        table_name = f"tab{doc.name}"
        print(f"[Koinonia Hook] Deleting schema/fields for {doc.name} (table: {table_name}) from pgvector...")
        
        conn = psycopg2.connect(**PG_CONFIG)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM koinonia_table_schemas WHERE table_name = %s;", (table_name,))
            cur.execute("DELETE FROM koinonia_field_schemas WHERE table_name = %s;", (table_name,))
        conn.commit()
        conn.close()
        print(f"[Koinonia Hook] Successfully deleted schema/fields for {table_name} from pgvector.")

    except Exception as e:
        print(f"[Koinonia Hook] Error deleting schema: {e}")


@frappe.whitelist()
def save_user_chat_threads(threads_json=None):
    """Saves user chat history threads in Frappe cache/Redis and persistent database defaults."""
    if not threads_json:
        return {"status": "ignored"}
    try:
        user = frappe.session.user
        frappe.cache.hset("koinonia_chat_threads", user, threads_json)
        try:
            frappe.defaults.set_user_default("koinonia_chat_threads", threads_json, user)
        except Exception as de:
            print(f"[Koinonia Threads] User default persist warning: {de}")
        return {"status": "success"}
    except Exception as e:
        print(f"[Koinonia Threads] Error saving threads: {e}")
        return {"status": "error", "message": str(e)}

@frappe.whitelist()
def get_user_chat_threads():
    """Retrieves saved chat history threads for current user (cache + database fallback)."""
    try:
        user = frappe.session.user
        data = frappe.cache.hget("koinonia_chat_threads", user)
        if not data:
            data = frappe.defaults.get_user_default("koinonia_chat_threads", user)
        return {"status": "success", "threads_json": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@frappe.whitelist()
def correct_spelling(text=None):
    """Corrects typos and church-specific terms in questions."""
    if not text or not str(text).strip():
        return {"corrected": text}
    try:
        from koinonia_assistant.rag.rag_engine import llm
        prompt = (
            "Correct any spelling mistakes, typos, or transliteration errors in this "
            "Catholic church/parish question. Maintain the original language (English or Tanglish) "
            "and meaning. Return ONLY the corrected sentence, nothing else.\n\n"
            f"Sentence: {text}"
        )
        res = llm.invoke([("user", prompt)])
        corrected = res.content.strip() if hasattr(res, "content") else str(res)
        return {"corrected": corrected.strip('"').strip("'")}
    except Exception as e:
        print(f"[Koinonia Spelling] Error: {e}")
        return {"corrected": text}

@frappe.whitelist()
def translate_to_tamil(text=None):
    """Translates response text to fluent, polite Tamil."""
    if not text and hasattr(frappe, "request") and frappe.request:
        try:
            data = json.loads(frappe.request.data or "{}")
            text = data.get("text")
        except Exception:
            pass
    if not text or not str(text).strip():
        return {"success": False, "error": "No text provided"}
    try:
        from koinonia_assistant.rag.rag_engine import llm
        prompt = (
            "Translate the following response into natural, polite Tamil suitable for a "
            "Catholic parish/diocese record context. Preserve all numbers, markdown tables, "
            "and formatting.\n\n"
            f"English Text:\n{text}\n\nTamil Translation:"
        )
        res = llm.invoke([("user", prompt)])
        tamil = res.content.strip() if hasattr(res, "content") else str(res)
        return {"success": True, "tamil_text": tamil}
    except Exception as e:
        print(f"[Koinonia Translate] Error: {e}")
        return {"success": False, "error": str(e)}

@frappe.whitelist()
def text_to_speech(text=None):
    """Generates audio speech using Sarvam AI Neural TTS."""
    if not text and hasattr(frappe, "request") and frappe.request:
        try:
            data = json.loads(frappe.request.data or "{}")
            text = data.get("text")
        except Exception:
            pass
    if not text or not str(text).strip():
        return {"error": "No text provided"}

    sarvam_key = frappe.conf.get("sarvam_api_key")
    if not sarvam_key:
        return {"error": "Sarvam API Key is missing from site configuration."}

    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": sarvam_key,
        "Content-Type": "application/json"
    }
    is_tamil = any("஀" <= c <= "௿" for c in text)
    target_lang = "ta-IN" if is_tamil else "en-IN"

    clean_text = text.replace("*", "").replace("#", "").strip()[:1350]
    payload = {
        "inputs": [clean_text],
        "target_language_code": target_lang,
        "speaker": "priya",
        "model": "bulbul:v3"
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        res_data = response.json()
        audios = res_data.get("audios", [])
        if audios:
            return {"audio_base64": audios[0]}
        return {"error": "No audio returned from TTS engine."}
    except Exception as e:
        print(f"[Koinonia TTS] Error: {e}")
        return {"error": f"TTS generation error: {str(e)}"}
