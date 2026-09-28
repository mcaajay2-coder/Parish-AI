import os
import sys
import time
import psycopg2
import torch
from transformers import AutoTokenizer, AutoModel
import frappe

# BGE-M3 Embedding Model Constants
BGE_MODEL_NAME = "BAAI/bge-m3"
_tokenizer = None
_model = None

# pgvector Connection Configuration
PG_CONFIG = {
    "host":     os.getenv("PG_HOST",     "postgres-vector"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB",      "parish_vectordb"),
    "user":     os.getenv("PG_USER",    "postgres"),
    "password": os.getenv("PG_PASS",    "password"),
}

TARGET_TABLES = [
    "tabFamily",
    "tabMember",
    "tabBaptism",
    "tabCommunion",
    "tabConfirmation",
    "tabMarriage",
    "tabAnointing Of Sick",
    "tabDeath",
    "tabParish",
    "tabDiocese",
    "tabVicariate",
    "tabSub Station"
]

META_COLS = {
    "name", "owner", "creation", "modified", "modified_by",
    "docstatus", "idx", "_user_tags", "_comments", "_assign", "_liked_by"
}

def load_bge_model():
    global _tokenizer, _model
    if _tokenizer is None:
        print("[BGE-M3] Loading model in koinonia_assistant...")
        _tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
        _model = AutoModel.from_pretrained(BGE_MODEL_NAME)
        _model.eval()
        print("[BGE-M3] Model loaded successfully.")

def embed_text(text: str) -> list[float]:
    load_bge_model()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = _model(**inputs)
    return output.last_hidden_state.mean(dim=1).squeeze().tolist()

def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    load_bge_model()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = _tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            output = _model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(output.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        all_embeddings.extend(mean_pooled.tolist())
    return all_embeddings

def init_pg_tables():
    """Create pgvector schema and history tables if not exist, and remove accidental internal tables."""
    print("[Postgres] Initializing koinonia pgvector tables and cleaning up...")
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS koinonia_table_schemas (
                    table_name VARCHAR(100) PRIMARY KEY,
                    schema_ddl TEXT NOT NULL,
                    embedding vector(1024) NOT NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS koinonia_field_schemas (
                    id SERIAL PRIMARY KEY,
                    table_name VARCHAR(100) NOT NULL,
                    field_name VARCHAR(100) NOT NULL,
                    field_type VARCHAR(50) NOT NULL,
                    field_label VARCHAR(100),
                    description TEXT NOT NULL,
                    embedding vector(1024) NOT NULL,
                    UNIQUE(table_name, field_name)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS koinonia_query_history (
                    id SERIAL PRIMARY KEY,
                    user_question TEXT NOT NULL,
                    generated_sql TEXT NOT NULL,
                    embedding vector(1024) NOT NULL,
                    correctness_flag INT DEFAULT NULL
                );
            """)
            cur.execute("""
                DELETE FROM koinonia_table_schemas
                WHERE table_name IN ('koinonia_field_schemas', 'koinonia_query_history', 'koinonia_table_schemas');
            """)
        conn.commit()
    finally:
        conn.close()

def get_docfield_metadata() -> dict[str, dict[str, dict]]:
    """Fetches docfield labels, descriptions, and options from tabDocField."""
    rows = frappe.db.sql("""
        SELECT parent, fieldname, label, fieldtype, options, description
        FROM `tabDocField`
    """, as_dict=True)
    res = {}
    for r in rows:
        p = r["parent"]
        tname = f"tab{p}"
        if tname not in res:
            res[tname] = {}
        res[tname][r["fieldname"]] = r
    return res

def get_mariadb_table_metadata() -> dict[str, list[dict]]:
    """Reads schema columns from information_schema for our 12 custom tables."""
    db_name = frappe.conf.db_name
    metadata = {}
    
    for table in TARGET_TABLES:
        exists = frappe.db.sql(f"SHOW TABLES LIKE '{table}'")
        if not exists:
            print(f"[Warning] Table {table} does not exist in MariaDB database {db_name}!")
            continue
        rows = frappe.db.sql("""
            SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COLUMN_KEY
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION;
        """, (db_name, table), as_dict=True)
        
        metadata[table] = []
        for r in rows:
            metadata[table].append({
                "column":   r["COLUMN_NAME"],
                "type":     r["COLUMN_TYPE"],
                "nullable": r["IS_NULLABLE"],
                "default":  r["COLUMN_DEFAULT"],
                "key":      r["COLUMN_KEY"]
            })
            
    return metadata

def build_table_description(table_name: str, columns: list[dict], docfield_meta: dict = None) -> str:
    clean_name = table_name[3:] if table_name.startswith("tab") else table_name
    lines = [
        f"Table Name: `{table_name}`",
        f"Entity/Concept: {clean_name}",
        "Columns:"
    ]
    for col in columns:
        cname = col["column"]
        nullable_str = "nullable" if col["nullable"] == "YES" else "NOT NULL"
        key_str      = f" [{col['key']}]" if col.get("key") else ""
        label_str    = ""
        if docfield_meta and cname in docfield_meta:
            lbl = docfield_meta[cname].get("label")
            if lbl:
                label_str = f" - Label: {lbl}"
        lines.append(f"  - `{cname}` ({col['type']}, {nullable_str}){key_str}{label_str}")
    return "\n".join(lines)

def build_field_description(table_name: str, field_name: str, field_type: str, df_info: dict = None) -> tuple[str, str]:
    table_concept = table_name[3:] if table_name.startswith("tab") else table_name
    
    if df_info and df_info.get("label"):
        field_label = df_info["label"].strip()
    else:
        field_label = field_name.replace("_", " ").title()
        
    desc = f"Table: `{table_name}` ({table_concept}), Field: `{field_name}` (Type: {field_type}), Label: {field_label}."
    
    extra = []
    fn_lower = field_name.lower()
    
    # Check Frappe DocField description or options
    if df_info:
        if df_info.get("description"):
            extra.append(f"Field Note: {df_info['description'].strip()}")
        if df_info.get("options") and df_info.get("fieldtype") in ("Select", "Link"):
            opts = df_info["options"].replace("\n", ", ").strip()
            extra.append(f"Options / References: {opts}.")
            
    # Domain-specific semantics
    if "god_parent" in fn_lower or "godparent" in fn_lower or "god_father" in fn_lower or "god_mother" in fn_lower or "sponsor" in fn_lower:
        extra.append("This field represents the godfather, godmother, sponsor, or witness details for a baptism or confirmation sacrament event.")
    elif "minister" in fn_lower or "priest" in fn_lower:
        extra.append("This field represents the priest, pastor, or minister officiating, solemnizing, or conducting the sacrament, ceremony, burial, or anointing.")
    elif "family_card" in fn_lower or "family_id" in fn_lower or "reference" in fn_lower or "family_register" in fn_lower:
        extra.append("This field links the record to their family register number, family card number, or family file name.")
    elif "fhc" in fn_lower or "communion" in fn_lower:
        extra.append("This field relates to First Holy Communion (FHC) sacrament event dates or locations.")
    elif "cnf" in fn_lower or "confirmation" in fn_lower:
        extra.append("This field relates to the Confirmation sacrament event dates, parishes, sponsors, or locations.")
    elif "bapt" in fn_lower or "baptism" in fn_lower:
        extra.append("This field relates to the Baptism sacrament event dates, parishes, godparents, or locations.")
    elif "mrg" in fn_lower or "marriage" in fn_lower or "bride" in fn_lower or "groom" in fn_lower:
        extra.append("This field relates to the Marriage sacrament register details, bridegroom details, bride details, wedding witnesses, bans, or date/place of marriage.")
    elif "burial" in fn_lower or "death" in fn_lower:
        extra.append("This field relates to the Death register, burial details, cause of death, cemetery, or date/place of burial.")
    elif "anointing" in fn_lower or "sick" in fn_lower:
        extra.append("This field relates to the Anointing of the Sick sacrament register, illness details, priest minister, or date/place of anointing.")
    elif "bcc" in fn_lower:
        extra.append("This field relates to the Basic Christian Community (BCC) neighborhood family cell or BCC group identifier.")
    elif "parish_id" in fn_lower or "diocese_id" in fn_lower or "vicariate_id" in fn_lower:
        extra.append("This field registers the location boundaries and jurisdiction details, such as parish name, diocese name, or vicariate name.")
    elif "sub_station" in fn_lower or table_name == "tabSub Station":
        extra.append("This field relates to the Sub Station, chapel, filial church, patron saint, or feast day celebration.")
    elif fn_lower in ("age", "dob", "date_of_birth", "is_dob_or_age"):
        extra.append("This field represents the age in years, date of birth, or birthdate calculation for the parishioner or member.")
    elif fn_lower == "gender":
        extra.append("This field represents the biological sex or gender of the member (e.g., Male, Female).")
    elif "relationship" in fn_lower:
        extra.append("This field represents the family relationship or kinship of the member (e.g., Head, Spouse, Son, Daughter, Father, Mother).")
    elif "is_family_head" in fn_lower:
        extra.append("This field indicates whether the member is the head of the family.")
    elif "living_status" in fn_lower:
        extra.append("This field indicates whether the member is currently living or deceased.")
    elif "marital_status" in fn_lower:
        extra.append("This field stores the marital status of the member (e.g., Single, Married, Widowed, Divorced).")
    elif "blood_group" in fn_lower:
        extra.append("This field records the blood group / blood type of the member.")
    elif fn_lower in ("mobile", "phone", "email"):
        extra.append("This field stores the telephone contact number, mobile phone number, or email address.")
    elif "occupation" in fn_lower or "education" in fn_lower:
        extra.append("This field stores the professional occupation, trade, work, or educational qualification.")
    elif "income" in fn_lower or "economic_status" in fn_lower or "rent_amt" in fn_lower or "house" in fn_lower:
        extra.append("This field stores the family economic status, income, house ownership, or rent amount.")
    elif "father_name" in fn_lower or "mother_name" in fn_lower:
        extra.append("This field stores the name of the father or mother.")

    if extra:
        desc += " " + " ".join(extra)
    else:
        desc += f" This field stores the {field_label.lower()} details."
        
    return field_label, desc

def get_current_pg_state():
    """Gets current table DDLs and field schemas from pgvector for differential checking."""
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT table_name, schema_ddl FROM koinonia_table_schemas;")
    existing_tables = {row[0]: row[1] for row in cur.fetchall()}
    
    cur.execute("SELECT table_name, field_name, field_type, field_label, description FROM koinonia_field_schemas;")
    existing_fields = {}
    for r in cur.fetchall():
        tname, fname, ftype, flabel, fdesc = r
        if tname not in existing_fields:
            existing_fields[tname] = {}
        existing_fields[tname][fname] = {
            "field_type": ftype,
            "field_label": flabel,
            "description": fdesc
        }
    conn.close()
    return existing_tables, existing_fields

def main():
    frappe.init(site="frontend")
    frappe.connect()
    
    print("=" * 70)
    print("  KOINONIA VECTOR DB RESYNC (MariaDB → BGE-M3 → pgvector)")
    print("=" * 70)
    
    init_pg_tables()
    docfield_metadata = get_docfield_metadata()
    mariadb_meta = get_mariadb_table_metadata()
    existing_pg_tables, existing_pg_fields = get_current_pg_state()
    
    print(f"\n[1/3] Synchronizing Table Schemas ({len(mariadb_meta)} tables)...")
    tables_to_embed = []
    table_payloads = []
    for table_name, columns in mariadb_meta.items():
        doc_meta = docfield_metadata.get(table_name, {})
        new_ddl = build_table_description(table_name, columns, doc_meta)
        curr_ddl = existing_pg_tables.get(table_name)
        
        if curr_ddl != new_ddl:
            status = "NEW" if table_name not in existing_pg_tables else "UPDATED"
            print(f"  [{status}] Table schema: {table_name}")
            tables_to_embed.append(new_ddl)
            table_payloads.append((table_name, new_ddl))
        else:
            print(f"  [UNCHANGED] Table schema: {table_name}")
            
    if tables_to_embed:
        print(f"  -> Generating BGE-M3 embeddings for {len(tables_to_embed)} tables...")
        table_embeddings = embed_texts(tables_to_embed, batch_size=16)
        conn = psycopg2.connect(**PG_CONFIG)
        with conn.cursor() as cur:
            for (tname, ddl), emb in zip(table_payloads, table_embeddings):
                cur.execute("""
                    INSERT INTO koinonia_table_schemas (table_name, schema_ddl, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (table_name) DO UPDATE
                        SET schema_ddl = EXCLUDED.schema_ddl,
                            embedding  = EXCLUDED.embedding;
                """, (tname, ddl, emb))
        conn.commit()
        conn.close()
        print(f"  ✓ Successfully updated {len(tables_to_embed)} table schemas in pgvector.")
    else:
        print("  ✓ All table schemas already up-to-date in pgvector.")
        
    print(f"\n[2/3] Analyzing and Synchronizing Field Schemas...")
    fields_to_embed = []
    field_payloads = []
    unchanged_fields_count = 0
    new_fields_count = 0
    updated_fields_count = 0
    
    active_columns_per_table = {}
    
    for table_name, columns in mariadb_meta.items():
        doc_meta = docfield_metadata.get(table_name, {})
        active_columns_per_table[table_name] = set()
        
        for col in columns:
            col_name = col["column"]
            col_type = col["type"]
            
            if col_name in META_COLS:
                continue
                
            active_columns_per_table[table_name].add(col_name)
            df_info = doc_meta.get(col_name)
            label, desc = build_field_description(table_name, col_name, col_type, df_info)
            
            existing = existing_pg_fields.get(table_name, {}).get(col_name)
            if existing is None:
                new_fields_count += 1
                fields_to_embed.append(desc)
                field_payloads.append((table_name, col_name, col_type, label, desc, "NEW"))
            elif existing["field_type"] != col_type or existing["field_label"] != label or existing["description"] != desc:
                updated_fields_count += 1
                fields_to_embed.append(desc)
                field_payloads.append((table_name, col_name, col_type, label, desc, "UPDATED"))
            else:
                unchanged_fields_count += 1
                
    print(f"  Field Analysis:")
    print(f"    - Unchanged fields : {unchanged_fields_count}")
    print(f"    - New fields       : {new_fields_count}")
    print(f"    - Updated fields   : {updated_fields_count}")
    print(f"    - Total to sync    : {len(field_payloads)}")
    
    if field_payloads:
        print(f"\n  -> Generating BGE-M3 embeddings for {len(field_payloads)} fields in batches...")
        t0 = time.time()
        field_embeddings = embed_texts(fields_to_embed, batch_size=32)
        print(f"  -> Generated {len(field_embeddings)} embeddings in {time.time() - t0:.2f}s.")
        
        conn = psycopg2.connect(**PG_CONFIG)
        with conn.cursor() as cur:
            for (tname, fname, ftype, flabel, fdesc, status), emb in zip(field_payloads, field_embeddings):
                cur.execute("""
                    INSERT INTO koinonia_field_schemas (table_name, field_name, field_type, field_label, description, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (table_name, field_name) DO UPDATE
                        SET field_type  = EXCLUDED.field_type,
                            field_label = EXCLUDED.field_label,
                            description = EXCLUDED.description,
                            embedding   = EXCLUDED.embedding;
                """, (tname, fname, ftype, flabel, fdesc, emb))
                print(f"    [{status}] {tname}.{fname} (Type: {ftype}, Label: {flabel})")
        conn.commit()
        conn.close()
        print(f"  ✓ Successfully synchronized {len(field_payloads)} fields into pgvector.")
    else:
        print("  ✓ All fields are already in perfect sync with MariaDB.")
        
    print(f"\n[3/3] Checking and cleaning obsolete fields...")
    conn = psycopg2.connect(**PG_CONFIG)
    with conn.cursor() as cur:
        deleted_count = 0
        for table_name, active_cols in active_columns_per_table.items():
            cur.execute("""
                SELECT field_name FROM koinonia_field_schemas WHERE table_name = %s;
            """, (table_name,))
            db_cols = {row[0] for row in cur.fetchall()}
            obsolete = db_cols - active_cols
            if obsolete:
                cur.execute("""
                    DELETE FROM koinonia_field_schemas
                    WHERE table_name = %s AND field_name = ANY(%s);
                """, (table_name, list(obsolete)))
                deleted_count += len(obsolete)
                print(f"    [DELETED OBSOLETE] {table_name}: {sorted(list(obsolete))}")
        conn.commit()
        if deleted_count == 0:
            print("  ✓ No obsolete fields found in pgvector.")
        else:
            print(f"  ✓ Cleaned up {deleted_count} obsolete fields.")
            
        cur.execute("SELECT table_name, count(*) FROM koinonia_field_schemas GROUP BY table_name ORDER BY table_name;")
        print("\n=== FINAL PGVECTOR SCHEMA SUMMARY ===")
        for r in cur.fetchall():
            print(f"  - Table `{r[0]}`: {r[1]} fields synced")
        cur.execute("SELECT count(*) FROM koinonia_field_schemas;")
        print(f"\nTotal synced fields in koinonia_field_schemas: {cur.fetchone()[0]}")
        cur.execute("SELECT table_name FROM koinonia_table_schemas ORDER BY table_name;")
        print(f"Total synced tables in koinonia_table_schemas: {[r[0] for r in cur.fetchall()]}")
    conn.close()
    
    print("\n" + "=" * 70)
    print("  VECTOR DB RESYNC COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
