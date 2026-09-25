import os
import sys
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

def load_bge_model():
    global _tokenizer, _model
    if _tokenizer is None:
        print("[BGE-M3] Loading model in koinonia_assistant...")
        _tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
        _model = AutoModel.from_pretrained(BGE_MODEL_NAME)
        _model.eval()
        print("[BGE-M3] Model loaded.")

def embed_text(text: str) -> list[float]:
    load_bge_model()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = _model(**inputs)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding

def init_pg_tables():
    """Create pgvector schema and history tables if not exist."""
    print("[Postgres] Initializing koinonia pgvector tables...")
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Table for table schemas
            cur.execute("""
                CREATE TABLE IF NOT EXISTS koinonia_table_schemas (
                    table_name VARCHAR(100) PRIMARY KEY,
                    schema_ddl TEXT NOT NULL,
                    embedding vector(1024) NOT NULL
                );
            """)
            # Table for field-level schemas
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
            # Table for query history (few-shot)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS koinonia_query_history (
                    id SERIAL PRIMARY KEY,
                    user_question TEXT NOT NULL,
                    generated_sql TEXT NOT NULL,
                    embedding vector(1024) NOT NULL,
                    correctness_flag INT DEFAULT NULL
                );
            """)
        conn.commit()
    finally:
        conn.close()

def get_custom_table_metadata() -> dict[str, list[dict]]:
    """Reads schema columns from information_schema for only our 8 custom tables."""
    tables = [
        "tabFamily",
        "tabMember",
        "tabBaptism",
        "tabCommunion",
        "tabConfirmation",
        "tabMarriage",
        "tabAnointing Of Sick",
        "tabDeath"
    ]
    
    db_name = frappe.conf.db_name
    metadata = {}
    
    for table in tables:
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

def build_table_description(table_name: str, columns: list[dict]) -> str:
    clean_name = table_name[3:] if table_name.startswith("tab") else table_name
    lines = [
        f"Table Name: `{table_name}`",
        f"Entity/Concept: {clean_name}",
        "Columns:"
    ]
    for col in columns:
        nullable_str = "nullable" if col["nullable"] == "YES" else "NOT NULL"
        key_str      = f" [{col['key']}]" if col["key"] else ""
        lines.append(f"  - `{col['column']}` ({col['type']}, {nullable_str}){key_str}")
    return "\n".join(lines)

def build_field_description(table_name: str, field_name: str, field_type: str) -> tuple[str, str]:
    field_label = field_name.replace("_", " ").title()
    table_concept = table_name[3:] if table_name.startswith("tab") else table_name
    
    desc = f"Table: `{table_name}` ({table_concept}), Field: `{field_name}` (Type: {field_type}), Label: {field_label}."
    
    extra = []
    fn_lower = field_name.lower()
    
    if "god_parent" in fn_lower or "godparent" in fn_lower or "god_father" in fn_lower or "god_mother" in fn_lower or "sponsor" in fn_lower:
        extra.append("This field represents the godfather, godmother, sponsor, or witness details for a baptism or confirmation sacrament event.")
    elif "minister" in fn_lower or "priest" in fn_lower:
        extra.append("This field represents the priest, pastor, or minister officiating, solemnizing, or conducting the sacrament, ceremony, burial, or anointing.")
    elif "family_card" in fn_lower or "family_id" in fn_lower or "reference" in fn_lower:
        extra.append("This field links the sacrament record or the member back to their family register number, family card number, or family file name.")
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
    elif "bcc" in fn_lower:
        extra.append("This field relates to the Basic Christian Community (BCC) neighborhood family cell or BCC group identifier.")
    elif "parish_id" in fn_lower or "diocese_id" in fn_lower or "vicariate_id" in fn_lower:
        extra.append("This field registers the location boundaries and jurisdiction details, such as parish name, diocese name, or vicariate name.")
        
    if extra:
        desc += " " + " ".join(extra)
    else:
        desc += f" This field stores the {field_label.lower()} details."
        
    return field_label, desc

def upsert_pg_schema(table_name: str, schema_ddl: str, embedding: list[float]):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO koinonia_table_schemas (table_name, schema_ddl, embedding)
                VALUES (%s, %s, %s::vector)
                ON CONFLICT (table_name) DO UPDATE
                    SET schema_ddl = EXCLUDED.schema_ddl,
                        embedding  = EXCLUDED.embedding;
            """, (table_name, schema_ddl, embedding))
        conn.commit()
    finally:
        conn.close()

def upsert_pg_field(table_name: str, field_name: str, field_type: str, field_label: str, description: str, embedding: list[float]):
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO koinonia_field_schemas (table_name, field_name, field_type, field_label, description, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (table_name, field_name) DO UPDATE
                    SET field_type = EXCLUDED.field_type,
                        field_label = EXCLUDED.field_label,
                        description = EXCLUDED.description,
                        embedding = EXCLUDED.embedding;
            """, (table_name, field_name, field_type, field_label, description, embedding))
        conn.commit()
    finally:
        conn.close()

def main():
    frappe.init(site="frontend")
    frappe.connect()
    
    print("=" * 60)
    print("  Koinonia Schema Ingest  (MariaDB → BGE-M3 → pgvector)")
    print("=" * 60)
    
    init_pg_tables()
    metadata = get_custom_table_metadata()
    
    print(f"\n[Ingest] Ingesting {len(metadata)} custom tables into pgvector...")
    for table_name, columns in metadata.items():
        description = build_table_description(table_name, columns)
        embedding   = embed_text(description)
        upsert_pg_schema(table_name, description, embedding)
        print(f"  ✓ Ingested table schema: {table_name}")
        
    print(f"\n[Ingest] Ingesting field-level schemas into pgvector...")
    for table_name, columns in metadata.items():
        for col in columns:
            col_name = col["column"]
            col_type = col["type"]
            
            # Skip standard Frappe meta fields to save space
            if col_name in ["name", "owner", "creation", "modified", "modified_by", "docstatus", "idx", "_user_tags", "_comments", "_assign", "_liked_by"]:
                continue
                
            label, desc = build_field_description(table_name, col_name, col_type)
            emb = embed_text(desc)
            upsert_pg_field(table_name, col_name, col_type, label, desc, emb)
        print(f"  ✓ Ingested field schemas for table: {table_name}")
        
    print("\n=== Ingest Completed Successfully ===")

if __name__ == "__main__":
    main()
