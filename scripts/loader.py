import os
import json
import time
import sys
import hashlib
import glob
import re
import psycopg2
from psycopg2 import OperationalError
from sentence_transformers import SentenceTransformer
import numpy as np
from elasticsearch import Elasticsearch, helpers
import logging

# ============================
#  LOGGING SETUP
# ============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_flush(msg):
    logger.info(msg)

# ============================
#  DATA LOADING HELPERS
# ============================

def load_events_from_json(path: str):
    print_flush(f"🔎 Looking for JSON at: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found at '{path}'")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of event objects (list[dict])")

    print_flush(f"✅ Loaded {len(data)} events from JSON.")
    return data


def resolve_events_json_path(project_root: str) -> str:
    env_json_name = os.getenv("EVENTS_JSON_NAME")
    if env_json_name:
        json_path = env_json_name if os.path.isabs(env_json_name) else os.path.join(project_root, env_json_name)
        return json_path

    data_dir = os.path.join(project_root, "data")
    dated_files = glob.glob(os.path.join(data_dir, "umd_events_*.json"))
    date_pattern = re.compile(r"umd_events_(\d{4}-\d{2}-\d{2})\.json$")

    dated_candidates: list[tuple[str, str]] = []
    for file_path in dated_files:
        match = date_pattern.search(os.path.basename(file_path))
        if match:
            dated_candidates.append((match.group(1), file_path))

    if dated_candidates:
        dated_candidates.sort(key=lambda item: item[0])
        return dated_candidates[-1][1]

    fallback_path = os.path.join(project_root, "data", "umd_calendar_2025-10-01_to_2025-10-31.json")
    return fallback_path

# ============================
#  CONNECTION HELPERS (ROBUST)
# ============================

def get_pg_connection_with_retry(max_attempts: int = 20, delay_sec: float = 3.0):
    """Waits for Postgres to be ready."""
    dbname = os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "umd_events"))
    user = os.getenv("POSTGRES_USER", os.getenv("DB_USER", "umd_user"))
    password = os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", "umd_password"))
    host = os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "localhost"))
    port = os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "5432"))

    for attempt in range(1, max_attempts + 1):
        try:
            conn = psycopg2.connect(
                dbname=dbname, user=user, password=password, host=host, port=port
            )
            print_flush(f"✅ Connected to Postgres on attempt {attempt}")
            return conn
        except OperationalError:
            print_flush(f"⏳ Postgres not ready (attempt {attempt}/{max_attempts})...")
            time.sleep(delay_sec)
    
    raise Exception("❌ Could not connect to Postgres after multiple attempts.")

def get_es_client_with_retry(max_attempts: int = 20, delay_sec: float = 5.0):
    """Waits for Elasticsearch to be ready (CRITICAL FIX)."""
    es_host = os.getenv("ELASTIC_HOST", "http://localhost:9200")
    es = Elasticsearch(es_host)

    for attempt in range(1, max_attempts + 1):
        try:
            if es.ping():
                print_flush(f"✅ Connected to Elasticsearch on attempt {attempt}")
                return es
            else:
                print_flush(f"⏳ Elasticsearch reachable but not ready (attempt {attempt}/{max_attempts})...")
        except Exception as e:
            print_flush(f"⏳ Elasticsearch not reachable (attempt {attempt}/{max_attempts}): {e}")
        
        time.sleep(delay_sec)

    raise Exception("❌ Could not connect to Elasticsearch after multiple attempts.")

def setup_elasticsearch(es, recreate: bool = False):
    """Creates the index with specific mappings for Hybrid Search."""
    index_name = "umd_events"
    
    mapping = {
        "mappings": {
            "properties": {
                "event": {"type": "text"},       # BM25 Searchable
                "description": {"type": "text"}, # BM25 Searchable
                "date": {"type": "keyword"},
                "location": {"type": "text"},
                "topic_id": {"type": "integer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    if es.indices.exists(index=index_name):
        if recreate:
            print_flush("🔄 Recreating ES index...")
            es.indices.delete(index=index_name)
            es.indices.create(index=index_name, body=mapping)
            print_flush("✅ Elasticsearch index 'umd_events' recreated.")
        else:
            print_flush("✅ Elasticsearch index 'umd_events' already exists.")
        return

    es.indices.create(index=index_name, body=mapping)
    print_flush("✅ Elasticsearch index 'umd_events' created.")


def compute_event_hash(event: dict) -> str:
    hash_input = "||".join([
        str(event.get("event") or "").strip(),
        str(event.get("date") or "").strip(),
        str(event.get("time") or "").strip(),
        str(event.get("url") or "").strip(),
    ])
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def upsert_events(events: list[dict], embeddings) -> dict:
    summary = {"inserted": 0, "updated": 0, "skipped": 0}

    conn = get_pg_connection_with_retry()
    conn.autocommit = True
    cur = conn.cursor()
    es = get_es_client_with_retry()

    actions = []
    select_sql = """
        SELECT id, description, location
        FROM umd_events
        WHERE content_hash = %s;
    """
    insert_sql = """
        INSERT INTO umd_events (event, date, time, url, location, description, topic_id, embedding, content_hash)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    update_sql = """
        UPDATE umd_events
        SET description = %s,
            location = %s
        WHERE id = %s;
    """

    for i, ev in enumerate(events):
        content_hash = compute_event_hash(ev)
        try:
            cur.execute(select_sql, (content_hash,))
            existing = cur.fetchone()

            source_doc = {
                "event": ev.get("event"),
                "date": ev.get("date"),
                "time": ev.get("time"),
                "url": ev.get("url"),
                "location": ev.get("location"),
                "description": ev.get("description"),
                "topic_id": -1,
                "embedding": embeddings[i].tolist(),
            }

            if existing is None:
                cur.execute(insert_sql, (
                    ev.get("event"),
                    ev.get("date"),
                    ev.get("time"),
                    ev.get("url"),
                    ev.get("location"),
                    ev.get("description"),
                    -1,
                    embeddings[i].tolist(),
                    content_hash,
                ))
                pg_id = cur.fetchone()[0]
                summary["inserted"] += 1
                actions.append({
                    "_op_type": "index",
                    "_index": "umd_events",
                    "_id": pg_id,
                    "_source": source_doc,
                })
                continue

            pg_id, existing_description, existing_location = existing
            description_changed = (existing_description or "") != (ev.get("description") or "")
            location_changed = (existing_location or "") != (ev.get("location") or "")

            if description_changed or location_changed:
                cur.execute(update_sql, (
                    ev.get("description"),
                    ev.get("location"),
                    pg_id,
                ))
                summary["updated"] += 1
                actions.append({
                    "_op_type": "index",
                    "_index": "umd_events",
                    "_id": pg_id,
                    "_source": source_doc,
                })
            else:
                summary["skipped"] += 1
        except Exception as e:
            print_flush(f"❌ Upsert failed for event {i+1}: {e}")
            summary["skipped"] += 1

    if actions:
        helpers.bulk(es, actions)

    cur.close()
    conn.close()
    return summary


def remove_stale_events(current_event_hashes: set[str]) -> int:
    conn = get_pg_connection_with_retry()
    conn.autocommit = True
    cur = conn.cursor()
    es = get_es_client_with_retry()

    cur.execute("SELECT id, content_hash FROM umd_events;")
    rows = cur.fetchall()

    stale_ids = [row[0] for row in rows if (not row[1]) or (row[1] not in current_event_hashes)]
    stale_removed = len(stale_ids)

    if stale_ids:
        cur.execute("DELETE FROM umd_events WHERE id = ANY(%s);", (stale_ids,))
        delete_actions = [
            {
                "_op_type": "delete",
                "_index": "umd_events",
                "_id": stale_id,
            }
            for stale_id in stale_ids
        ]
        helpers.bulk(es, delete_actions, raise_on_error=False)

    print_flush(f"🧹 Removed {stale_removed} stale events.")
    cur.close()
    conn.close()
    return stale_removed

# ============================
#  MAIN LOADING LOGIC
# ============================

def load_data(events: list[dict] | None = None):
    if events is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        json_path = resolve_events_json_path(project_root)
        print_flush(f"📂 Loading events from: {json_path}")
        events = load_events_from_json(json_path)

    if not events:
        print_flush("❌ No events to load. Exiting.")
        return {"inserted": 0, "updated": 0, "skipped": 0, "stale_removed": 0}

    # --- 1. Compute Embeddings (ONCE for both DBs) ---
    print_flush("🔄 Loading embedding model (all-mpnet-base-v2)...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    print_flush("🔄 Computing embeddings...")
    # Clean text to avoid newlines breaking things
    texts = [
        f"{ev.get('event', '')} {ev.get('description', '')} {ev.get('location', '')}".replace("\n", " ").strip() 
        for ev in events
    ]
    embeddings = model.encode(texts, normalize_embeddings=True).astype('float32')
    print_flush(f"✅ Computed {len(embeddings)} embeddings.")

    # --- 2. Load into PostgreSQL (Storage & Topics) ---
    conn = get_pg_connection_with_retry()
    conn.autocommit = True
    cur = conn.cursor()

    # Step A: Check and create pgvector extension
    print_flush("🔄 Checking pgvector extension...")
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print_flush("✅ pgvector extension ensured.")
    except Exception as e:
        print_flush(f"❌ pgvector extension failed: {e}")
        raise

    # Step B: Create Table
    print_flush("🔄 Ensuring table schema...")
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS umd_events (
                id SERIAL PRIMARY KEY,
                event TEXT,
                date TEXT,
                time TEXT,
                url TEXT,
                location TEXT,
                description TEXT,
                topic_id INTEGER DEFAULT -1,
                embedding VECTOR(768)
            );
        """)

        cur.execute("ALTER TABLE umd_events ADD COLUMN IF NOT EXISTS content_hash TEXT UNIQUE;")
    except Exception as e:
        print_flush(f"❌ Table creation failed: {e}")
        raise

    force_clean_schema = os.getenv("FORCE_CLEAN_SCHEMA", "false").lower() == "true"

    # Step C: Optional Clean Slate
    if force_clean_schema:
        cur.execute("TRUNCATE TABLE umd_events RESTART IDENTITY;")
        print_flush("🗑️ Cleared existing Postgres data.")

    # Step E: Create Index (Postgres)
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS umd_events_embedding_idx ON umd_events USING hnsw (embedding vector_cosine_ops);")
        print_flush("🗂️ Postgres HNSW index created.")
    except Exception as e:
        print_flush(f"⚠️ Postgres index creation warning: {e}")

    cur.close()
    conn.close()

    # --- 3. Load into Elasticsearch (Search Engine) ---
    print_flush("🔄 Connecting to Elasticsearch...")
    es = get_es_client_with_retry()
    setup_elasticsearch(es, recreate=force_clean_schema)

    summary = upsert_events(events, embeddings)
    current_hashes = {compute_event_hash(ev) for ev in events}
    stale_removed = remove_stale_events(current_hashes)
    print_flush(
        f"✅ Loader complete: {summary['inserted']} inserted, {summary['updated']} updated, "
        f"{summary['skipped']} skipped, {stale_removed} stale removed"
    )
    return {
        "inserted": summary["inserted"],
        "updated": summary["updated"],
        "skipped": summary["skipped"],
        "stale_removed": stale_removed,
    }

def main():
    try:
        load_data()
    except Exception as e:
        print_flush(f"❌ FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()