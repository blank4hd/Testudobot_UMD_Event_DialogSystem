import os
import json
import time
import sys
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

# ============================
#  CONNECTION HELPERS (ROBUST)
# ============================

def get_pg_connection_with_retry(max_attempts: int = 20, delay_sec: float = 3.0):
    """Waits for Postgres to be ready."""
    dbname = os.getenv("POSTGRES_DB", "umd_events")
    user = os.getenv("POSTGRES_USER", "umd_user")
    password = os.getenv("POSTGRES_PASSWORD", "umd_password")
    host = os.getenv("POSTGRES_HOST", "db")
    port = os.getenv("POSTGRES_PORT", "5432")

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
    es_host = os.getenv("ELASTIC_HOST", "http://elasticsearch:9200")
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

def setup_elasticsearch(es):
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
        print_flush("🔄 Recreating ES index...")
        es.indices.delete(index=index_name)
    
    es.indices.create(index=index_name, body=mapping)
    print_flush("✅ Elasticsearch index 'umd_events' created.")

# ============================
#  MAIN LOADING LOGIC
# ============================

def load_data(events):
    if not events:
        print_flush("❌ No events to load. Exiting.")
        return

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
    except Exception as e:
        print_flush(f"❌ Table creation failed: {e}")
        raise

    # Step C: Optional Clean Slate
    if os.getenv("FORCE_CLEAN_SCHEMA", "true") == "true":
        cur.execute("TRUNCATE TABLE umd_events RESTART IDENTITY;")
        print_flush("🗑️ Cleared existing Postgres data.")

    # Step D: Insert into Postgres
    print_flush("🔄 Inserting into Postgres...")
    pg_ids = []
    
    insert_sql = """
        INSERT INTO umd_events (event, date, time, url, location, description, topic_id, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    
    for i, ev in enumerate(events):
        try:
            cur.execute(insert_sql, (
                ev.get("event"), 
                ev.get("date"), 
                ev.get("time"), 
                ev.get("url"), 
                ev.get("location"), 
                ev.get("description"), 
                -1,  # Default topic
                embeddings[i].tolist(), 
            ))
            pg_id = cur.fetchone()[0]
            pg_ids.append(pg_id)
        except Exception as e:
            print_flush(f"❌ Postgres insert failed for event {i+1}: {e}")
            pg_ids.append(None) # Keep index alignment

    print_flush(f"✅ Inserted {len([x for x in pg_ids if x is not None])} events into Postgres.")

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
    # CRITICAL CHANGE: Use the retry function here
    es = get_es_client_with_retry() 
    setup_elasticsearch(es)
    
    actions = []
    print_flush("🔄 Preparing Elasticsearch bulk insert...")
    for i, ev in enumerate(events):
        if pg_ids[i] is None: continue # Skip if Postgres insert failed

        doc = {
            "_index": "umd_events",
            "_id": pg_ids[i], # Link directly to Postgres ID
            "_source": {
                "event": ev.get("event"),
                "date": ev.get("date"),
                "time": ev.get("time"),
                "url": ev.get("url"),
                "location": ev.get("location"),
                "description": ev.get("description"),
                "topic_id": -1, 
                "embedding": embeddings[i].tolist()
            }
        }
        actions.append(doc)

    try:
        success, _ = helpers.bulk(es, actions)
        print_flush(f"🎉 Indexed {success} documents in Elastic!")
    except Exception as e:
        print_flush(f"❌ Elasticsearch loading failed: {e}")
        raise

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_filename = os.getenv("EVENTS_JSON_NAME", "umd_calendar_2025-10-01_to_2025-10-31.json")
    json_path = os.path.join(base_dir, json_filename)
    print_flush(f"📂 Loading events from: {json_path}")

    try:
        events = load_events_from_json(json_path)
        load_data(events)
    except Exception as e:
        print_flush(f"❌ FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()