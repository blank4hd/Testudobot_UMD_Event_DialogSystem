import logging
import os
from contextlib import contextmanager
from typing import Dict

import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DB_NAME = os.getenv("DB_NAME", "umd_events")
DB_USER = os.getenv("DB_USER", "umd_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "umd_password")
DB_HOST = os.getenv("DB_HOST", "db")

db_pool = None
try:
    db_pool = pool.SimpleConnectionPool(
        1,
        20,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port="5432",
    )
    if db_pool:
        logger.info("✅ Database connection pool created successfully")
except Exception:
    logger.exception("❌ Error creating connection pool")


@contextmanager
def get_db_cursor(cursor_factory=None):
    if db_pool is None:
        raise RuntimeError("Database connection pool is not initialized")
    conn = db_pool.getconn()
    try:
        if cursor_factory:
            yield conn.cursor(cursor_factory=cursor_factory)
        else:
            yield conn.cursor()
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        db_pool.putconn(conn)


def init_db():
    """Initializes the database schema."""
    try:
        with get_db_cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("ALTER TABLE umd_events ADD COLUMN IF NOT EXISTS topic_id INTEGER DEFAULT -1;")
            cur.execute("ALTER TABLE umd_events ADD COLUMN IF NOT EXISTS embedding VECTOR(768);")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_labels (
                    topic_id INTEGER PRIMARY KEY,
                    label TEXT,
                    keywords TEXT
                );
            """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS umd_events_embedding_idx
                ON umd_events USING hnsw (embedding vector_cosine_ops);
            """
            )
            logger.info("✅ Database initialized.")
    except Exception:
        logger.exception("DB initialization failed")


def fetch_topic_map() -> Dict[str, str]:
    """Returns { 'Career': '1', 'Music': '2' } for the dropdown."""
    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM topic_labels;")
            count = cur.fetchone()[0]
            logger.info("🔍 DEBUG: 'topic_labels' table contains %s rows.", count)

            cur.execute("SELECT topic_id, label FROM topic_labels WHERE topic_id != -1 ORDER BY label;")
            rows = cur.fetchall()

            result = {row[1]: str(row[0]) for row in rows}
            logger.info("🔍 DEBUG: Topic Map returning: %s", result)
            return result
    except Exception:
        logger.exception("Failed to fetch topic map")
        return {}
