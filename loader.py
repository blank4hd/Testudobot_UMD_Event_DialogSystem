import os
import json
import time
import psycopg2
from psycopg2 import OperationalError

# -------------------------------------------------------------------
# 1. JSON loading
# -------------------------------------------------------------------

def load_events_from_json(path: str):
    print(f"🔎 Looking for JSON at: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found at '{path}'")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of event objects (list[dict])")

    return data


# -------------------------------------------------------------------
# 2. Postgres connection with retry
# -------------------------------------------------------------------

def get_pg_connection_with_retry(max_attempts: int = 20, delay_sec: float = 2.0):
    """
    Try to connect to Postgres with retries while the DB container is starting up.
    """
    dbname = os.getenv("POSTGRES_DB", "umd_events")
    user = os.getenv("POSTGRES_USER", "umd_user")
    password = os.getenv("POSTGRES_PASSWORD", "umd_password")
    host = os.getenv("POSTGRES_HOST", "db")
    port = os.getenv("POSTGRES_PORT", "5432")

    for attempt in range(1, max_attempts + 1):
        try:
            conn = psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
            )
            print(f"✅ Connected to Postgres on attempt {attempt}")
            return conn
        except OperationalError as e:
            print(
                f"⏳ Postgres not ready yet (attempt {attempt}/{max_attempts}): "
                f"{e.__class__.__name__}: {e}"
            )
            if attempt == max_attempts:
                print("❌ Giving up connecting to Postgres.")
                raise
            time.sleep(delay_sec)


# -------------------------------------------------------------------
# 3. Load into DB
# -------------------------------------------------------------------

def load_to_postgres(events):
    """
    Load a list of event dicts into Postgres.
    Table is created if it doesn't exist. We TRUNCATE on each run
    so the table always reflects the JSON snapshot.
    """
    conn = get_pg_connection_with_retry()
    conn.autocommit = True
    cur = conn.cursor()

    # Create table if not exists
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS umd_events (
            id SERIAL PRIMARY KEY,
            event TEXT,
            date TEXT,
            time TEXT,
            url TEXT,
            location TEXT,
            description TEXT
        );
        """
    )

    # Clear old data
    cur.execute("TRUNCATE TABLE umd_events;")

    insert_sql = """
        INSERT INTO umd_events (event, date, time, url, location, description)
        VALUES (%s, %s, %s, %s, %s, %s);
    """

    for ev in events:
        cur.execute(
            insert_sql,
            (
                ev.get("event"),
                ev.get("date"),
                ev.get("time"),
                ev.get("url"),
                ev.get("location"),
                ev.get("description"),
            ),
        )

    cur.close()
    conn.close()
    print(f"🎉 Loaded {len(events)} events into Postgres successfully")


# -------------------------------------------------------------------
# 4. Main entrypoint
# -------------------------------------------------------------------

def main():
    # Directory where loader.py lives (same as your JSON file on host)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # JSON filename in the *current* folder
    json_filename = os.getenv(
        "EVENTS_JSON_NAME",
        "umd_calendar_2025-10-01_to_2025-10-31.json",
    )

    json_path = os.path.join(base_dir, json_filename)
    print(f"📂 Loading events from: {json_path}")

    events = load_events_from_json(json_path)
    load_to_postgres(events)


if __name__ == "__main__":
    main()
