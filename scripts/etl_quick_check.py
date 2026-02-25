import os
import time
import argparse
from datetime import date, timedelta

import psycopg2
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

try:
    from scripts.scrape import scrape_events
except ImportError:
    from scrape import scrape_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick ETL health diagnostics")
    parser.add_argument(
        "--scrape-days",
        type=int,
        default=0,
        help="Number of days ahead to include in scrape check (0 = skip scrape)",
    )
    args = parser.parse_args()

    load_dotenv()

    print("--- quick etl diagnostics ---")
    print(f"DB_HOST from .env: {os.getenv('DB_HOST')}")
    print(f"POSTGRES_HOST env: {os.getenv('POSTGRES_HOST')}")
    print(f"ELASTIC_HOST env: {os.getenv('ELASTIC_HOST')}")

    db_start = time.time()
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "umd_events")),
            user=os.getenv("POSTGRES_USER", os.getenv("DB_USER", "umd_user")),
            password=os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", "umd_password")),
            host=os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "localhost")),
            port=os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "5432")),
            connect_timeout=5,
        )
        conn.close()
        print(f"DB connect: OK in {time.time() - db_start:.2f}s")
    except Exception as exc:
        print(f"DB connect: FAIL in {time.time() - db_start:.2f}s -> {exc}")

    es_start = time.time()
    try:
        es = Elasticsearch(os.getenv("ELASTIC_HOST", "http://localhost:9200"))
        print(f"ES ping: {es.ping()} in {time.time() - es_start:.2f}s")
    except Exception as exc:
        print(f"ES ping: FAIL in {time.time() - es_start:.2f}s -> {exc}")

    if args.scrape_days > 0:
        start_date = date.today().isoformat()
        end_date = (date.today() + timedelta(days=args.scrape_days)).isoformat()
        scrape_start = time.time()
        try:
            events = scrape_events(start_date, end_date)
            print(
                f"Scrape {args.scrape_days}-day range ({start_date} to {end_date}): "
                f"{len(events)} events in {time.time() - scrape_start:.2f}s"
            )
        except Exception as exc:
            print(f"Scrape: FAIL in {time.time() - scrape_start:.2f}s -> {exc}")
    else:
        print("Scrape check: SKIPPED (use --scrape-days N to enable)")


if __name__ == "__main__":
    main()