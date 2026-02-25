import logging
import sys
import time
import argparse
import os
import json
from datetime import date, datetime

import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

try:
    from scripts.loader import load_data
    from scripts.scrape import scrape_events
except ImportError:
    from loader import load_data
    from scrape import scrape_events


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _save_scrape_snapshot(events: list[dict], scraped_date: date) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    filename = f"umd_events_{scraped_date.isoformat()}.json"
    file_path = os.path.join(data_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    logger.info("💾 Saved scrape snapshot to %s", file_path)
    return file_path


def _scrape_with_retry(start_date: str, end_date: str) -> list[dict]:
    try:
        return scrape_events(start_date, end_date)
    except requests.RequestException as exc:
        logger.error("❌ Scrape failed on first attempt: %s", exc)
        logger.info("⏳ Retrying scrape in 60 seconds...")
        time.sleep(60)
        try:
            return scrape_events(start_date, end_date)
        except requests.RequestException as retry_exc:
            logger.error("❌ Scrape failed on retry: %s", retry_exc)
            raise RuntimeError("Scraping failed after retry") from retry_exc
    except Exception as exc:
        logger.error("❌ Scrape failed: %s", exc)
        raise RuntimeError("Scraping failed") from exc


def run_etl_cycle(days_ahead: int | None = None) -> str:
    cycle_start = time.time()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("🔄 ETL cycle started at %s", started_at)

    start_date = date.today()
    end_date = start_date + relativedelta(months=3) if days_ahead is None else start_date + relativedelta(days=days_ahead)
    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()

    events = _scrape_with_retry(start_date_str, end_date_str)
    logger.info(
        "📥 Scraped %d events (range: %s to %s)",
        len(events),
        start_date_str,
        end_date_str,
    )
    snapshot_path = _save_scrape_snapshot(events, start_date)

    try:
        load_summary = load_data(events) or {}
    except Exception as exc:
        logger.error("❌ Loader failed: %s", exc)
        raise RuntimeError("Loading failed") from exc

    inserted = int(load_summary.get("inserted", 0))
    updated = int(load_summary.get("updated", 0))
    skipped = int(load_summary.get("skipped", 0))
    stale_removed = int(load_summary.get("stale_removed", 0))

    logger.info(
        "✅ Loader complete: %d inserted, %d updated, %d skipped, %d stale removed",
        inserted,
        updated,
        skipped,
        stale_removed,
    )

    elapsed_seconds = time.time() - cycle_start
    logger.info("✅ ETL cycle completed in %.1f seconds", elapsed_seconds)

    return (
        f"ETL complete: scraped={len(events)}, inserted={inserted}, "
        f"updated={updated}, skipped={skipped}, stale_removed={stale_removed}, "
        f"duration_seconds={elapsed_seconds:.1f}, snapshot={snapshot_path}"
    )


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run one-shot ETL cycle")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Override date range to N days ahead (default: 3 months)",
    )
    args = parser.parse_args()

    try:
        summary = run_etl_cycle(days_ahead=args.days)
        logger.info(summary)
    except Exception as exc:
        logger.exception("❌ ETL cycle failed: %s", exc)
        sys.exit(1)