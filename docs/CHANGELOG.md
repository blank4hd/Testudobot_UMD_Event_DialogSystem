# Changelog

All notable changes to the TestudoBot UMD Event Dialog System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.5.0] - 2026-03-05

### Added — Evaluation Pipeline

- **RAGAS evaluation core (`scripts/evaluation.py`)** — New module implementing the full RAGAS evaluation workflow. Defines `EvalSample` dataclass for typed eval samples, `load_eval_samples()` to read from `eval/dataset.json` (with hardcoded fallback), and async `run_ragas_evaluation()` that retrieves context, generates answers, and computes RAGAS metrics (context_precision, context_recall, faithfulness, answer_relevancy). Includes `persist_run()` to save timestamped result JSON files and `summarize_results()` for aggregate metric reporting.
- **CLI evaluation runner (`scripts/evaluate.py`)** — Standalone CLI script for running RAGAS evaluations end-to-end. Supports `--dataset`, `--tag`, `--output-dir`, `--answer-model`, `--judge-model`, `--top-k`, `--limit`, and `--delay` arguments. Outputs timestamped JSON results to `eval/results/`.
- **Evaluation comparison tool (`scripts/compare_evals.py`)** — CLI tool to compare two RAGAS run files side-by-side. Shows aggregate metric deltas, config diffs, and per-sample regressions above a configurable `--threshold` (default: 0.10).
- **LLM-powered eval dataset generator (`scripts/generate_eval_dataset.py`)** — Uses LLM to generate candidate evaluation questions from event data across 10 categories (career, music, sports, food, academic, social, location, temporal, negation, multi-constraint). Supports `--input`, `--output`, `--model`, `--per-category`, and `--delay` CLI args.
- **Curated evaluation dataset (`eval/dataset.json`)** — 25 hand-reviewed QA samples (v2.0) with event-grounded ground truths across 13 categories (career, music, social, sports, academic, food, location, temporal, negation, culture, staff, performance, wellness).
- **`/test` chat command (`app.py`)** — Inline RAGAS evaluation from the Chainlit chat UI. Runs a quick smoke check using the shared evaluation pipeline with a small sample set and reports metrics directly in chat.
- **Smoke test for query expansion (`scripts/smoke_expand_query.py`)** — Unit-style smoke test that validates `expand_query()` success and fallback paths using mocked LLM responses.

### Added — UI Enhancements

- **Quick-action buttons** — Chat start screen now includes preset query buttons: 🍕 Free Food, 💼 Career Fairs, 🎵 Music, 🐢 Sports, and 🔄 Refresh Events for one-click common queries.
- **Streaming LLM responses** — RAG answers now stream token-by-token using `stream=True` with Groq API and Chainlit's `msg.stream_token()`, improving perceived response latency.

### Added — Evaluation Infrastructure

- **`eval/` directory** — New directory for evaluation artifacts: `dataset.json` (curated samples), `dataset_candidates.json` (LLM-generated candidate placeholder), and `results/` (timestamped RAGAS run outputs).
- **Baseline evaluation result** — `eval/results/2026-03-06T02-24-25Z_baseline.json` capturing initial RAGAS scores (context_precision: 1.0, context_recall: 0.0, faithfulness: 0.875, answer_relevancy: 0.887) on 5 samples.

---

## [0.4.0] - 2026-02-25

### Added — Data Pipeline & Incremental Ingestion

- **Rolling-range scraper upgrade (`scrape.py`)** — Scraper now iterates day-by-day through a configurable date range (default: today → 3 months ahead) instead of a single hardcoded month. Added CLI arguments (`--start-date`, `--end-date`, `--output`, `--use-cloudscraper`), day-page pagination, updated listing-page selectors with fallbacks, `normalize_event()` helper (whitespace stripping, date normalization, HTML tag removal), browser-like request headers, session warm-up, retry logic (3 attempts / 5 s delay), and structured logging. Added importable `scrape_events(start_date, end_date) -> list[dict]` API for use by `etl.py`. Invalid date inputs exit with code 2 and a friendly message.
- **On-demand ETL script (`etl.py`)** — New standalone script that orchestrates the full scrape → load pipeline: computes date range, calls `scrape_events()`, passes results to `load_data()`, and reports a summary with timing. Supports `--days N` CLI flag for short test windows. Scrape failures retry once after 60 seconds. Can be invoked via `python scripts/etl.py`, `docker compose run --rm etl`, or the `/refresh` chat command.
- **ETL quick-check diagnostics (`etl_quick_check.py`)** — Lightweight health check script that tests Postgres connectivity, Elasticsearch ping, and optionally runs a minimal scrape timing baseline (`--scrape-days N`).
- **`/refresh` chat command and quick-action button (`app.py`)** — Users can type `/refresh` in chat or click the 🔄 Refresh Events button to trigger an on-demand ETL cycle from within the Chainlit UI. After scraping, the topic modeling pipeline also re-runs for new events.
- **ETL service in Docker Compose** — Added `etl` service with `profiles: ["etl"]` so it does not start with `docker compose up`. Run on-demand via `docker compose run --rm etl`. Depends on `db` and `elasticsearch` health checks.

### Changed — Loader Incremental Upsert (`loader.py`)

- **Incremental upsert with deduplication** — Replaced the destructive truncate-and-reload with content-hash-based upsert logic. New `compute_event_hash()` generates a SHA-256 fingerprint from title + date + time + url. New `upsert_events()` inserts new events, updates changed events (description/location), and skips unchanged events. New `remove_stale_events()` deletes events no longer in the current feed.
- **`content_hash` column** — Added `content_hash TEXT UNIQUE` column to the `umd_events` PostgreSQL table for deduplication.
- **`load_data()` returns stats** — Function signature changed to `load_data(events: list[dict] | None = None)` and now returns a summary dict (`inserted`, `updated`, `skipped`, `stale_removed`). If events is `None`, loads from the JSON seed file.
- **`FORCE_CLEAN_SCHEMA` default changed to `false`** — Loader service in Docker Compose now defaults to incremental upserts. Set to `true` to restore the legacy truncate-and-reload behavior.
- **Dual environment variable support** — Postgres and Elasticsearch connection settings now resolve `POSTGRES_*` first, then fall back to `DB_*`, then to sensible defaults (host defaults to `localhost` instead of `db`). This allows `loader.py` and `etl.py` to work both inside Docker and locally.

### Changed — Dependencies

- **`cloudscraper==1.2.71`** — Added to `requirements.txt` for optional anti-bot bypass mode (`--use-cloudscraper` flag).

---

## [0.3.0] - 2026-02-25

### Changed — Repository Organization

- **Organized project layout** — Grouped operational scripts under `scripts/`, dataset snapshots under `data/`, and secondary documentation under `docs/`.
- **Updated runtime module paths** — Adjusted imports and execution paths so `/refresh`, ETL jobs, and diagnostics still work after file moves.
- **Updated container commands** — Docker Compose now runs `scripts/loader.py` and `scripts/etl.py`, and loader defaults to `data/` seed JSON.
- **Documentation sync** — Refreshed architecture and structure references in `README.md`, and moved changelog/roadmap into `docs/`.

---

## [0.2.0] - 2025-02-24

### Added — Retrieval & Search Quality

- **Cross-encoder re-ranker** — Integrated `cross-encoder/ms-marco-MiniLM-L-6-v2` as a second-stage reranker after RRF fusion. The cross-encoder rescores the top-K candidates using fine-grained query–document relevance, with a graceful fallback to RRF-only ranking on failure.
- **LLM-powered query expansion** — Added `expand_query()` which uses a fast LLM (`llama-3.1-8b-instant`) to rewrite natural-language user input into concise, search-optimized queries before retrieval. Handles conversational filler removal, keyword extraction, and relative date preservation.
- **Upgraded embedding model** — Replaced `all-MiniLM-L6-v2` (384-dim) with `all-mpnet-base-v2` (768-dim) for higher-quality semantic embeddings and improved retrieval accuracy.
- **Tuned BM25 / vector weight blending** — Moved from equal-weight RRF fusion to a 60/40 vector/keyword split (`vector_weight=0.6`, `keyword_weight=0.4`), reflecting that semantic similarity captures user intent better for natural-language queries while keyword search still helps with exact event name matches.

---

## [0.1.0] - Initial Release

### Added

- Event scraper (`scripts/scrape.py`) for `calendar.umd.edu` producing static JSON output.
- PostgreSQL + pgvector storage for event data and embeddings.
- Elasticsearch indexing with BM25 and dense-vector hybrid search using RRF fusion.
- LLM-based topic modeling pipeline for automatic event categorization.
- Chainlit-based conversational UI with RAG-powered event Q&A.
- RAGAS evaluation framework with context precision, recall, faithfulness, and answer relevancy metrics.
- Docker Compose setup for local development (PostgreSQL, Elasticsearch, app).
