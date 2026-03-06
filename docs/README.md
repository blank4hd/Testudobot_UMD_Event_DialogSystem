# Documentation Index

This folder contains project-level documentation beyond the main setup guide.

## Files

- [CHANGELOG.md](CHANGELOG.md): Versioned history of completed changes.
- [ROADMAP.md](ROADMAP.md): Planned improvements and future priorities.

## Where to start

- For setup and running the project, see [../README.md](../README.md).
- For release history, see [CHANGELOG.md](CHANGELOG.md).
- For next milestones, see [ROADMAP.md](ROADMAP.md).

## Repository conventions (current)

- [../scripts](../scripts): Operational scripts (`etl`, `etl_quick_check`, loader, scraper, diagnostics). `app.py` now delegates to `db.py` (database), `search.py` (retrieval), and `pipeline.py` (topic modeling).
- [../data](../data): Seed/snapshot JSON datasets used by loader and ETL workflows.
- [../docs](.): Project documentation and planning artifacts.

### Key scripts

| Script                             | Purpose                                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `scripts/scrape.py`                | Rolling-range scraper for `calendar.umd.edu` with CLI args, pagination, and normalization. Exports `scrape_events()` API. |
| `scripts/loader.py`                | Incremental upsert loader: inserts new events, updates changed, removes stale. Exports `load_data()`.                     |
| `scripts/etl.py`                   | On-demand ETL orchestrator: scrape → load → summary. CLI: `--days N`.                                                     |
| `scripts/etl_quick_check.py`       | Lightweight Postgres/Elasticsearch connectivity diagnostic. CLI: `--scrape-days N`.                                       |
| `scripts/evaluation.py`            | RAGAS evaluation core: `EvalSample` dataclass, async runner, result persistence. Exports `run_ragas_evaluation()`.        |
| `scripts/evaluate.py`              | CLI evaluation runner: `--dataset`, `--tag`, `--limit`, `--delay`, etc.                                                   |
| `scripts/compare_evals.py`         | Compares two RAGAS run JSON files: aggregate deltas, config diffs, per-sample regressions.                                |
| `scripts/generate_eval_dataset.py` | LLM-powered eval question generator across 10 event categories.                                                           |
| `scripts/smoke_expand_query.py`    | Smoke test for `expand_query()` success and fallback paths.                                                               |
| `tests/test_date_parsing.py`       | Pytest suite for date extraction: today, tomorrow, weekend, months, year rollover.                                        |
