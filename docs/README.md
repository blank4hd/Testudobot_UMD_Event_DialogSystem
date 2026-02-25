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

- [../scripts](../scripts): Operational scripts (`etl`, `etl_quick_check`, loader, scraper, diagnostics).
- [../data](../data): Seed/snapshot JSON datasets used by loader and ETL workflows.
- [../docs](.): Project documentation and planning artifacts.

### Key scripts

| Script                       | Purpose                                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `scripts/scrape.py`          | Rolling-range scraper for `calendar.umd.edu` with CLI args, pagination, and normalization. Exports `scrape_events()` API. |
| `scripts/loader.py`          | Incremental upsert loader: inserts new events, updates changed, removes stale. Exports `load_data()`.                     |
| `scripts/etl.py`             | On-demand ETL orchestrator: scrape → load → summary. CLI: `--days N`.                                                     |
| `scripts/etl_quick_check.py` | Lightweight Postgres/Elasticsearch connectivity diagnostic. CLI: `--scrape-days N`.                                       |
