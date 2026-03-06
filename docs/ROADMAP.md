# Roadmap

> Last updated: 2026-03-05

---

## ✅ Completed

### Evaluation Pipeline & UI (v0.5.0)

- [x] **CLI-first RAGAS evaluation workflow** — Added `scripts/evaluation.py` (core RAGAS runner with `EvalSample` dataclass, async evaluation, and result persistence) and `scripts/evaluate.py` (CLI runner with `--dataset`, `--tag`, `--limit`, `--delay` args). Results saved as timestamped JSON under `eval/results/`.
- [x] **Evaluation comparison tool** — Added `scripts/compare_evals.py` to diff two RAGAS runs: shows aggregate metric deltas, config diffs, and per-sample regressions.
- [x] **LLM-powered eval dataset generator** — Added `scripts/generate_eval_dataset.py` to auto-generate candidate QA pairs from event data across 10 categories.
- [x] **Curated 25-sample evaluation dataset** — Expanded from 3 hardcoded samples to 25 hand-reviewed, event-grounded QA pairs in `eval/dataset.json` (v2.0) across 13 categories.
- [x] **`/test` chat command** — Inline RAGAS smoke check from the Chainlit UI using the shared evaluation pipeline.
- [x] **Smoke test for query expansion** — Added `scripts/smoke_expand_query.py` to validate `expand_query()` success and fallback paths.
- [x] **Quick-action buttons** — Added preset query buttons (🍕 Free Food, 💼 Career Fairs, 🎵 Music, 🐢 Sports, 🔄 Refresh Events) to the chat start screen.
- [x] **Streaming LLM responses** — RAG answers now stream token-by-token via Groq API `stream=True` and Chainlit `msg.stream_token()`.

### Data Pipeline & Incremental Ingestion (v0.4.0)

- [x] **On-demand ETL pipeline** — Added `scripts/etl.py` as a standalone one-shot ETL script (scrape → upsert). Runnable via `python scripts/etl.py`, `docker compose run --rm etl`, or the `/refresh` chat command. No always-running scheduler required.
- [x] **Rolling date range scraping** — Scraper now iterates day-by-day through a configurable window (default: today → 3 months ahead) with CLI args (`--start-date`, `--end-date`, `--days`).
- [x] **Content-hash deduplication on ingestion** — Loader computes SHA-256 hashes from event fingerprints; inserts new events, updates changed events, skips unchanged, and removes stale events no longer in the feed.
- [x] **Incremental upsert loader** — Replaced destructive truncate-and-reload with `upsert_events()` + `remove_stale_events()`. `FORCE_CLEAN_SCHEMA=false` by default.
- [x] **`/refresh` command in Chainlit UI** — Chat command and quick-action button to trigger ETL from the UI.
- [x] **ETL Docker Compose service** — Added `etl` service with `profiles: ["etl"]` for on-demand runs.
- [x] **Quick diagnostics script** — `scripts/etl_quick_check.py` tests Postgres/Elasticsearch connectivity and optional scrape timing.
- [x] **Dual env-var support** — Loader and ETL resolve `POSTGRES_*` → `DB_*` → defaults, enabling both Docker and local execution.

### Repository Hygiene (v0.3.0)

- [x] **Project structure reorganization** — Moved operational scripts into `scripts/`, JSON snapshots into `data/`, and secondary docs into `docs/`.
- [x] **Path compatibility updates** — Updated imports, Docker commands, and loader path handling to preserve behavior after reorganization.

### Retrieval & Search Quality (v0.2.0)

- [x] **Cross-encoder re-ranker** — Integrated `cross-encoder/ms-marco-MiniLM-L-6-v2` as a second-stage reranker after RRF fusion, with graceful fallback.
- [x] **Upgraded embedding model** — Replaced `all-MiniLM-L6-v2` with `all-mpnet-base-v2` (768-dim) for stronger semantic matching.
- [x] **Query expansion / reformulation** — LLM-powered query rewriting via `expand_query()` converts natural-language input into optimized search queries before retrieval.
- [x] **Tuned BM25 and vector weight blending** — Moved from equal-weight RRF to a 60/40 vector/keyword split based on empirical tuning.

---

## 🔜 Planned

### Data Pipeline & Ingestion

- [ ] **Scrape additional data sources** — Pull from department-specific calendars, TerpLink student org events, and athletics schedules for broader coverage.
- [ ] **Scheduled ETL automation** — Add optional recurring scheduling (e.g., APScheduler or cron) on top of the existing on-demand ETL pipeline for fully unattended updates.

### Database & Storage

- [ ] **Remove database redundancy** — Consolidate storage: use Elasticsearch as the single search engine for both BM25 and dense vector retrieval, keeping PostgreSQL only for structured metadata (topic labels, admin state).
- [ ] **Proper date typing** — Convert dates from TEXT/keyword to native DATE types to enable range queries and eliminate regex-based date parsing in `search_events()`.
- [ ] **Event versioning/history** — Track when events are added, modified, or removed to support queries like "what events were recently added?"

### Evaluation & Quality

- [x] **Expand the RAGAS evaluation set** — Built 25-sample curated dataset (v2.0) covering career, music, social, sports, academic, food, location, temporal, negation, culture, staff, performance, and wellness categories.
- [ ] **Improve Answer Relevancy score** — Improve through better prompt engineering, context window management, or a stronger generation LLM.
- [ ] **Add automated CI evaluation** — Run RAGAS evaluations automatically on code changes to catch retrieval or generation regressions.

### Conversational Experience

- [ ] **Multi-turn conversation memory** — Add conversation history to the LLM context to enable follow-ups like "tell me more about that one" or "any others nearby?"
- [ ] **User preference tracking** — Remember user interests across sessions to personalize event recommendations.
- [x] **Streaming responses** — Implemented via Groq API `stream=True` and Chainlit `msg.stream_token()` for token-by-token display.
- [ ] **Better error handling and fallback responses** — When no events match, provide suggestions or ask clarifying questions instead of returning empty results.

### Infrastructure & DevOps

- [ ] **Reverse proxy (Nginx/Traefik)** — Add HTTPS termination, rate limiting, and proper routing for production deployment.
- [ ] **Health monitoring and alerting** — Add application-level health checks, logging aggregation, and alerts for scraping or database failures.
- [ ] **Environment-based configuration** — Separate dev/staging/production configs instead of relying on a single `.env` file.
- [ ] **Resource optimization** — Address the 512 MB Elasticsearch heap allocation and the duplicated embedding model loading; consider shared model serving or embedding caching.

### UI/UX Enhancements

- [ ] **Calendar view integration** — Add a visual calendar alongside the chat for browsing events by date.
- [ ] **Event bookmarking/reminders** — Let users save events and optionally receive reminders.
- [ ] **Rich event cards with images** — Scrape and display event images or flyers in chat responses.
- [ ] **Mobile-responsive design** — Ensure the Chainlit interface works well on mobile devices.

### Security & Robustness

- [ ] **API key management** — Move from `.env` files to a proper secrets manager (Docker secrets, Vault) for production.
- [ ] **Input sanitization** — Add guardrails against prompt injection in user queries passed to the LLM.
- [ ] **Rate limiting on the chat endpoint** — Prevent abuse of the LLM API quota.
