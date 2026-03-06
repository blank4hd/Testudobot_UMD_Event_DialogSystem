import json
import logging

import numpy as np

from db import fetch_topic_map

logger = logging.getLogger(__name__)

MIN_TOPIC_SIZE = 3

es_client = None
llm_client = None
LABEL_MODEL = None
get_db_cursor = None


def init_pipeline(es, llm, label_model, db_cursor_getter):
    global es_client, llm_client, LABEL_MODEL, get_db_cursor
    es_client = es
    llm_client = llm
    LABEL_MODEL = label_model
    get_db_cursor = db_cursor_getter


def run_pipeline_if_needed():
    """Checks if we need to run the categorization pipeline."""
    if get_db_cursor is None:
        raise RuntimeError("Pipeline is not initialized. Call init_pipeline first.")

    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM umd_events WHERE topic_id IS NULL OR topic_id = -1;")
            uncategorized_count = cur.fetchone()[0]
    except Exception:
        uncategorized_count = 0

    topic_map = fetch_topic_map()

    if not topic_map or uncategorized_count > 0:
        logger.info("⚠️ Found %s uncategorized events or no topics. Running Pipeline...", uncategorized_count)
        return run_topic_modeling_pipeline()

    logger.info("✅ Topics exist and data is categorized. Skipping pipeline.")
    return False


def run_topic_modeling_pipeline():
    """
    Runs BERTopic -> Updates Postgres -> Syncs Elasticsearch -> Generates Labels
    """
    if get_db_cursor is None:
        raise RuntimeError("Pipeline is not initialized. Call init_pipeline first.")

    from bertopic import BERTopic
    from psycopg2.extras import DictCursor

    logger.info("🚀 Starting Topic Modeling Pipeline...")

    events = []
    embeddings = []

    with get_db_cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT id, event, description, embedding FROM umd_events")
        rows = cur.fetchall()
        for r in rows:
            if r["embedding"] is None:
                continue
            emb = np.array(json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"])
            if np.count_nonzero(emb) == 0:
                continue
            events.append(dict(r))
            embeddings.append(emb)

    if len(events) < 5:
        logger.warning("⚠️ Not enough data to run topic modeling (<5 events).")
        return "Not enough data."

    logger.info("📊 Running BERTopic on %s events...", len(events))
    docs = [f"{ev.get('event', '')} {ev.get('description', '')}" for ev in events]
    embeddings_np = np.stack(embeddings)

    norm = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    embeddings_np = embeddings_np / norm

    topic_model = BERTopic(min_topic_size=MIN_TOPIC_SIZE, verbose=True)
    topics, _ = topic_model.fit_transform(docs, embeddings_np)

    logger.info("💾 Saving topic assignments to Postgres...")
    with get_db_cursor() as cur:
        for ev, tid in zip(events, topics):
            cur.execute("UPDATE umd_events SET topic_id = %s WHERE id = %s", (int(tid), ev["id"]))

    logger.info("🔄 Syncing topics to Elasticsearch...")
    success_count = 0
    for ev, tid in zip(events, topics):
        q = {
            "query": {"match_phrase": {"event": ev["event"]}},
            "script": {"source": "ctx._source.topic_id = params.tid", "params": {"tid": int(tid)}},
        }
        try:
            es_client.update_by_query(index="umd_events", body=q, conflicts="proceed")
            success_count += 1
        except Exception as e:
            logger.warning("Failed to sync event %s to Elastic: %s", ev["id"], e)

    logger.info("🏷️ Generating Topic Labels (AI)...")
    topic_info = topic_model.get_topic_info()
    labels_to_save = []

    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue

        keywords = [x[0] for x in topic_model.get_topic(tid)[:5]]

        try:
            prompt = f"Keywords: {', '.join(keywords)}. Provide a concise category name (max 3 words). Return ONLY the name."
            resp = llm_client.chat.completions.create(
                model=LABEL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15,
            )
            label = resp.choices[0].message.content.strip().replace('"', "")
        except Exception as e:
            logger.warning("Label generation failed for topic %s: %s", tid, e)
            label = f"Topic {tid}"

        labels_to_save.append((int(tid), label, ", ".join(keywords)))

    logger.info("💾 Saving labels to Database...")
    with get_db_cursor() as cur:
        cur.execute("TRUNCATE TABLE topic_labels")
        for tid, label, kw in labels_to_save:
            cur.execute(
                "INSERT INTO topic_labels (topic_id, label, keywords) VALUES (%s, %s, %s)",
                (tid, label, kw),
            )

    logger.info("✅ Pipeline Complete. Synced %s events.", success_count)
    return f"Pipeline Complete. Synced {success_count} events to Elastic."
