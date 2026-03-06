# app.py
import os
import json
import time
import asyncio
import logging
from pathlib import Path
import chainlit as cl
from datetime import datetime
from typing import List, Dict
import numpy as np 
import psycopg2
from sentence_transformers import SentenceTransformer, CrossEncoder
from psycopg2 import pool
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from openai import OpenAI
from dateutil import parser  # ADD: pip install python-dateutil (for date parsing)
from dateutil.relativedelta import relativedelta

from datetime import datetime
from dateutil.relativedelta import relativedelta # Ensure you installed python-dateutil

import re
from dateutil import parser
from datetime import timedelta
from contextlib import contextmanager
from scripts.evaluation import load_eval_samples, run_ragas_evaluation

# --- LOGGING & CONFIG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
os.environ['PYTORCH_JIT_LOG_LEVEL'] = '0'

DB_NAME = os.getenv("DB_NAME", "umd_events")
DB_USER = os.getenv("DB_USER", "umd_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "umd_password")
DB_HOST = os.getenv("DB_HOST", "db")
ELASTIC_HOST = os.getenv("ELASTIC_HOST", "http://elasticsearch:9200")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# 1. Model for simple tasks (Topic Labeling) - Faster & Cheaper
LABEL_MODEL = "llama-3.1-8b-instant"

# 2. Model for complex tasks (Chat & RAG) - Smarter & More Detailed
CHAT_MODEL = "llama-3.3-70b-versatile"
LLM_MODEL = "llama-3.3-70b-versatile"

# --- GLOBAL CLIENTS ---
es_client = Elasticsearch(ELASTIC_HOST)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

# ============================
#  DATABASE & SEARCH LOGIC
# ============================

# 1. Create a Global Connection Pool
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(
        1, 20,
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port="5432"
    )
    if db_pool:
        logger.info("✅ Database connection pool created successfully")
except Exception as e:
    logger.error(f"❌ Error creating connection pool: {e}")

# 2. Helper to get a cursor
@contextmanager
def get_db_cursor(cursor_factory=None):
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS topic_labels (
                    topic_id INTEGER PRIMARY KEY,
                    label TEXT,
                    keywords TEXT
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS umd_events_embedding_idx 
                ON umd_events USING hnsw (embedding vector_cosine_ops);
            """)
            logger.info("✅ Database initialized.")
    except Exception as e:
        logger.error(f"❌ DB Init Error: {e}")
    

def fetch_topic_map() -> Dict[str, str]:
    """Returns { 'Career': '1', 'Music': '2' } for the dropdown."""
    try:
        with get_db_cursor() as cur:
            # Check if table exists and has data
            cur.execute("SELECT COUNT(*) FROM topic_labels;")
            count = cur.fetchone()[0]
            logger.info(f"🔍 DEBUG: 'topic_labels' table contains {count} rows.")

            cur.execute("SELECT topic_id, label FROM topic_labels WHERE topic_id != -1 ORDER BY label;")
            rows = cur.fetchall()
            
            result = {row[1]: str(row[0]) for row in rows}
            logger.info(f"🔍 DEBUG: Topic Map returning: {result}")
            return result
    except Exception as e:
        logger.error(f"❌ ERROR in fetch_topic_map: {e}")
        return {}




# ------------------------------------------------------------------
# ROBUST HYBRID SEARCH  (drop-in replacement for search_events)
# ------------------------------------------------------------------
 

def expand_query(user_query: str) -> str:
    """Uses a fast LLM to rewrite user input into an optimized search query."""
    try:
        system_prompt = """You are a search query optimizer for a university events database.
Given a user's natural language question, rewrite it into a concise search query
that captures the key intent. Extract the core topic/event type the user is looking for.

Rules:
- Return ONLY the optimized search query, nothing else
- Remove conversational filler words (hey, can you, I want, please, etc.)
- Keep important keywords: event types, topics, locations, dates
- If the query mentions relative dates like "this weekend" or "tomorrow", keep those as-is
- Keep it under 10 words

Examples:
User: "hey is there anything fun happening this weekend?" -> "entertainment social events this weekend"
User: "I'm looking for career related stuff" -> "career fairs job workshops"
User: "any free food events on campus?" -> "free food events campus"
User: "what music concerts are coming up?" -> "music concerts performances upcoming"""  # noqa: E501

        resp = llm_client.chat.completions.create(
            model=LABEL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0,
            max_tokens=30,
        )
        expanded = (resp.choices[0].message.content or "").strip()
        return expanded if expanded else user_query
    except Exception as e:
        logger.warning(f"Query expansion failed, using original query: {e}")
        return user_query


import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def search_events(
    query_text: str,
    top_k: int = 15,
    filter_topic_id: int | None = None,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> list[tuple[dict, float]]:
    """
    ULTIMATE HYBRID SEARCH (Fixed for 'Upcoming' Events):
    1. Parsing: Handles 'upcoming', 'next week', 'this month', and specific dates.
    2. Retrieval: Runs Vector (Semantic) and Keyword (BM25) searches.
    3. Fusion: Uses RRF for robust ranking.
    """
    current = datetime.now()
    q_lower = query_text.lower()
    cleaned_query = query_text
    start_date, end_date = None, None

    # --- 1. ROBUST DATE PARSING ---
    
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }

    # A. Specific Date: "November 13"
    specific_date_match = re.search(r'\b(' + '|'.join(months.keys()) + r')\s+(\d{1,2})\b', q_lower)
    
    if specific_date_match:
        m_name = specific_date_match.group(1)
        day_num = int(specific_date_match.group(2))
        m_num = months[m_name]
        year = current.year 
        try:
            target_date = datetime(year, m_num, day_num)
            start_date = target_date.strftime("%Y-%m-%d")
            end_date = target_date.strftime("%Y-%m-%d") 
            cleaned_query = re.sub(rf"\b{m_name}\s+{day_num}\b", "", cleaned_query, flags=re.I)
        except ValueError:
            logger.warning(f"Invalid date parsed: {m_name} {day_num}")

    # B. "Upcoming" / "Future" (The Fix!)
    # Sets filter from TODAY -> 1 Year from now
    elif "upcoming" in q_lower or "future" in q_lower or "soon" in q_lower:
        start_date = current.strftime("%Y-%m-%d")
        end_date = (current + relativedelta(years=1)).strftime("%Y-%m-%d")
        cleaned_query = re.sub(r"\b(upcoming|future|soon)\b", "", cleaned_query, flags=re.I)

    # C. Relative Dates
    elif "this month" in q_lower:
        start_date = current.replace(day=1).strftime("%Y-%m-%d")
        end_date = (current + relativedelta(months=1, days=-1)).strftime("%Y-%m-%d")
        cleaned_query = re.sub(r"\bthis month\b", "", cleaned_query, flags=re.I)
    elif "next month" in q_lower:
        start = (current + relativedelta(months=1)).replace(day=1)
        start_date = start.strftime("%Y-%m-%d")
        end_date = (start + relativedelta(months=1, days=-1)).strftime("%Y-%m-%d")
        cleaned_query = re.sub(r"\bnext month\b", "", cleaned_query, flags=re.I)
    elif "next week" in q_lower:
        start_date = current.strftime("%Y-%m-%d")
        end_date = (current + timedelta(days=7)).strftime("%Y-%m-%d")
        cleaned_query = re.sub(r"\bnext week\b", "", cleaned_query, flags=re.I)

    # D. Whole Month: "In October"
    else:
        for m_name, m_num in months.items():
            if m_name in q_lower:
                year = current.year
                dt = datetime(year, m_num, 1)
                start_date = dt.strftime("%Y-%m-%d")
                end_date = (dt + relativedelta(months=1, days=-1)).strftime("%Y-%m-%d")
                cleaned_query = re.sub(rf"\b{m_name}\b", "", cleaned_query, flags=re.I)
                break

    # Strip filler words
    cleaned_query = re.sub(r"\b(events?|happening|show me|find|list|from|on|in)\b", "", cleaned_query, flags=re.I).strip()
    
    # If query became empty (e.g. "upcoming events" -> ""), reset it to find *anything* in range
    if not cleaned_query: 
        cleaned_query = query_text 

    # --- 2. BUILD FILTERS ---
    es_filters = []
    if filter_topic_id is not None:
        es_filters.append({"term": {"topic_id": filter_topic_id}})
    if start_date and end_date:
        es_filters.append({"range": {"date": {"gte": start_date, "lte": end_date, "format": "yyyy-MM-dd"}}})
    
    filter_query = {"bool": {"filter": es_filters}} if es_filters else None

    # --- 3. VECTOR SEARCH ---
    try:
        vector_body = {
            "size": top_k,
            "knn": {
                "field": "embedding",
                "query_vector": embedding_model.encode(cleaned_query, normalize_embeddings=True).tolist(),
                "k": top_k,
                "num_candidates": 200,
            },
            "_source": ["event", "description", "date", "time", "location", "url"]
        }
        if filter_query: vector_body["knn"]["filter"] = filter_query
        v_res = es_client.search(index="umd_events", body=vector_body)["hits"]["hits"]
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        v_res = []

    # --- 4. KEYWORD SEARCH ---
    try:
        keyword_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": cleaned_query,
                            "fields": ["event^3", "description^2", "location"],
                            "fuzziness": "AUTO",
                            "operator": "or"
                        }
                    },
                    "filter": es_filters
                }
            },
            "_source": ["event", "description", "date", "time", "location", "url"]
        }
        k_res = es_client.search(index="umd_events", body=keyword_body)["hits"]["hits"]
    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        k_res = []

    # --- 5. RRF FUSION ---
    def rrf_score(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    # Vector search is weighted higher (0.6) because semantic similarity captures
    # user intent better than exact keyword matching for natural language queries,
    # while keyword search still helps with exact event name matches.
    fused = {} 
    for r, hit in enumerate(v_res):
        _id = hit["_id"]
        if _id not in fused: fused[_id] = {"doc": hit["_source"], "score": 0.0}
        fused[_id]["score"] += vector_weight * rrf_score(r)

    for r, hit in enumerate(k_res):
        _id = hit["_id"]
        if _id not in fused: fused[_id] = {"doc": hit["_source"], "score": 0.0}
        fused[_id]["score"] += keyword_weight * rrf_score(r)

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    try:
        if ranked:
            pairs = []
            for item in ranked:
                doc = item.get("doc", {})
                title = doc.get("event", "") or ""
                description = doc.get("description", "") or ""
                document_text = f"{title} {description}".strip()
                pairs.append([query_text, document_text])

            ce_scores = cross_encoder.predict(pairs)
            for item, ce_score in zip(ranked, ce_scores):
                item["score"] = float(ce_score)

            ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)[:top_k]
    except Exception as e:
        logger.warning(f"Cross-encoder reranking failed; falling back to RRF ranking: {e}")

    return [(item["doc"], item["score"]) for item in ranked]
# --- PIPELINE & ADMIN TASKS ---

def run_pipeline_if_needed():
    """Checks if we need to run the categorization pipeline."""
    # Check for uncategorized events
    try:
        with get_db_cursor() as cur:
            # Count events that are either NULL or -1 (uncategorized)
            cur.execute("SELECT COUNT(*) FROM umd_events WHERE topic_id IS NULL OR topic_id = -1;")
            uncategorized_count = cur.fetchone()[0]
    except Exception:
        uncategorized_count = 0

    topic_map = fetch_topic_map()
    
    # Run if topics are empty OR we have new uncategorized data
    if not topic_map or uncategorized_count > 0:
        logger.info(f"⚠️ Found {uncategorized_count} uncategorized events or no topics. Running Pipeline...")
        return run_topic_modeling_pipeline()
        
    logger.info("✅ Topics exist and data is categorized. Skipping pipeline.")
    return False

def run_topic_modeling_pipeline():
    """
    Runs BERTopic -> Updates Postgres -> Syncs Elasticsearch -> Generates Labels
    """
    from bertopic import BERTopic
    from psycopg2.extras import DictCursor

    logger.info("🚀 Starting Topic Modeling Pipeline...")

    # 1. Fetch Data
    events = []
    embeddings = []
    
    with get_db_cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT id, event, description, embedding FROM umd_events")
        rows = cur.fetchall()
        for r in rows:
            if r['embedding'] is None: continue
            emb = np.array(json.loads(r['embedding']) if isinstance(r['embedding'], str) else r['embedding'])
            if np.count_nonzero(emb) == 0: continue
            events.append(dict(r))
            embeddings.append(emb)

    if len(events) < 5: 
        logger.warning("⚠️ Not enough data to run topic modeling (<5 events).")
        return "Not enough data."
    
    # 2. Run BERTopic
    logger.info(f"📊 Running BERTopic on {len(events)} events...")
    docs = [f"{ev.get('event','')} {ev.get('description','')}" for ev in events]
    embeddings_np = np.stack(embeddings)
    
    norm = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    norm[norm==0] = 1.0
    embeddings_np = embeddings_np / norm

    topic_model = BERTopic(min_topic_size=3, verbose=True)
    topics, _ = topic_model.fit_transform(docs, embeddings_np)

    # 3. Update Postgres
    logger.info("💾 Saving topic assignments to Postgres...")
    with get_db_cursor() as cur:
        for ev, tid in zip(events, topics):
            cur.execute("UPDATE umd_events SET topic_id = %s WHERE id = %s", (int(tid), ev['id']))

    # 4. Sync Elasticsearch
    logger.info("🔄 Syncing topics to Elasticsearch...")
    success_count = 0
    for ev, tid in zip(events, topics):
        q = {
            "query": { "match_phrase": { "event": ev['event'] } },
            "script": { "source": "ctx._source.topic_id = params.tid", "params": {"tid": int(tid)} }
        }
        try:
            es_client.update_by_query(index="umd_events", body=q, conflicts="proceed")
            success_count += 1
        except Exception as e:
            logger.warning(f"Failed to sync event {ev['id']} to Elastic: {e}")

    # 5. Generate Labels (Calculated IN MEMORY first to avoid DB Locks)
    logger.info("🏷️ Generating Topic Labels (AI)...")
    topic_info = topic_model.get_topic_info()
    labels_to_save = [] # Store them here first

    for _, row in topic_info.iterrows():
        tid = row['Topic']
        if tid == -1: continue 
        
        keywords = [x[0] for x in topic_model.get_topic(tid)[:5]]
        
        try:
            # AI Call happens HERE (No DB connection held)
            prompt = f"Keywords: {', '.join(keywords)}. Provide a concise category name (max 3 words). Return ONLY the name."
            resp = llm_client.chat.completions.create(
                model=LABEL_MODEL,
                messages=[{"role":"user","content":prompt}],
                max_tokens=15
            )
            label = resp.choices[0].message.content.strip().replace('"','')
        except Exception as e:
            logger.error(f"Label generation failed for topic {tid}: {e}")
            label = f"Topic {tid}"
        
        # Add to list
        labels_to_save.append((int(tid), label, ", ".join(keywords)))

    # 6. Save Labels to DB (Fast Batch Insert)
    logger.info("💾 Saving labels to Database...")
    with get_db_cursor() as cur:
        cur.execute("TRUNCATE TABLE topic_labels")
        for tid, label, kw in labels_to_save:
            cur.execute("INSERT INTO topic_labels (topic_id, label, keywords) VALUES (%s, %s, %s)", 
                        (tid, label, kw))
    
    logger.info(f"✅ Pipeline Complete. Synced {success_count} events.")
    return f"Pipeline Complete. Synced {success_count} events to Elastic."

# ============================
#  CHAINLIT EVENT HANDLERS
# ============================

@cl.on_chat_start
async def start():
    init_db()
    
    start_msg = cl.Message(content="🐢 **TestudoBot is booting up...**")
    await start_msg.send()

    # Check for startup work
    with get_db_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM umd_events WHERE topic_id IS NULL OR topic_id = -1;")
        uncategorized_count = cur.fetchone()[0]

    if uncategorized_count > 100:
        start_msg.content = f"🐢 **TestudoBot is booting up...**\n\n⚠️ Found {uncategorized_count} new/uncategorized events.\nRunning AI categorization pipeline (this may take 30s)..."
        await start_msg.update()
        
        # Run pipeline
        await cl.make_async(run_pipeline_if_needed)()
        
        start_msg.content = "✅ **Optimization Complete!**\nLoading interface..."
        await start_msg.update()

    # Create Settings
    topic_map = fetch_topic_map()
    topic_labels = ["All Topics"] + list(topic_map.keys())
    
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="topic_filter",
                label="Filter by Topic",
                values=topic_labels,
                initial_value="All Topics"
            ),
            cl.input_widget.Slider(
                id="top_k",
                label="Max Results",
                initial=20,
                min=1,
                max=30,
                step=1
            ),
        ]
    ).send()
    
    cl.user_session.set("topic_map", topic_map)
    cl.user_session.set("settings", {"topic_filter": "All Topics", "top_k": 5})
    cl.user_session.set("history", [])

    # Quick Actions
    actions = [
        cl.Action(name="quick_search", value="Is there free food today?", label="🍕 Free Food"),
        cl.Action(name="quick_search", value="Career fairs this month", label="💼 Career Fairs"),
        cl.Action(name="quick_search", value="Music performances next week", label="🎵 Music"),
        cl.Action(name="quick_search", value="Sports games this weekend", label="🐢 Sports"),
        cl.Action(name="quick_search", value="/refresh", label="🔄 Refresh Events"),
    ]

    start_msg.content = f"✅ **Ready!** I know about {len(topic_map)} categories of events.\n\nClick a button or type a query to start!"
    start_msg.actions = actions
    await start_msg.update()

@cl.action_callback("quick_search")
async def on_action(action: cl.Action):
    await cl.Message(content=action.value, author="User").send()
    await main(cl.Message(content=action.value))

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)
    await cl.Message(content=f"⚙️ **Filter Updated:** {settings['topic_filter']}").send()

# ============================
#  RAGAS EVALUATION LOGIC
# ============================

async def run_quick_ragas_evaluation(sample_limit: int = 5):
    dataset_path = Path("eval/dataset.json")
    eval_samples = load_eval_samples(dataset_path, sample_limit=sample_limit)
    if not eval_samples:
        raise ValueError("No valid evaluation samples found.")

    return await run_ragas_evaluation(
        eval_samples=eval_samples,
        search_events_fn=search_events,
        llm_client=llm_client,
        answer_model=LLM_MODEL,
        judge_model=LABEL_MODEL,
        groq_api_key=GROQ_API_KEY,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        top_k=5,
        per_sample_delay_seconds=2.5,
    )

@cl.on_message
async def main(message: cl.Message):
    if message.content.strip() == "/refresh":
        await cl.Message(content="🔄 **Refreshing events...** This may take a minute.").send()
        try:
            from scripts.etl import run_etl_cycle
            summary = run_etl_cycle()
            pipeline_result = run_pipeline_if_needed()
            await cl.Message(content=f"✅ **Refresh complete!**\n{summary}\n🏷️ {pipeline_result}").send()
        except Exception as e:
            logger.error(f"ETL refresh failed: {e}")
            await cl.Message(content=f"❌ **Refresh failed:** {str(e)}").send()
        return

    if message.content.strip() == "/test":
        await cl.Message(content="📊 **Starting RAGAS Evaluation...** (Check terminal)").send()
        try:
            payload = await run_quick_ragas_evaluation(sample_limit=5)
            results = payload["results"]
            df = results.to_pandas()
            csv_file = cl.File(name="ragas_results.csv", content=df.to_csv().encode("utf-8"))

            scores = df.select_dtypes(include=[np.number]).mean().dropna()
            if scores.empty:
                await cl.Message(
                    content="❌ **Evaluation failed:** RAGAS returned no valid numeric scores. Check terminal logs for model/API errors.",
                    elements=[csv_file],
                ).send()
                return

            summary = "\n".join([f"- **{k}**: {v:.4f}" for k, v in scores.to_dict().items()])
            timing = payload.get("timing", {})
            time_info = f"\n\n⏱️ **Time:** {timing.get('total_formatted', 'N/A')} (generation {timing.get('generation_formatted', 'N/A')}, RAGAS judging {timing.get('ragas_judging_formatted', 'N/A')})"
            await cl.Message(content=f"✅ **Evaluation Complete!**\n\n{summary}{time_info}", elements=[csv_file]).send()
        except Exception as e:
            logger.exception("RAGAS evaluation failed")
            await cl.Message(content=f"❌ **Evaluation failed:** {str(e)}").send()
        return

    history = cl.user_session.get("history", [])
    settings = cl.user_session.get("settings")
    topic_map = cl.user_session.get("topic_map")
    
    selected_label = settings.get("topic_filter", "All Topics")
    top_k = int(settings.get("top_k", 5))
    max_results = top_k
    topic_list = ", ".join(topic_map.keys()) if topic_map else "General Events"
    original_query = message.content
    expanded_query = expand_query(original_query)
    logger.info(f"🔍 Query expanded: '{original_query}' -> '{expanded_query}'")

    filter_id = int(topic_map[selected_label]) if (selected_label != "All Topics" and selected_label in topic_map) else None

    # Search
    async with cl.Step(name="Searching UMD Events...", type="tool") as step:
        step.input = original_query
        results = search_events(expanded_query, top_k=top_k, filter_topic_id=filter_id)
        if results:
            step.output = "\n".join([f"- {ev.get('event','Unknown')} ({score:.2f})" for ev, score in results])
        else:
            step.output = "No matches found."

    # Build Context safely
    context_text = "\n\n".join([
        f"Event: {ev.get('event','N/A')}\nDate: {ev.get('date','N/A')}\nDesc: {(ev.get('description') or '')[:300]}"
        for ev, score in results
    ]) or "No events found."

    current_date = datetime.now()
    date_str = current_date.strftime("%B %Y")
    sys_prompt = f"""
    You are TestudoBot, a knowledgeable, friendly, and enthusiastic AI assistant for University of Maryland (UMD) events. Your goal: Help users discover lectures, career fairs, performances, workshops, and more – using ONLY the provided Context. Do not invent details, dates, or URLs. If Context lacks info, say so politely and suggest alternatives from data.

    Key Parameters:
    - Current Date: {date_str} (resolve relatives; for 'December', use Dec {current_date.year} unless specified).
    - Known Categories: {topic_list}
    - Max Results: {max_results} (list ALL relevant from Context, even if >{max_results}; prioritize & sort by date).

    Instructions:
    1. Think step-by-step: (a) Parse query for date (e.g., 'December' = Dec 1-31), topic ('basketball'), etc. (b) From Context, filter STRICTLY to matching date range (ignore non-Dec events). Sort by soonest date first. Include women's/men's variants. (c) Summarize top {max_results} (or all if fewer), even with 'N/A' desc—use event name for details.
    2. If few/no matches: 'I found X basketball events in December. For more, try specifying date/type.' Don't say 'couldn't find' if Context has any.
    3. If Context has non-matching dates (e.g., Nov/Jan), ignore them entirely.
    4. Maintain conversation...
    Output Rules:
    - Concise (<180 words), engaging, and polite academic tone.
    - Structure: Direct answer sentence (note date interpretation if key). Bullet list of top events:
    - **Event Name**
    - Date/Time
    - Location (if available)
    - Brief Description (≤20 words)
    - [Source: Event ID (URL if provided)]
    - Omit missing fields. End with a short follow-up if ambiguous (e.g., "What else interests you?").

    Examples:
    - Query: 'basketball in december' → 'Here are December basketball events: \n• **Men's vs. Wagner** \nDate/Time: Dec 2, 2025, 8pm \n... \n• **Men's vs. Old Dominion** \nDate/Time: Dec 28, 2025, 6pm \n... \nFound 3 total.'
    - Query: "Career events next week?" → "Upcoming Career Services events next week: \n• **Job Fair** \nDate/Time: Oct 10, 2-5pm \nLocation: Stamp Student Union \nDesc: Networking for students and grads. \n[Source: Event 123 (https://umd.edu/fair)]"
    - Query: "Virtual events tomorrow?" (None) → "No virtual events tomorrow, but here's a close in-person alternative on Oct 9. Interested in more options?"
    - Query: "Fun events this month?" (Broad) → "Diverse fun events this month: \n• **Concert Series** \nDate/Time: Oct 15, 7pm \nLocation: Clarice Smith Center \nDesc: Live music performances. \n[Source: Event 456] \nPrefer a specific genre?"

    User Query (with history): {original_query}
    Context: {context_text}"""

    msg = cl.Message(content="")
    await msg.send()

    stream = llm_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuery: {original_query}"}
        ],
        temperature=0.1,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            await msg.stream_token(token)

    # Attach Sources (Rich Cards)
    if results:
        elements = []
        for i, (ev, score) in enumerate(results[:5]):
            lines = [
                f"**📅 When:** {ev.get('date', 'N/A')}" + (f" at {ev['time']}" if ev.get('time') else ""),
                f"**📍 Where:** {ev.get('location', 'TBD')}"
            ]
            if ev.get('url'):
                lines.append(f"**🔗 Link:** [Official Event Page]({ev['url']})")
            
            desc = (ev.get('description') or '')[:350]
            if len(ev.get('description') or '') > 350: desc += "..."
            lines.append(f"\n**📝 Details:**\n{desc}")
            
            elements.append(cl.Text(name=f"{i+1}. {ev.get('event','Unknown')}", content="\n".join(lines), display="inline"))
        msg.elements = elements
    
    await msg.update()
    
    # Save to history with FULL response
    history.append({"user": original_query, "bot": full_response}) 
    cl.user_session.set("history", history)