# app.py
import os
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
import chainlit as cl

from db import db_pool, get_db_cursor, init_db, fetch_topic_map
from search import expand_query, search_events, extract_date_range, init_search
from pipeline import run_pipeline_if_needed, init_pipeline
from scripts.evaluation import load_eval_samples, run_ragas_evaluation

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "rag_system.txt"
try:
    SYSTEM_PROMPT_TEMPLATE = PROMPT_TEMPLATE_PATH.read_text()
except FileNotFoundError as exc:
    raise RuntimeError(f"Missing required prompt template: {PROMPT_TEMPLATE_PATH}") from exc

# --- LOGGING & CONFIG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
os.environ['PYTORCH_JIT_LOG_LEVEL'] = '0'

def resolve_elastic_host() -> str:
    configured_host = os.getenv("ELASTIC_HOST")
    candidates = [configured_host] if configured_host else []
    candidates.extend(["http://localhost:9200", "http://elasticsearch:9200"])

    seen = set()
    for host in candidates:
        if not host or host in seen:
            continue
        seen.add(host)
        try:
            if Elasticsearch(host).ping():
                logger.info("✅ Using Elasticsearch host: %s", host)
                return host
        except Exception:
            logger.warning("Elasticsearch host not reachable: %s", host)

    fallback = configured_host or "http://localhost:9200"
    logger.warning("⚠️ Falling back to Elasticsearch host: %s", fallback)
    return fallback


ELASTIC_HOST = resolve_elastic_host()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is required. Get one at https://console.groq.com")

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

init_search(es_client, embedding_model, cross_encoder, llm_client, LABEL_MODEL)
init_pipeline(es_client, llm_client, LABEL_MODEL, get_db_cursor)

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
    cl.user_session.set("settings", {"topic_filter": "All Topics", "top_k": 20})
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
            logger.exception("ETL refresh failed")
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
    retrieval_query = original_query
    followup_markers = [
        "which of those", "which of these", "those", "these", "them", "ones", "that one", "what about"
    ]
    original_query_lower = original_query.lower()
    last_user_query = history[-1]["user"] if history and "user" in history[-1] else ""
    if last_user_query and any(marker in original_query_lower for marker in followup_markers):
        retrieval_query = f"{last_user_query}. {original_query}"

    date_range = extract_date_range(retrieval_query)
    expanded_query = expand_query(retrieval_query)
    logger.info(f"🔍 Query expanded: '{retrieval_query}' -> '{expanded_query}'")

    filter_id = int(topic_map[selected_label]) if (selected_label != "All Topics" and selected_label in topic_map) else None

    # Search
    async with cl.Step(name="Searching UMD Events...", type="tool") as step:
        step.input = retrieval_query
        results = search_events(
            expanded_query,
            top_k=top_k,
            filter_topic_id=filter_id,
            override_date_range=date_range,
        )
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
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        date_str=current_date.strftime("%B %Y"),
        current_year=current_date.year,
        topic_list=topic_list,
        max_results=max_results,
        context_text=context_text,
    )

    msg = cl.Message(content="")
    await msg.send()

    # Build messages with recent history for multi-turn context
    messages = [{"role": "system", "content": sys_prompt}]
    for turn in history[-3:]:  # Last 3 turns to stay within context limits
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})
    messages.append({"role": "user", "content": original_query})

    stream = llm_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
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