# app.py

import os
from typing import List, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

# Load .env when the container/app starts
load_dotenv()

# ============================
#  CONFIG (from env with defaults)
# ============================

DB_NAME = os.getenv("DB_NAME", "umd_events")
DB_USER = os.getenv("DB_USER", "umd_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "umd_password")
# default to "db" because that's the Docker service name
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # must be set (via .env/env vars)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"  # fast + decent quality
)


# ============================
#  LOW-LEVEL HELPERS
# ============================

def get_db_connection():
    """Create a new PostgreSQL connection."""
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    return conn


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformer model once and cache it."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Put it in your .env file, e.g.\n"
            "GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=GROQ_API_KEY)


def fetch_all_events() -> List[dict]:
    """Load ALL events from the umd_events table as dicts."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT id, event, date, time, url, location, description
                FROM umd_events
                ORDER BY id;
                """
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def build_event_text(ev: dict) -> str:
    """
    Create a single text string for embedding:
    combine title, date, location, description, etc.
    """
    parts = [
        ev.get("event") or "",
        ev.get("date") or "",
        ev.get("time") or "",
        ev.get("location") or "",
        ev.get("description") or "",
    ]
    return " | ".join(p.strip() for p in parts if p and p.strip())


@st.cache_resource(show_spinner="Building semantic index from database...")
def build_semantic_index() -> Tuple[List[dict], np.ndarray]:
    """
    1. Fetch all events from DB
    2. Build embeddings for each
    3. Return (events_list, embeddings_matrix)
    """
    events = fetch_all_events()
    if not events:
        return [], np.empty((0, 384), dtype="float32")

    model = get_embedding_model()
    texts = [build_event_text(ev) for ev in events]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")  # shape: (N, D)

    return events, embeddings


def semantic_search(query: str, top_k: int = 5) -> List[Tuple[dict, float]]:
    """
    Semantic search over all events using cosine similarity.
    Returns list of (event_dict, score), sorted by score desc.
    """
    events, emb_matrix = build_semantic_index()
    if len(events) == 0:
        return []

    model = get_embedding_model()
    q_emb = model.encode([query], normalize_embeddings=True)[0]  # (D,)
    q_emb = q_emb.astype("float32")

    # cosine similarity because all vectors are normalized
    scores = emb_matrix @ q_emb  # (N,)

    top_k = min(top_k, len(events))
    top_indices = np.argsort(-scores)[:top_k]

    results: List[Tuple[dict, float]] = []
    for idx in top_indices:
        results.append((events[idx], float(scores[idx])))

    return results


def format_events_for_context(events_with_scores: List[Tuple[dict, float]]) -> str:
    """Turn retrieved events into a text block to feed into the LLM."""
    if not events_with_scores:
        return "No matching events were found in the database."

    lines = []
    for i, (ev, score) in enumerate(events_with_scores, 1):
        lines.append(f"[{i}] Event: {ev.get('event', 'N/A')}")
        lines.append(f"    Date: {ev.get('date', 'N/A')}")
        lines.append(f"    Time: {ev.get('time', 'N/A')}")
        lines.append(f"    Location: {ev.get('location', 'N/A')}")
        lines.append(f"    URL: {ev.get('url', 'N/A')}")
        desc = (ev.get("description") or "").strip()
        if desc:
            lines.append(
                f"    Description: {desc[:400]}{'...' if len(desc) > 400 else ''}"
            )
        lines.append(f"    Semantic score: {score:.3f}")
        lines.append("")  # blank line

    return "\n".join(lines)


def call_groq_rag(
    user_question: str,
    events_with_scores: List[Tuple[dict, float]]
) -> str:
    """Call Groq LLM with RAG context."""
    client = get_groq_client()

    context_block = format_events_for_context(events_with_scores)

    system_prompt = (
        "You are an assistant that helps students explore events from the "
        "University of Maryland campus calendar.\n"
        "You are given retrieved event entries from a database.\n"
        "Use ONLY this information to answer the user's question.\n"
        "If the answer is not in the events, say you don't know.\n"
        "Always mention the event title and date when recommending something."
    )

    user_prompt = f"""
Here are the retrieved events:

{context_block}

User question: {user_question}

Answer concisely. Use bullet points when recommending multiple events.
If appropriate, reference specific events by their index [1], [2], etc.
"""

    chat_completion = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=512,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return chat_completion.choices[0].message.content.strip()


# ============================
#  STREAMLIT UI
# ============================

st.set_page_config(page_title="UMD Events Semantic RAG", page_icon="🎓")

st.title("🎓 UMD Events Assistant (Semantic RAG + Groq)")
st.write(
    "Ask questions about campus events. "
    "Search is **semantic** (embeddings), not just keyword matching."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of events to retrieve (top_k)", 1, 10, 5)
    st.markdown("---")
    st.markdown("**DB connection**")
    st.write(f"Host: `{DB_HOST}`")
    st.write(f"DB: `{DB_NAME}`")
    st.write(f"User: `{DB_USER}`")
    st.markdown("---")
    if st.button("🔁 Rebuild semantic index"):
        build_semantic_index.clear()
        st.success("Semantic index will be rebuilt on next query.")

question = st.text_input(
    "Ask about events (e.g., *What career fairs are happening in October?*)"
)

if st.button("Ask") and question.strip():
    with st.spinner("Running semantic search and calling Groq..."):
        retrieved = semantic_search(question, top_k=top_k)

        if not retrieved:
            st.warning("No events found in the database.")
        else:
            st.subheader("🔎 Retrieved Events (Semantic matches)")
            for i, (ev, score) in enumerate(retrieved, 1):
                with st.expander(f"[{i}] {ev.get('event', 'N/A')}  (score={score:.3f})"):
                    st.markdown(f"**Date:** {ev.get('date', 'N/A')}")
                    st.markdown(f"**Time:** {ev.get('time', 'N/A')}")
                    st.markdown(f"**Location:** {ev.get('location', 'N/A')}")
                    st.markdown(f"**URL:** {ev.get('url', 'N/A')}")
                    st.markdown("**Description:**")
                    st.write(ev.get("description", "N/A"))

            answer = call_groq_rag(question, retrieved)

            st.subheader("💬 Assistant Answer")
            st.write(answer)
