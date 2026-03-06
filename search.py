import logging
import re
from datetime import datetime, timedelta

import numpy as np
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

RRF_K = 60
VECTOR_SEARCH_CANDIDATES = 200

es_client = None
embedding_model = None
cross_encoder = None
llm_client = None
LABEL_MODEL = None


def init_search(es, embed_model, ce_model, llm, label_model):
    global es_client, embedding_model, cross_encoder, llm_client, LABEL_MODEL
    es_client = es
    embedding_model = embed_model
    cross_encoder = ce_model
    llm_client = llm
    LABEL_MODEL = label_model


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
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
            max_tokens=30,
        )
        expanded = (resp.choices[0].message.content or "").strip()
        return expanded if expanded else user_query
    except Exception as e:
        logger.warning("Query expansion failed, using original query: %s", e)
        return user_query


def extract_date_range(query_text: str) -> tuple[str | None, str | None]:
    """Extract date range from a query without performing search."""
    current = datetime.now()
    q_lower = query_text.lower()

    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    if "today" in q_lower:
        d = current.strftime("%Y-%m-%d")
        return d, d
    elif "tomorrow" in q_lower:
        d = (current + timedelta(days=1)).strftime("%Y-%m-%d")
        return d, d
    elif "this weekend" in q_lower:
        days_until_sat = (5 - current.weekday()) % 7
        if days_until_sat == 0 and current.weekday() != 5:
            days_until_sat = 7
        sat = current + timedelta(days=days_until_sat)
        if current.weekday() >= 5:
            sat = current if current.weekday() == 5 else current - timedelta(days=1)
        return sat.strftime("%Y-%m-%d"), (sat + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "upcoming" in q_lower or "future" in q_lower or "soon" in q_lower:
        return current.strftime("%Y-%m-%d"), (current + relativedelta(years=1)).strftime("%Y-%m-%d")
    elif "this month" in q_lower:
        return current.replace(day=1).strftime("%Y-%m-%d"), (
            current + relativedelta(months=1, days=-1)
        ).strftime("%Y-%m-%d")
    elif "next month" in q_lower:
        s = (current + relativedelta(months=1)).replace(day=1)
        return s.strftime("%Y-%m-%d"), (s + relativedelta(months=1, days=-1)).strftime("%Y-%m-%d")
    elif "next week" in q_lower:
        return current.strftime("%Y-%m-%d"), (current + timedelta(days=7)).strftime("%Y-%m-%d")
    else:
        for m_name, m_num in months.items():
            if m_name in q_lower:
                year = current.year
                if m_num < current.month:
                    year += 1
                dt = datetime(year, m_num, 1)
                return dt.strftime("%Y-%m-%d"), (dt + relativedelta(months=1, days=-1)).strftime("%Y-%m-%d")
    return None, None


def search_events(
    query_text: str,
    top_k: int = 15,
    filter_topic_id: int | None = None,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
    override_date_range: tuple[str | None, str | None] = (None, None),
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
    override_start, override_end = override_date_range

    if override_start and override_end:
        start_date = override_start
        end_date = override_end
    else:
        if "today" in q_lower:
            start_date = current.strftime("%Y-%m-%d")
            end_date = current.strftime("%Y-%m-%d")
            cleaned_query = re.sub(r"\btoday\b", "", cleaned_query, flags=re.I)

        elif "tomorrow" in q_lower:
            tmrw = current + timedelta(days=1)
            start_date = tmrw.strftime("%Y-%m-%d")
            end_date = tmrw.strftime("%Y-%m-%d")
            cleaned_query = re.sub(r"\btomorrow\b", "", cleaned_query, flags=re.I)

        elif "this weekend" in q_lower:
            days_until_saturday = (5 - current.weekday()) % 7
            if days_until_saturday == 0 and current.weekday() != 5:
                days_until_saturday = 7
            saturday = current + timedelta(days=days_until_saturday)
            sunday = saturday + timedelta(days=1)
            if current.weekday() >= 5:
                saturday = current if current.weekday() == 5 else current - timedelta(days=1)
                sunday = saturday + timedelta(days=1)
            start_date = saturday.strftime("%Y-%m-%d")
            end_date = sunday.strftime("%Y-%m-%d")
            cleaned_query = re.sub(r"\bthis weekend\b", "", cleaned_query, flags=re.I)

        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        specific_date_match = re.search(r"\b(" + "|".join(months.keys()) + r")\s+(\d{1,2})\b", q_lower)

        if start_date is None and end_date is None:
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
                    logger.warning("Invalid date parsed: %s %s", m_name, day_num)

            elif "upcoming" in q_lower or "future" in q_lower or "soon" in q_lower:
                start_date = current.strftime("%Y-%m-%d")
                end_date = (current + relativedelta(years=1)).strftime("%Y-%m-%d")
                cleaned_query = re.sub(r"\b(upcoming|future|soon)\b", "", cleaned_query, flags=re.I)

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

            else:
                for m_name, m_num in months.items():
                    if m_name in q_lower:
                        year = current.year
                        if m_num < current.month:
                            year += 1
                        dt = datetime(year, m_num, 1)
                        start_date = dt.strftime("%Y-%m-%d")
                        end_date = (dt + relativedelta(months=1, days=-1)).strftime("%Y-%m-%d")
                        cleaned_query = re.sub(rf"\b{m_name}\b", "", cleaned_query, flags=re.I)
                        break

    cleaned_query = re.sub(
        r"(?<!\w)(events?|happening|show me|find me|find|list|from|on|in)(?!\w)",
        "",
        cleaned_query,
        flags=re.I,
    ).strip()
    cleaned_query = re.sub(r"\s{2,}", " ", cleaned_query).strip()

    if not cleaned_query:
        cleaned_query = query_text

    es_filters = []
    if filter_topic_id is not None:
        es_filters.append({"term": {"topic_id": filter_topic_id}})
    if start_date and end_date:
        es_filters.append({"range": {"date": {"gte": start_date, "lte": end_date, "format": "yyyy-MM-dd"}}})

    filter_query = {"bool": {"filter": es_filters}} if es_filters else None

    try:
        vector_body = {
            "size": top_k,
            "knn": {
                "field": "embedding",
                "query_vector": embedding_model.encode(cleaned_query, normalize_embeddings=True).tolist(),
                "k": top_k,
                "num_candidates": VECTOR_SEARCH_CANDIDATES,
            },
            "_source": ["event", "description", "date", "time", "location", "url"],
        }
        if filter_query:
            vector_body["knn"]["filter"] = filter_query
        v_res = es_client.search(index="umd_events", body=vector_body)["hits"]["hits"]
    except Exception:
        logger.exception("Vector search failed")
        v_res = []

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
                            "operator": "or",
                        }
                    },
                    "filter": es_filters,
                }
            },
            "_source": ["event", "description", "date", "time", "location", "url"],
        }
        k_res = es_client.search(index="umd_events", body=keyword_body)["hits"]["hits"]
    except Exception:
        logger.exception("Keyword search failed")
        k_res = []

    def rrf_score(rank: int, k: int = RRF_K) -> float:
        return 1.0 / (k + rank)

    fused = {}
    for r, hit in enumerate(v_res):
        _id = hit["_id"]
        if _id not in fused:
            fused[_id] = {"doc": hit["_source"], "score": 0.0}
        fused[_id]["score"] += vector_weight * rrf_score(r)

    for r, hit in enumerate(k_res):
        _id = hit["_id"]
        if _id not in fused:
            fused[_id] = {"doc": hit["_source"], "score": 0.0}
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
            ce_scores = np.asarray(ce_scores, dtype=float)
            for item, ce_score in zip(ranked, ce_scores):
                item["score"] = float(ce_score)

            ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)[:top_k]
    except Exception as e:
        logger.warning("Cross-encoder reranking failed; falling back to RRF ranking: %s", e)

    return [(item["doc"], item["score"]) for item in ranked]
