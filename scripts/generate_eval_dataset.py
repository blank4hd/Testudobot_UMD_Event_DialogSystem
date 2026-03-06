import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Generate candidate RAG evaluation questions from event data")
    parser.add_argument("--input", default="data/umd_events_2026-03-06.json", help="Input events JSON file")
    parser.add_argument("--output", default="eval/dataset_candidates.json", help="Output candidate dataset JSON")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Groq model used to generate candidates")
    parser.add_argument("--per-category", type=int, default=4, help="Candidate questions per category")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between category calls to respect rate limits")
    return parser.parse_args()


def load_events(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON array of event objects")
    return payload


def sample_context(events: List[Dict], max_items: int = 25) -> str:
    rows = []
    for event in events[:max_items]:
        title = event.get("event", "")
        date = event.get("date", "")
        location = event.get("location", "")
        description = (event.get("description", "") or "")[:180]
        rows.append(f"- Event: {title} | Date: {date} | Location: {location} | Description: {description}")
    return "\n".join(rows)


def parse_generated_json(text: str) -> List[Dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    parsed = json.loads(text)
    if isinstance(parsed, dict) and "samples" in parsed:
        parsed = parsed["samples"]
    if not isinstance(parsed, list):
        raise ValueError("Model output must be a JSON list")
    return parsed


def main():
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is required")

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    events = load_events(Path(args.input))
    context = sample_context(events)

    categories = [
        "career",
        "music",
        "sports",
        "food",
        "academic",
        "social",
        "location",
        "temporal",
        "negation",
        "multi-constraint",
    ]

    all_samples: List[Dict] = []
    index = 1

    for category in categories:
        prompt = f"""You are generating evaluation questions for a UMD events RAG system.
Given this event snapshot, create exactly {args.per_category} diverse QA evaluation samples for category '{category}'.

Event snapshot:
{context}

Output requirements:
- Return ONLY valid JSON list
- Each item keys: question, ground_truth, category, difficulty, tags
- ground_truth should be a concise expected-answer summary (not exact final phrasing)
- tags must be a JSON array of short strings
"""

        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=900,
        )
        content = (response.choices[0].message.content or "[]").strip()

        try:
            generated = parse_generated_json(content)
        except Exception:
            generated = []

        for item in generated:
            question = (item.get("question") or "").strip()
            ground_truth = (item.get("ground_truth") or "").strip()
            if not question or not ground_truth:
                continue
            all_samples.append(
                {
                    "id": f"cand_{index:03d}",
                    "question": question,
                    "ground_truth": ground_truth,
                    "category": item.get("category", category),
                    "difficulty": item.get("difficulty", "medium"),
                    "tags": item.get("tags", [category]),
                }
            )
            index += 1

        if args.delay > 0:
            time.sleep(args.delay)

    payload = {
        "version": "candidate-1.0",
        "created": "2026-03-05",
        "source_file": args.input,
        "model": args.model,
        "samples": all_samples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(f"Generated {len(all_samples)} candidate samples at {output_path}")


if __name__ == "__main__":
    main()
