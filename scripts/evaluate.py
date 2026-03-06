import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from scripts.evaluation import (
    build_run_filename,
    load_eval_samples,
    persist_run,
    run_ragas_evaluation,
    summarize_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation against the current RAG stack")
    parser.add_argument("--dataset", default="eval/dataset.json", help="Path to eval dataset JSON file")
    parser.add_argument("--tag", default="manual", help="Run tag used for output file naming")
    parser.add_argument("--output-dir", default="eval/results", help="Directory to save evaluation output")
    parser.add_argument("--answer-model", default="llama-3.3-70b-versatile", help="Model used to generate answers")
    parser.add_argument("--judge-model", default="llama-3.1-8b-instant", help="Model used by RAGAS metrics")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K retrieval size")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples")
    parser.add_argument("--delay", type=float, default=2.5, help="Per-sample delay in seconds")
    return parser.parse_args()


async def run():
    args = parse_args()
    load_dotenv()

    if "ELASTIC_HOST" not in os.environ:
        os.environ["ELASTIC_HOST"] = "http://localhost:9200"
    if "DB_HOST" not in os.environ and "POSTGRES_HOST" not in os.environ:
        os.environ["DB_HOST"] = "localhost"

    print("[eval] Loading application modules...", flush=True)

    import app

    if not app.es_client.ping():
        raise RuntimeError(
            "Elasticsearch is not reachable. Start services first (e.g., `docker compose up -d`) "
            "or set ELASTIC_HOST to a reachable endpoint."
        )

    with app.get_db_cursor() as cur:
        cur.execute("SELECT 1;")

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_file = output_dir / build_run_filename(args.tag)

    eval_samples = load_eval_samples(dataset_path, sample_limit=args.limit)
    if not eval_samples:
        raise ValueError(f"No valid eval samples found in {dataset_path}")

    print(
        f"[eval] Starting run tag={args.tag} samples={len(eval_samples)} "
        f"answer_model={args.answer_model} judge_model={args.judge_model}",
        flush=True,
    )

    def progress(message: str):
        print(f"[eval] {message}", flush=True)

    payload = await run_ragas_evaluation(
        eval_samples=eval_samples,
        search_events_fn=app.search_events,
        llm_client=app.llm_client,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        groq_api_key=app.GROQ_API_KEY,
        embedding_model_name=app.EMBEDDING_MODEL_NAME,
        top_k=args.top_k,
        per_sample_delay_seconds=args.delay,
        progress_callback=progress,
    )

    persist_run(
        output_file,
        tag=args.tag,
        dataset_path=dataset_path,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        top_k=args.top_k,
        embedding_model_name=app.EMBEDDING_MODEL_NAME,
        eval_payload=payload,
    )

    summary = summarize_results(payload["results"])
    timing = payload.get("timing", {})
    print("\nEvaluation complete")
    print(f"Samples:    {len(eval_samples)}")
    print(f"Total time: {timing.get('total_formatted', 'N/A')}")
    print(f"  Generation:    {timing.get('generation_formatted', 'N/A')}")
    print(f"  RAGAS judging: {timing.get('ragas_judging_formatted', 'N/A')}")
    print(f"  Avg/sample:    {timing.get('avg_sample_seconds', 'N/A')}s")
    print(f"Output:     {output_file}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
