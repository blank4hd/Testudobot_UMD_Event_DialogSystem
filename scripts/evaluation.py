import asyncio
import copy
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig


@dataclass
class EvalSample:
    id: str
    question: str
    ground_truth: str
    category: str = "general"
    difficulty: str = "medium"
    tags: Optional[List[str]] = None


def build_default_eval_samples() -> List[Dict[str, str]]:
    return [
        {
            "id": "default_001",
            "question": "Are there any career fairs happening this month?",
            "ground_truth": "Lists upcoming career fairs at UMD.",
            "category": "career",
            "difficulty": "easy",
            "tags": ["temporal", "category-filter"],
        },
        {
            "id": "default_002",
            "question": "What music performances are scheduled?",
            "ground_truth": "Summarizes music or concert events.",
            "category": "music",
            "difficulty": "easy",
            "tags": ["category-filter"],
        },
        {
            "id": "default_003",
            "question": "Is there free food at any event?",
            "ground_truth": "Identifies events that explicitly mention free food.",
            "category": "food",
            "difficulty": "medium",
            "tags": ["constraint"],
        },
    ]


def load_eval_samples(dataset_path: Path, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not dataset_path.exists():
        return build_default_eval_samples()

    with dataset_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    raw_samples = payload.get("samples", [])
    samples = []
    for index, sample in enumerate(raw_samples, start=1):
        item = {
            "id": sample.get("id", f"q{index:03d}"),
            "question": sample.get("question", "").strip(),
            "ground_truth": sample.get("ground_truth", "").strip(),
            "category": sample.get("category", "general"),
            "difficulty": sample.get("difficulty", "medium"),
            "tags": sample.get("tags", []),
        }
        if item["question"] and item["ground_truth"]:
            samples.append(item)

    if sample_limit is not None:
        return samples[:sample_limit]

    return samples


def build_eval_run_config() -> RunConfig:
    return RunConfig(
        max_workers=1,
        max_retries=12,
        max_wait=90,
        timeout=300,
    )


def _build_eval_metrics():
    eval_metrics = [
        copy.deepcopy(context_precision),
        copy.deepcopy(context_recall),
        copy.deepcopy(faithfulness),
        copy.deepcopy(answer_relevancy),
    ]
    for metric in eval_metrics:
        if hasattr(metric, "strictness"):
            metric.strictness = 1
    return eval_metrics


def _condense_context(results: Sequence[Any], max_context_items: int = 5) -> List[str]:
    """Return each retrieved event as a *separate* context string.

    RAGAS expects ``contexts`` to be a ``List[str]`` where each element is an
    independent chunk.  Joining everything into a single string collapsed the
    list to length-1, which made context_recall always 0.
    """
    condensed_contexts: List[str] = []
    for event_doc, _score in results[:max_context_items]:
        event_name = (event_doc.get("event") or "")[:120]
        date = (event_doc.get("date") or "")[:30]
        time = (event_doc.get("time") or "")[:30]
        location = (event_doc.get("location") or "")[:90]
        description = (event_doc.get("description") or "")[:350]
        condensed_contexts.append(
            f"Event: {event_name}\nDate: {date}\nTime: {time}\n"
            f"Location: {location}\nDescription: {description}"
        )
    return condensed_contexts


async def run_ragas_evaluation(
    eval_samples: List[Dict[str, Any]],
    search_events_fn: Callable[..., Sequence[Any]],
    llm_client: Any,
    answer_model: str,
    judge_model: str,
    groq_api_key: str,
    embedding_model_name: str,
    *,
    top_k: int = 5,
    per_sample_delay_seconds: float = 2.5,
    max_answer_tokens: int = 180,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    questions: List[str] = []
    answers: List[str] = []
    ground_truths: List[str] = []
    contexts_list: List[List[str]] = []
    per_sample_times: List[float] = []

    overall_start = time.monotonic()
    generation_start = time.monotonic()

    ragas_llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
        model=judge_model,
        temperature=0.0,
        n=1,
    )
    ragas_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    total_samples = len(eval_samples)
    est_per_sample = per_sample_delay_seconds + 6.0  # retrieval + LLM + delay
    est_generation = est_per_sample * total_samples
    est_ragas = total_samples * 12.0  # ~12s per sample for RAGAS judging
    est_total = est_generation + est_ragas

    if progress_callback:
        progress_callback(
            f"Prepared RAGAS judge model and embeddings for {total_samples} samples  "
            f"| Estimated time: {_format_duration(est_total)} "
            f"(generation ~{_format_duration(est_generation)}, "
            f"RAGAS judging ~{_format_duration(est_ragas)})"
        )

    for index, sample in enumerate(eval_samples, start=1):
        sample_start = time.monotonic()
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        if progress_callback:
            if index > 1:
                avg_sample = sum(per_sample_times) / len(per_sample_times)
                remaining_gen = avg_sample * (total_samples - index + 1)
                eta = remaining_gen + est_ragas
                progress_callback(
                    f"Sample {index}/{total_samples}: retrieving context  "
                    f"| avg {avg_sample:.1f}s/sample, ETA ~{_format_duration(eta)}"
                )
            else:
                progress_callback(f"Sample {index}/{total_samples}: retrieving context")

        results = search_events_fn(question, top_k=top_k)
        contexts = _condense_context(results)
        if not contexts:
            contexts = ["No relevant events found."]
        context_text = "\n\n".join(contexts)

        if progress_callback:
            progress_callback(f"Sample {index}/{total_samples}: generating answer ({len(contexts)} chunks)")

        response = llm_client.chat.completions.create(
            model=answer_model,
            messages=[
                {
                    "role": "user",
                    "content": f"Context: {context_text}\nQuestion: {question}",
                }
            ],
            temperature=0.1,
            max_tokens=max_answer_tokens,
        )
        answer_text = (response.choices[0].message.content or "").strip()

        questions.append(question)
        answers.append(answer_text)
        ground_truths.append(ground_truth)
        contexts_list.append(contexts)

        sample_elapsed = time.monotonic() - sample_start
        per_sample_times.append(sample_elapsed)

        if per_sample_delay_seconds > 0:
            await asyncio.sleep(per_sample_delay_seconds)

    generation_elapsed = time.monotonic() - generation_start
    ragas_start = time.monotonic()

    if progress_callback:
        progress_callback(
            f"Starting RAGAS metric evaluation  "
            f"| Generation phase took {_format_duration(generation_elapsed)}"
        )

    eval_dataset = Dataset.from_dict(
        {
            "question": questions,
            "contexts": contexts_list,
            "answer": answers,
            "ground_truth": ground_truths,
        }
    )

    results = evaluate(
        dataset=eval_dataset,
        metrics=_build_eval_metrics(),
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=True,
        run_config=build_eval_run_config(),
        batch_size=1,
    )

    ragas_elapsed = time.monotonic() - ragas_start
    total_elapsed = time.monotonic() - overall_start

    timing = {
        "total_seconds": round(total_elapsed, 2),
        "generation_seconds": round(generation_elapsed, 2),
        "ragas_judging_seconds": round(ragas_elapsed, 2),
        "avg_sample_seconds": round(sum(per_sample_times) / len(per_sample_times), 2) if per_sample_times else 0,
        "per_sample_seconds": [round(t, 2) for t in per_sample_times],
        "total_formatted": _format_duration(total_elapsed),
        "generation_formatted": _format_duration(generation_elapsed),
        "ragas_judging_formatted": _format_duration(ragas_elapsed),
    }

    if progress_callback:
        progress_callback(
            f"RAGAS evaluation complete  "
            f"| Total: {timing['total_formatted']} "
            f"(generation {timing['generation_formatted']}, "
            f"RAGAS judging {timing['ragas_judging_formatted']})"
        )

    return {
        "results": results,
        "eval_samples": eval_samples,
        "questions": questions,
        "answers": answers,
        "ground_truths": ground_truths,
        "contexts": contexts_list,
        "timing": timing,
    }


def _format_duration(seconds: float) -> str:
    """Return a human-readable duration string like '2m 35s' or '48s'."""
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"


def summarize_results(result_obj: Any) -> Dict[str, float]:
    frame = result_obj.to_pandas()
    summary = frame.select_dtypes(include=["number"]).mean().dropna().to_dict()
    return {k: float(v) for k, v in summary.items()}


def build_run_filename(tag: str, suffix: str = "json") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    safe_tag = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in tag)
    return f"{timestamp}_{safe_tag}.{suffix}"


def persist_run(
    output_path: Path,
    *,
    tag: str,
    dataset_path: Path,
    answer_model: str,
    judge_model: str,
    top_k: int,
    embedding_model_name: str,
    eval_payload: Dict[str, Any],
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_obj = eval_payload["results"]
    summary = summarize_results(result_obj)
    frame = result_obj.to_pandas()

    serialized = {
        "run_id": output_path.stem,
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset_path": str(dataset_path),
            "num_samples": len(eval_payload["eval_samples"]),
            "answer_model": answer_model,
            "judge_model": judge_model,
            "top_k": top_k,
            "embedding_model": embedding_model_name,
        },
        "timing": eval_payload.get("timing", {}),
        "aggregate_scores": summary,
        "per_sample": frame.to_dict(orient="records"),
        "samples": eval_payload["eval_samples"],
        "answers": eval_payload["answers"],
        "contexts": eval_payload["contexts"],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serialized, file, indent=2, ensure_ascii=False)

    return serialized
