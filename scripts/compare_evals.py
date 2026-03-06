import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two RAGAS evaluation run files")
    parser.add_argument("baseline", help="Path to baseline run JSON")
    parser.add_argument("candidate", help="Path to candidate run JSON")
    parser.add_argument("--threshold", type=float, default=0.10, help="Regression threshold for per-sample metrics")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metric_deltas(base_scores: Dict[str, float], cand_scores: Dict[str, float]) -> Dict[str, Tuple[float, float, float]]:
    all_keys = sorted(set(base_scores.keys()) | set(cand_scores.keys()))
    deltas = {}
    for key in all_keys:
        base = float(base_scores.get(key, 0.0))
        cand = float(cand_scores.get(key, 0.0))
        deltas[key] = (base, cand, cand - base)
    return deltas


def build_row_map(per_sample):
    row_map = {}
    for index, row in enumerate(per_sample):
        question = row.get("question") or f"row_{index}"
        row_map[question] = row
    return row_map


def find_regressions(base_rows, cand_rows, threshold: float):
    numeric_metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    regressions = []

    shared_questions = set(base_rows.keys()) & set(cand_rows.keys())
    for question in sorted(shared_questions):
        base_row = base_rows[question]
        cand_row = cand_rows[question]

        for metric in numeric_metrics:
            base_val = base_row.get(metric)
            cand_val = cand_row.get(metric)
            if isinstance(base_val, (int, float)) and isinstance(cand_val, (int, float)):
                diff = cand_val - base_val
                if diff <= -abs(threshold):
                    regressions.append(
                        {
                            "question": question,
                            "metric": metric,
                            "baseline": round(float(base_val), 4),
                            "candidate": round(float(cand_val), 4),
                            "delta": round(float(diff), 4),
                        }
                    )
    return regressions


def main():
    args = parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)

    baseline = load_json(baseline_path)
    candidate = load_json(candidate_path)

    base_scores = baseline.get("aggregate_scores", {})
    cand_scores = candidate.get("aggregate_scores", {})
    deltas = metric_deltas(base_scores, cand_scores)

    print("\n=== Aggregate Metric Comparison ===")
    for metric, (base, cand, delta) in deltas.items():
        sign = "+" if delta >= 0 else ""
        print(f"- {metric}: {base:.4f} -> {cand:.4f} ({sign}{delta:.4f})")

    print("\n=== Config Diff ===")
    base_cfg = baseline.get("config", {})
    cand_cfg = candidate.get("config", {})
    cfg_keys = sorted(set(base_cfg.keys()) | set(cand_cfg.keys()))
    for key in cfg_keys:
        b_val = base_cfg.get(key)
        c_val = cand_cfg.get(key)
        if b_val != c_val:
            print(f"- {key}: {b_val} -> {c_val}")

    base_rows = build_row_map(baseline.get("per_sample", []))
    cand_rows = build_row_map(candidate.get("per_sample", []))
    regressions = find_regressions(base_rows, cand_rows, threshold=args.threshold)

    print("\n=== Regressions ===")
    if not regressions:
        print("- None above threshold")
    else:
        for item in regressions:
            print(
                f"- {item['metric']}: {item['baseline']:.4f} -> {item['candidate']:.4f} "
                f"({item['delta']:+.4f}) | {item['question']}"
            )


if __name__ == "__main__":
    main()
