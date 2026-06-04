"""Run strategy evaluation and write per-example plus aggregate tables."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from src.data.load_examples import load_examples
from src.eval_framework.judge import build_judge
from src.eval_framework.metrics import compute_automatic_metrics
from src.eval_framework.model_runner import build_model_runner, count_tokens_rough
from src.eval_framework.strategies import build_strategy
from src.eval_framework.task_types import get_task_type


DEFAULT_TASKS = ["qasper", "hotpotqa", "gov_report", "multi_news", "passage_count"]
DEFAULT_STRATEGIES = ["full_context", "retrieval", "compression", "summarization"]


ROW_FIELDS = [
    "task",
    "example_id",
    "task_type",
    "strategy",
    "model",
    "query",
    "prediction",
    "reference_answer",
    "quality_source",
    "quality_score",
    "automatic_metric_name",
    "automatic_quality_score",
    "llm_judge_score",
    "llm_judge_reason",
    "llm_judge_model",
    "exact_match",
    "token_f1",
    "rouge_1",
    "rouge_l",
    "numeric_exact_match",
    "numeric_absolute_error",
    "original_context_tokens",
    "strategy_context_tokens",
    "input_tokens",
    "output_tokens",
    "compression_ratio",
    "strategy_latency_sec",
    "model_latency_sec",
    "total_latency_sec",
    "estimated_cost",
    "utility_score",
    "error",
]


AGG_FIELDS = [
    "task",
    "task_type",
    "strategy",
    "model",
    "num_examples",
    "error_rate",
    "avg_quality_score",
    "avg_automatic_quality_score",
    "avg_llm_judge_score",
    "avg_exact_match",
    "avg_token_f1",
    "avg_rouge_l",
    "avg_numeric_exact_match",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_compression_ratio",
    "avg_total_latency_sec",
    "avg_estimated_cost",
    "avg_utility_score",
]


def choose_quality_score(metrics: dict[str, Any], judge_score: float | None, quality_source: str) -> float:
    if quality_source == "llm_judge" and judge_score is not None:
        return judge_score
    return float(metrics["automatic_quality_score"])


def utility_score(quality: float, estimated_cost: float, latency_sec: float, lambda_cost: float, beta_latency: float) -> float:
    return quality - lambda_cost * estimated_cost - beta_latency * latency_sec


def run_one_row(example: dict[str, Any], strategy_name: str, args: argparse.Namespace, model_runner, judge) -> dict[str, Any]:
    task = example["task"]
    task_type = get_task_type(task)
    original_context_tokens = count_tokens_rough(example["context"])

    base_row = {
        "task": task,
        "example_id": example["example_id"],
        "task_type": task_type,
        "strategy": strategy_name,
        "model": args.model,
        "query": example["query"],
        "reference_answer": example["reference_answer"],
    }

    try:
        strategy = build_strategy(
            strategy_name,
            top_k=args.top_k,
            chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
            summarization_model=args.summarization_model,
        )

        # Step 1: strategy transforms full context into the prompt payload.
        strategy_start = time.perf_counter()
        strategy_result = strategy.prepare(example)
        strategy_latency = time.perf_counter() - strategy_start
        del strategy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 2: model answers from the strategy-produced prompt.
        model_result = model_runner.generate(strategy_result.prompt, example)

        # Step 3: compute task-type automatic metrics.
        metrics = compute_automatic_metrics(task_type, model_result.prediction, example["reference_answer"])

        # Step 4: optionally ask an LLM judge for a universal 0-1 quality score.
        judge_result = judge.score(
            task_type=task_type,
            query=example["query"],
            reference_answer=example["reference_answer"],
            prediction=model_result.prediction,
        )

        total_latency = strategy_latency + model_result.model_latency_sec
        quality = choose_quality_score(metrics, judge_result.score, args.quality_source)
        utility = utility_score(quality, model_result.estimated_cost, total_latency, args.lambda_cost, args.beta_latency)
        strategy_context_tokens = count_tokens_rough(strategy_result.context)

        return {
            **base_row,
            "prediction": model_result.prediction,
            "quality_source": args.quality_source,
            "quality_score": quality,
            "llm_judge_score": judge_result.score,
            "llm_judge_reason": judge_result.reason,
            "llm_judge_model": judge_result.judge_model,
            "original_context_tokens": original_context_tokens,
            "strategy_context_tokens": strategy_context_tokens,
            "input_tokens": model_result.input_tokens,
            "output_tokens": model_result.output_tokens,
            "compression_ratio": strategy_context_tokens / original_context_tokens if original_context_tokens else 0.0,
            "strategy_latency_sec": strategy_latency,
            "model_latency_sec": model_result.model_latency_sec,
            "total_latency_sec": total_latency,
            "estimated_cost": model_result.estimated_cost,
            "utility_score": utility,
            "error": "",
            **metrics,
        }
    except Exception as exc:
        return {
            **base_row,
            "prediction": "",
            "quality_source": args.quality_source,
            "quality_score": 0.0,
            "automatic_metric_name": "",
            "automatic_quality_score": 0.0,
            "llm_judge_score": None,
            "llm_judge_reason": "",
            "llm_judge_model": "",
            "exact_match": 0.0,
            "token_f1": 0.0,
            "rouge_1": 0.0,
            "rouge_l": 0.0,
            "numeric_exact_match": 0.0,
            "numeric_absolute_error": None,
            "original_context_tokens": original_context_tokens,
            "strategy_context_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "compression_ratio": 0.0,
            "strategy_latency_sec": 0.0,
            "model_latency_sec": 0.0,
            "total_latency_sec": 0.0,
            "estimated_cost": 0.0,
            "utility_score": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def average(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and value != ""]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["task"], row["task_type"], row["strategy"], row["model"])
        groups.setdefault(key, []).append(row)

    aggregate = []
    for (task, task_type, strategy, model), group in sorted(groups.items()):
        errors = [row for row in group if row["error"]]
        aggregate.append(
            {
                "task": task,
                "task_type": task_type,
                "strategy": strategy,
                "model": model,
                "num_examples": len(group),
                "error_rate": len(errors) / len(group) if group else 0.0,
                "avg_quality_score": average([row["quality_score"] for row in group]),
                "avg_automatic_quality_score": average([row["automatic_quality_score"] for row in group]),
                "avg_llm_judge_score": average([row["llm_judge_score"] for row in group]),
                "avg_exact_match": average([row["exact_match"] for row in group]),
                "avg_token_f1": average([row["token_f1"] for row in group]),
                "avg_rouge_l": average([row["rouge_l"] for row in group]),
                "avg_numeric_exact_match": average([row["numeric_exact_match"] for row in group]),
                "avg_input_tokens": average([row["input_tokens"] for row in group]),
                "avg_output_tokens": average([row["output_tokens"] for row in group]),
                "avg_compression_ratio": average([row["compression_ratio"] for row in group]),
                "avg_total_latency_sec": average([row["total_latency_sec"] for row in group]),
                "avg_estimated_cost": average([row["estimated_cost"] for row in group]),
                "avg_utility_score": average([row["utility_score"] for row in group]),
            }
        )
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model", default="mock_model")
    parser.add_argument("--judge", default="disabled")
    parser.add_argument("--quality-source", choices=["automatic", "llm_judge"], default="automatic")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-chars", type=int, default=2000)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--summarization-model", default=None, help="Override the summarization strategy model.")
    parser.add_argument("--lambda-cost", type=float, default=1.0)
    parser.add_argument("--beta-latency", type=float, default=0.0)
    parser.add_argument("--rows-output", type=Path, default=Path("outputs/processed/eval_rows.csv"))
    parser.add_argument("--aggregate-output", type=Path, default=Path("outputs/processed/eval_summary.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("outputs/processed/eval_rows.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(tasks=args.tasks, limit=args.limit)
    model_runner = build_model_runner(args.provider, args.model)
    judge = build_judge(args.judge)

    rows: list[dict[str, Any]] = []
    total = len(examples) * len(args.strategies)
    for example in tqdm(examples, desc="Examples"):
        for strategy_name in tqdm(args.strategies, desc="Strategies", leave=False):
            rows.append(run_one_row(example, strategy_name, args, model_runner, judge))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(args.rows_output, rows, ROW_FIELDS)
    aggregate = aggregate_rows(rows)
    write_csv(args.aggregate_output, aggregate, AGG_FIELDS)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)}/{total} rows to {args.rows_output}")
    print(f"Wrote {len(aggregate)} aggregate rows to {args.aggregate_output}")
    print(f"Wrote JSONL rows to {args.json_output}")


if __name__ == "__main__":
    main()
