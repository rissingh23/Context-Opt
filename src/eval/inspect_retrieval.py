"""Create a small JSON report showing what retrieval does on LongBench examples."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.load_examples import load_examples
from src.strategies.retrieval import RetrievalStrategy


DEFAULT_TASKS = ["qasper", "hotpotqa", "gov_report", "multi_news", "passage_count"]


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def token_f1(reference: str, retrieved_context: str) -> dict[str, float | bool]:
    """Score whether retrieval preserved words from the reference answer."""

    normalized_reference = normalize_text(reference)
    normalized_context = normalize_text(retrieved_context)
    if not normalized_reference:
        return {
            "reference_exactly_in_retrieved_context": False,
            "reference_token_precision": 0.0,
            "reference_token_recall": 0.0,
            "reference_token_f1": 0.0,
        }

    # Step 1: exact containment catches the cleanest case: the gold answer text
    # appears verbatim somewhere in the retrieved chunks.
    exact_match = normalized_reference in normalized_context

    reference_tokens = normalized_reference.split()
    context_tokens = normalized_context.split()
    reference_counts = Counter(reference_tokens)
    context_counts = Counter(context_tokens)
    overlap = sum((reference_counts & context_counts).values())

    if overlap == 0:
        precision = recall = f1 = 0.0
    else:
        # Step 2: token overlap gives a softer signal when the retrieved context
        # contains answer evidence but not the full reference string verbatim.
        precision = overlap / len(context_tokens) if context_tokens else 0.0
        recall = overlap / len(reference_tokens)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "reference_exactly_in_retrieved_context": exact_match,
        "reference_token_precision": precision,
        "reference_token_recall": recall,
        "reference_token_f1": f1,
    }


def build_report_row(example: dict[str, Any], strategy: RetrievalStrategy) -> dict[str, Any]:
    result = strategy.prepare(example)
    coverage = token_f1(example["reference_answer"], result.context)

    # Step 3: keep enough original/example fields to understand the task without
    # needing to reopen the dataset file.
    return {
        "task": example["task"],
        "example_id": example["example_id"],
        "task_type": example["task_type"],
        "query": example["query"],
        "reference_answer": example["reference_answer"],
        "full_context_length_chars": len(example["context"]),
        "retrieved_context_length_chars": len(result.context),
        "compression_ratio_chars": len(result.context) / len(example["context"]) if example["context"] else 0.0,
        "strategy": result.strategy,
        "retrieval_metadata": result.metadata,
        "retrieval_coverage_score": coverage,
        "prompt": result.prompt,
        "retrieved_context": result.context,
    }


def load_balanced_examples(tasks: list[str], limit: int) -> list[dict[str, Any]]:
    """Load examples round-robin so small reports cover multiple tasks."""

    examples_by_task = {task: load_examples(tasks=[task], limit=limit) for task in tasks}
    selected: list[dict[str, Any]] = []
    offset = 0

    # Step 0: choose examples in task order, one at a time, so --limit 5 with
    # the default five tasks gives a broad first look instead of only qasper.
    while len(selected) < limit:
        added = False
        for task in tasks:
            task_examples = examples_by_task[task]
            if offset < len(task_examples):
                selected.append(task_examples[offset])
                added = True
                if len(selected) == limit:
                    return selected
        if not added:
            break
        offset += 1

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--limit", type=int, default=5, help="Total number of examples to inspect.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-chars", type=int, default=2000)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("outputs/processed/retrieval_examples.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy = RetrievalStrategy(top_k=args.top_k, chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars)
    examples = load_balanced_examples(tasks=args.tasks, limit=args.limit)

    # Step 4: write a JSON report instead of only printing, so collaborators can
    # inspect the same retrieval outputs later.
    report = {
        "strategy": "retrieval",
        "strategy_params": {
            "top_k": args.top_k,
            "chunk_chars": args.chunk_chars,
            "overlap_chars": args.overlap_chars,
        },
        "scoring_note": "Scores are retrieval coverage only, not final model-answer quality.",
        "examples": [build_report_row(example, strategy) for example in examples],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['examples'])} retrieval examples to {args.output}")


if __name__ == "__main__":
    main()
