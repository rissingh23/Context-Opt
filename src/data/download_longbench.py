"""Download and normalize starter LongBench tasks into JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from tqdm.auto import tqdm


DEFAULT_TASKS = ["qasper", "hotpotqa", "gov_report", "multi_news", "passage_count"]
SUMMARIZATION_TASKS = {"gov_report", "multi_news"}
DATASET_NAME = "THUDM/LongBench"


def _first_present(record: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _normalize_answer(value: Any) -> tuple[str, list[str]]:
    if value is None:
        return "", []
    if isinstance(value, list):
        answers = [str(item) for item in value]
        return (answers[0] if answers else ""), answers
    return str(value), [str(value)]


def normalize_record(task: str, record: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert a LongBench task record into the project-wide example schema."""

    query = str(_first_present(record, ["input", "query", "question", "instruction"]))
    context = str(_first_present(record, ["context", "document", "documents", "article"]))
    reference_answer, answers = _normalize_answer(_first_present(record, ["answers", "answer", "summary"]))
    example_id = str(_first_present(record, ["_id", "id"], f"{task}-{index}"))

    metadata = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "_id",
            "id",
            "input",
            "query",
            "question",
            "instruction",
            "context",
            "document",
            "documents",
            "article",
            "answers",
            "answer",
            "summary",
        }
    }

    return {
        "task": task,
        "example_id": example_id,
        "query": query,
        "context": context,
        "reference_answer": reference_answer,
        "answers": answers,
        "task_type": "summarization" if task in SUMMARIZATION_TASKS else "qa",
        "context_length_chars": len(context),
        "metadata": metadata,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def download_task(task: str, output_dir: Path, split: str, limit: int | None) -> int:
    dataset = load_dataset(DATASET_NAME, task, split=split, trust_remote_code=True)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    output_path = output_dir / f"{task}.jsonl"
    rows = (normalize_record(task, dict(record), index) for index, record in enumerate(dataset))
    return write_jsonl(output_path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="LongBench task names to download.")
    parser.add_argument("--split", default="test", help="Dataset split to load.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max examples per task.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/longbench/processed"),
        help="Directory for normalized JSONL files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "split": args.split,
        "limit": args.limit,
        "tasks": {},
    }

    for task in tqdm(args.tasks, desc="LongBench tasks"):
        count = download_task(task, args.output_dir, args.split, args.limit)
        manifest["tasks"][task] = {"path": str(args.output_dir / f"{task}.jsonl"), "count": count}

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
