"""Load normalized LongBench examples for strategy and eval code."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path("data/longbench/processed")


def iter_examples(
    tasks: list[str] | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    limit: int | None = None,
    offset: int = 0,
) -> Iterator[dict[str, Any]]:
    paths = [data_dir / f"{task}.jsonl" for task in tasks] if tasks else sorted(data_dir.glob("*.jsonl"))
    emitted = 0

    for path in paths:
        if path.name == "manifest.json":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Missing normalized dataset file: {path}")
        skipped = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if skipped < offset:
                    skipped += 1
                    continue
                yield json.loads(line)
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def load_examples(
    tasks: list[str] | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    return list(iter_examples(tasks=tasks, data_dir=data_dir, limit=limit, offset=offset))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect normalized LongBench examples.")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for example in load_examples(tasks=args.tasks, data_dir=args.data_dir, limit=args.limit):
        preview = {
            "task": example["task"],
            "example_id": example["example_id"],
            "query": example["query"][:120],
            "context_length_chars": example["context_length_chars"],
            "reference_answer": example["reference_answer"][:120],
        }
        print(json.dumps(preview, ensure_ascii=False))


if __name__ == "__main__":
    main()
