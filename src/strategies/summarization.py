"""Summarization-based context strategy using Llama via HuggingFace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.load_examples import iter_examples
from src.strategies.base import StrategyResult

DEFAULT_OUTPUT_DIR = Path("data/longbench/processed")


class SummarizationStrategy:
    name = "summarization"

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs: dict = {"device_map": "auto"}
        try:
            import bitsandbytes  # noqa: F401
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            print(f"[SummarizationStrategy] Loaded {model_name} with 4-bit quantization")
        except Exception as e:
            print(f"[SummarizationStrategy] 4-bit failed ({e}), falling back to float16")
            kwargs["torch_dtype"] = torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            print(f"[SummarizationStrategy] Loaded {model_name} in float16")
        self.model.eval()

    def _summarize(self, context: str, query: str, max_summary_tokens: int = 500) -> str:
        prompt = (
            "Given the following question, summarize the text below keeping all information "
            "relevant to answering it:\n\n"
            f"Question: {query}\n\nText: {context}\n\nSummary:"
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(  # type: ignore[attr-defined]
                **inputs,
                max_new_tokens=max_summary_tokens,
                do_sample=False,
            )

        summary = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return summary.strip()

    def prepare(self, example: dict[str, Any]) -> StrategyResult:
        query = example["query"]
        context = example["context"]
        summary = self._summarize(context, query)

        prompt = f"Context: {summary}\n\nQuestion: {query}\n\nAnswer:"

        return StrategyResult(
            strategy=self.name,
            query=query,
            context=summary,
            prompt=prompt,
            metadata={
                "original_context_length": len(context.split()),
                "summary_length": len(summary.split()),
            },
        )


def _result_to_record(example: dict[str, Any], result: StrategyResult) -> dict[str, Any]:
    return {
        "example_id": example["example_id"],
        "task": example["task"],
        "task_type": example["task_type"],
        "strategy": result.strategy,
        "query": result.query,
        "context": result.context,
        "prompt": result.prompt,
        "reference_answer": example["reference_answer"],
        "answers": example["answers"],
        "metadata": {**example.get("metadata", {}), **result.metadata},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to process (default: all).")
    parser.add_argument("--limit", type=int, default=None, help="Max examples per task.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing normalized JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write summarization result JSONL files.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    strategy = SummarizationStrategy(model_name=args.model)

    # Group examples by task so each task gets its own output file.
    buckets: dict[str, list[dict[str, Any]]] = {}
    for example in iter_examples(tasks=args.tasks, data_dir=args.data_dir, limit=args.limit):
        buckets.setdefault(example["task"], []).append(example)

    for task, examples in buckets.items():
        out_path = args.output_dir / f"summarization_{task}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for example in examples:
                result = strategy.prepare(example)
                record = _result_to_record(example, result)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(examples)} records → {out_path}")


if __name__ == "__main__":
    main()
