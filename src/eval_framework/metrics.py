"""Task-aware automatic metrics.

These metrics are cheap and deterministic. They are not a replacement for an
LLM judge, but they give us stable sanity checks for every run.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from src.eval_framework.task_types import primary_automatic_metric


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    # SequenceMatcher gives us a lightweight ROUGE-L-style longest common
    # subsequence proxy without adding another dependency.
    matcher = SequenceMatcher(a=ref_tokens, b=pred_tokens, autojunk=False)
    lcs = sum(block.size for block in matcher.get_matching_blocks())
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_1(prediction: str, reference: str) -> float:
    return token_f1(prediction, reference)


def extract_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return float(match.group(0)) if match else None


def numeric_exact_match(prediction: str, reference: str) -> float:
    pred_number = extract_number(prediction)
    ref_number = extract_number(reference)
    if pred_number is None or ref_number is None:
        return 0.0
    return float(pred_number == ref_number)


def numeric_absolute_error(prediction: str, reference: str) -> float | None:
    pred_number = extract_number(prediction)
    ref_number = extract_number(reference)
    if pred_number is None or ref_number is None:
        return None
    return abs(pred_number - ref_number)


def compute_automatic_metrics(task_type: str, prediction: str, reference: str) -> dict[str, Any]:
    metrics = {
        "exact_match": exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "rouge_1": rouge_1(prediction, reference),
        "rouge_l": rouge_l(prediction, reference),
        "numeric_exact_match": numeric_exact_match(prediction, reference),
        "numeric_absolute_error": numeric_absolute_error(prediction, reference),
    }
    metric_name = primary_automatic_metric(task_type)
    metrics["automatic_metric_name"] = metric_name
    metrics["automatic_quality_score"] = metrics[metric_name]
    return metrics
