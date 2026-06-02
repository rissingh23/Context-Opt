"""Task type mapping used by the eval framework."""

from __future__ import annotations


QA_TASKS = {"qasper", "hotpotqa"}
SUMMARIZATION_TASKS = {"gov_report", "multi_news"}
COUNTING_TASKS = {"passage_count"}


def get_task_type(task: str) -> str:
    if task in QA_TASKS:
        return "qa"
    if task in SUMMARIZATION_TASKS:
        return "summarization"
    if task in COUNTING_TASKS:
        return "counting"
    return "unknown"


def primary_automatic_metric(task_type: str) -> str:
    if task_type == "qa":
        return "token_f1"
    if task_type == "summarization":
        return "rouge_l"
    if task_type == "counting":
        return "numeric_exact_match"
    return "token_f1"
