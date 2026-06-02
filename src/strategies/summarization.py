"""Summarization-based context strategy placeholder."""

from __future__ import annotations

from src.strategies.base import StrategyResult


class SummarizationStrategy:
    name = "summarization"

    def prepare(self, example: dict) -> StrategyResult:
        raise NotImplementedError("Model-backed context summarization will be implemented next.")
