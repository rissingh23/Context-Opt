"""Hybrid retrieval/compression/summary strategy placeholders."""

from __future__ import annotations

from src.strategies.base import StrategyResult


class RetrievalCompressionStrategy:
    name = "retrieval_compression"

    def prepare(self, example: dict) -> StrategyResult:
        raise NotImplementedError("Retrieve-then-compress will be implemented next.")


class RetrievalSummaryStrategy:
    name = "retrieval_summary"

    def prepare(self, example: dict) -> StrategyResult:
        raise NotImplementedError("Retrieve-then-summarize will be implemented next.")
