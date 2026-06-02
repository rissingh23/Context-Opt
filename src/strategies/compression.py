"""Compression-based context strategy placeholder."""

from __future__ import annotations

from src.strategies.base import StrategyResult


class CompressionStrategy:
    name = "compression"

    def prepare(self, example: dict) -> StrategyResult:
        raise NotImplementedError("Approximate context compression will be implemented next.")
