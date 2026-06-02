"""Retrieval-based context strategy placeholder."""

from __future__ import annotations

from src.strategies.base import StrategyResult


class RetrievalStrategy:
    name = "retrieval"

    def __init__(self, top_k: int = 5, chunk_chars: int = 2000) -> None:
        self.top_k = top_k
        self.chunk_chars = chunk_chars

    def prepare(self, example: dict) -> StrategyResult:
        raise NotImplementedError("Retrieval chunking/ranking will be implemented next.")
