"""Strategy registry for the eval framework."""

from __future__ import annotations

from typing import Any

from src.strategies.compression import CompressionStrategy
from src.strategies.full_context import FullContextStrategy
from src.strategies.retrieval import RetrievalStrategy
from src.strategies.summarization import SummarizationStrategy


def build_strategy(name: str, **kwargs: Any):
    if name == "full_context":
        return FullContextStrategy()
    if name == "retrieval":
        return RetrievalStrategy(
            top_k=kwargs.get("top_k", 3),
            chunk_chars=kwargs.get("chunk_chars", 2000),
            overlap_chars=kwargs.get("overlap_chars", 200),
        )
    if name == "compression":
        return CompressionStrategy()
    if name == "summarization":
        return SummarizationStrategy()
    raise ValueError(f"Unknown strategy '{name}'")
