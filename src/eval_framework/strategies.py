"""Strategy registry for the eval framework."""

from __future__ import annotations

from typing import Any

from src.strategies.compression import CompressionStrategy
from src.strategies.full_context import FullContextStrategy
from src.strategies.gemini_summarization import GeminiSummarizationStrategy
from src.strategies.hybrid import RetrievalCompressionStrategy, RetrievalSummaryStrategy
from src.strategies.retrieval import RetrievalStrategy
from src.strategies.summarization import SummarizationStrategy


def build_strategy(name: str, **kwargs: Any):
    top_k = kwargs.get("top_k", 3)
    chunk_chars = kwargs.get("chunk_chars", 2000)
    overlap_chars = kwargs.get("overlap_chars", 200)
    project = kwargs.get("vertexai_project", "")
    location = kwargs.get("vertexai_location", "us-central1")

    if name == "full_context":
        return FullContextStrategy()
    if name == "retrieval":
        return RetrievalStrategy(top_k=top_k, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    if name == "compression":
        return CompressionStrategy()
    if name == "summarization":
        model_name = kwargs.get("summarization_model")
        return SummarizationStrategy(**({} if model_name is None else {"model_name": model_name}))
    if name == "gemini_summarization":
        return GeminiSummarizationStrategy(project=project, location=location)
    if name == "retrieval_compression":
        return RetrievalCompressionStrategy(top_k=top_k, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    if name == "retrieval_summary":
        model_name = kwargs.get("summarization_model")
        return RetrievalSummaryStrategy(
            top_k=top_k,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            **({} if model_name is None else {"model_name": model_name}),
        )
    raise ValueError(f"Unknown strategy '{name}'")
