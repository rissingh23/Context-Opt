"""Hybrid retrieval + compression/summarization strategies."""

from __future__ import annotations

from typing import Any

from src.strategies.base import StrategyResult
from src.strategies.compression import CompressionStrategy
from src.strategies.retrieval import RetrievalStrategy, build_retrieval_prompt


class RetrievalCompressionStrategy:
    name = "retrieval_compression"

    def __init__(
        self,
        top_k: int = 5,
        chunk_chars: int = 2000,
        overlap_chars: int = 200,
        compression_ratio: float = 0.3,
    ) -> None:
        self.retrieval = RetrievalStrategy(top_k=top_k, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        self.compression = CompressionStrategy(compression_ratio=compression_ratio)

    def prepare(self, example: dict[str, Any]) -> StrategyResult:
        retrieval_result = self.retrieval.prepare(example)

        retrieved_example = {**example, "context": retrieval_result.context}
        compression_result = self.compression.prepare(retrieved_example)

        prompt = build_retrieval_prompt(example, compression_result.context)

        return StrategyResult(
            strategy=self.name,
            query=example["query"],
            context=compression_result.context,
            prompt=prompt,
            metadata={
                "retrieval_context_length": len(retrieval_result.context),
                "compressed_context_length": len(compression_result.context),
                **{f"retrieval_{k}": v for k, v in retrieval_result.metadata.items()},
                **{f"compression_{k}": v for k, v in compression_result.metadata.items()},
            },
        )


class RetrievalSummaryStrategy:
    name = "retrieval_summary"

    def __init__(
        self,
        top_k: int = 5,
        chunk_chars: int = 2000,
        overlap_chars: int = 200,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    ) -> None:
        from src.strategies.summarization import SummarizationStrategy
        self.retrieval = RetrievalStrategy(top_k=top_k, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        self.summarization = SummarizationStrategy(model_name=model_name)

    def prepare(self, example: dict[str, Any]) -> StrategyResult:
        retrieval_result = self.retrieval.prepare(example)

        retrieved_example = {**example, "context": retrieval_result.context}
        summary_result = self.summarization.prepare(retrieved_example)

        prompt = build_retrieval_prompt(example, summary_result.context)

        return StrategyResult(
            strategy=self.name,
            query=example["query"],
            context=summary_result.context,
            prompt=prompt,
            metadata={
                "retrieval_context_length": len(retrieval_result.context),
                "summary_context_length": len(summary_result.context),
                **{f"retrieval_{k}": v for k, v in retrieval_result.metadata.items()},
                **{f"summary_{k}": v for k, v in summary_result.metadata.items()},
            },
        )
