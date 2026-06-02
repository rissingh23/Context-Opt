"""Full-context baseline strategy."""

from __future__ import annotations

from src.strategies.base import StrategyResult


class FullContextStrategy:
    name = "full_context"

    def prepare(self, example: dict) -> StrategyResult:
        query = example["query"]
        context = example["context"]
        prompt = f"Answer the question using the context.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        return StrategyResult(strategy=self.name, query=query, context=context, prompt=prompt)
