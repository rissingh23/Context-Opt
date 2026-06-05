"""Summarization strategy using Gemini via Vertex AI."""

from __future__ import annotations

from typing import Any

from src.strategies.base import StrategyResult


class GeminiSummarizationStrategy:
    name = "gemini_summarization"

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        project: str = "",
        location: str = "us-central1",
    ) -> None:
        from google import genai
        self.model_name = model_name
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def _summarize(self, context: str, query: str) -> str:
        prompt = (
            "Given the following question, summarize the text below keeping all information "
            "relevant to answering it. Be concise but preserve key facts.\n\n"
            f"Question: {query}\n\nText: {context}\n\nSummary:"
        )
        response = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return (response.text or "").strip()

    def prepare(self, example: dict[str, Any]) -> StrategyResult:
        query = example["query"]
        context = example["context"]
        summary = self._summarize(context, query)

        prompt = f"Context: {summary}\n\nQuestion: {query}\n\nAnswer:"

        return StrategyResult(
            strategy=self.name,
            query=query,
            context=summary,
            prompt=prompt,
            metadata={
                "original_context_length": len(context.split()),
                "summary_length": len(summary.split()),
            },
        )
