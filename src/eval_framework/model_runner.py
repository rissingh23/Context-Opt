"""Model runner abstraction for eval.

Supported providers: mock, vertexai.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResult:
    prediction: str
    model: str
    input_tokens: int
    output_tokens: int
    model_latency_sec: float
    estimated_cost: float


def count_tokens_rough(text: str) -> int:
    return max(1, len(text) // 4)


class MockModelRunner:
    """Cheap local model stand-in for testing table generation."""

    def __init__(self, model_name: str = "mock_model") -> None:
        self.model_name = model_name

    def generate(self, prompt: str, example: dict) -> ModelResult:
        start = time.perf_counter()
        prediction = example.get("reference_answer", "")
        input_tokens = count_tokens_rough(prompt)
        output_tokens = count_tokens_rough(prediction)
        latency = time.perf_counter() - start

        return ModelResult(
            prediction=prediction,
            model=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_latency_sec=latency,
            estimated_cost=0.0,
        )


class VertexAIModelRunner:
    """Answer model backed by Gemini via Vertex AI."""

    # Gemini 2.5 Flash pricing per token
    _COST_PER_INPUT_TOKEN = 0.075 / 1_000_000
    _COST_PER_OUTPUT_TOKEN = 0.30 / 1_000_000

    def __init__(self, model_name: str = "gemini-2.5-flash", project: str = "", location: str = "us-central1") -> None:
        from google import genai
        self.model_name = model_name
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def generate(self, prompt: str, example: dict) -> ModelResult:
        start = time.perf_counter()
        response = self.client.models.generate_content(model=self.model_name, contents=prompt)
        latency = time.perf_counter() - start

        prediction = response.text or ""
        input_tokens = count_tokens_rough(prompt)
        output_tokens = count_tokens_rough(prediction)
        estimated_cost = (
            input_tokens * self._COST_PER_INPUT_TOKEN
            + output_tokens * self._COST_PER_OUTPUT_TOKEN
        )

        return ModelResult(
            prediction=prediction,
            model=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_latency_sec=latency,
            estimated_cost=estimated_cost,
        )


def build_model_runner(provider: str, model_name: str, **kwargs) -> MockModelRunner | VertexAIModelRunner:
    if provider == "mock":
        return MockModelRunner(model_name=model_name)
    if provider == "vertexai":
        return VertexAIModelRunner(
            model_name=model_name,
            project=kwargs.get("project", ""),
            location=kwargs.get("location", "us-central1"),
        )
    raise ValueError(f"Unknown provider '{provider}'. Supported: mock, vertexai")
