"""Model runner abstraction for eval.

The framework needs a model-like component that turns a prompt into a prediction.
For now, `mock` makes the table runnable without API keys; real providers can be
added here without changing the eval loop.
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

        # This is intentionally not a real model. It lets us validate the eval
        # framework shape before plugging in Llama, Qwen, or API providers.
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


def build_model_runner(provider: str, model_name: str) -> MockModelRunner:
    if provider == "mock":
        return MockModelRunner(model_name=model_name)
    raise ValueError(f"Unknown provider '{provider}'. Supported: mock")
