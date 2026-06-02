"""LLM judge interface.

The default judge is disabled so local smoke tests do not spend money. Later we
can add OpenAI-compatible, Anthropic, or local-model judge clients behind this
same shape.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeResult:
    score: float | None
    reason: str
    judge_model: str


class BaseJudge:
    judge_model = "none"

    def score(
        self,
        *,
        task_type: str,
        query: str,
        reference_answer: str,
        prediction: str,
    ) -> JudgeResult:
        raise NotImplementedError


class DisabledJudge(BaseJudge):
    judge_model = "disabled"

    def score(
        self,
        *,
        task_type: str,
        query: str,
        reference_answer: str,
        prediction: str,
    ) -> JudgeResult:
        return JudgeResult(score=None, reason="LLM judge disabled for this run.", judge_model=self.judge_model)


def build_judge(name: str) -> BaseJudge:
    if name == "disabled":
        return DisabledJudge()
    raise ValueError(f"Unknown judge '{name}'. Supported: disabled")
