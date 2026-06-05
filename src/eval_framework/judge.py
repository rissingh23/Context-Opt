"""LLM judge interface for scoring answer quality 0-1."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeResult:
    score: float | None
    reason: str
    judge_model: str


class DisabledJudge:
    judge_model = "disabled"

    def score(self, *, task_type: str, query: str, reference_answer: str, prediction: str) -> JudgeResult:
        return JudgeResult(score=None, reason="LLM judge disabled for this run.", judge_model=self.judge_model)


class VertexAIJudge:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        project: str = "",
        location: str = "us-central1",
    ) -> None:
        from google import genai
        self.judge_model = model_name
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def score(self, *, task_type: str, query: str, reference_answer: str, prediction: str) -> JudgeResult:
        prompt = (
            "You are evaluating an AI system's answer against a reference answer.\n\n"
            f"Task type: {task_type}\n"
            f"Question: {query}\n"
            f"Reference answer: {reference_answer}\n"
            f"AI answer: {prediction}\n\n"
            "Score the AI answer from 0.0 to 1.0:\n"
            "- 1.0: completely correct and complete\n"
            "- 0.5: partially correct or missing details\n"
            "- 0.0: incorrect or irrelevant\n\n"
            'Respond with only a JSON object: {"score": <float 0-1>, "reason": "<one sentence>"}'
        )
        try:
            response = self.client.models.generate_content(model=self.judge_model, contents=prompt)
            text = (response.text or "").strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return JudgeResult(
                    score=float(data["score"]),
                    reason=str(data.get("reason", "")),
                    judge_model=self.judge_model,
                )
            return JudgeResult(score=None, reason=f"Parse error: {text}", judge_model=self.judge_model)
        except Exception as exc:
            return JudgeResult(score=None, reason=f"Judge error: {exc}", judge_model=self.judge_model)


def build_judge(name: str, **kwargs) -> DisabledJudge | VertexAIJudge:
    if name == "disabled":
        return DisabledJudge()
    if name == "vertexai":
        return VertexAIJudge(
            model_name=kwargs.get("judge_model", "gemini-2.5-flash"),
            project=kwargs.get("project", ""),
            location=kwargs.get("location", "us-central1"),
        )
    raise ValueError(f"Unknown judge '{name}'. Supported: disabled, vertexai")
