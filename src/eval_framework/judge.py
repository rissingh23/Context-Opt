"""LLM judge interface for scoring answer quality 0-1."""

from __future__ import annotations

import json
import re
import time
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
            "You are evaluating an AI system's answer against a reference answer. "
            "Score the answer on a continuous scale from 0.0 to 1.0 — do not round to 0, 0.5, or 1.\n\n"
            f"Task type: {task_type}\n"
            f"Question: {query}\n"
            f"Reference answer: {reference_answer}\n"
            f"AI answer: {prediction}\n\n"
            "Scoring rubric:\n"
            "0.9-1.0: all key facts correct and complete\n"
            "0.7-0.9: mostly correct, minor omissions or imprecision\n"
            "0.5-0.7: partially correct, some relevant content but missing important details\n"
            "0.3-0.5: some relevant content but significant errors or omissions\n"
            "0.1-0.3: mostly incorrect, only superficial relevance\n"
            "0.0-0.1: completely wrong or irrelevant\n\n"
            'Respond with only a JSON object: {"score": <float>, "reason": "<one sentence>"}'
        )
        for attempt in range(3):
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
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return JudgeResult(score=None, reason=f"Judge error after retries: {exc}", judge_model=self.judge_model)


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
