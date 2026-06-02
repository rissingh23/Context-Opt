"""Shared interface for context strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    query: str
    context: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextStrategy(Protocol):
    name: str

    def prepare(self, example: dict[str, Any]) -> StrategyResult:
        """Return the context/prompt payload for one normalized example."""
