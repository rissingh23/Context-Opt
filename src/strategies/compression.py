"""Lightweight extractive compression strategy."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.strategies.base import StrategyResult


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class _CompressionStrategyResult(StrategyResult):
    @property
    def prepared_context(self) -> str:
        """Alias used by some ad hoc strategy smoke tests."""
        return self.context


class CompressionStrategy:
    name = "compression"

    def __init__(
        self,
        compression_ratio: float = 0.3,
        min_chars: int = 1000,
        max_chunk_chars: int = 1200,
    ) -> None:
        if compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")
        if min_chars < 0:
            raise ValueError("min_chars must be non-negative")
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive")

        self.compression_ratio = min(compression_ratio, 1.0)
        self.min_chars = min_chars
        self.max_chunk_chars = max_chunk_chars

    def prepare(self, example: dict[str, Any]) -> StrategyResult:
        query = str(example.get("query") or "")
        context = str(example.get("context") or "")
        original_length = len(context)
        target_chars = self._target_chars(original_length)

        if not context or original_length <= target_chars:
            compressed_context = context
            selected_chunks = 1 if context else 0
        else:
            chunks = self._split_context(context)
            selected = self._select_chunks(chunks, query, target_chars)
            compressed_context = "\n\n".join(chunk["text"] for chunk in selected)
            selected_chunks = len(selected)

        compressed_length = len(compressed_context)
        metadata = {
            "original_context_chars": original_length,
            "compressed_context_chars": compressed_length,
            "target_compression_ratio": self.compression_ratio,
            "actual_compression_ratio": (
                compressed_length / original_length if original_length else 0.0
            ),
            "target_chars": target_chars,
            "selected_chunks": selected_chunks,
            "min_chars": self.min_chars,
            "max_chunk_chars": self.max_chunk_chars,
        }
        prompt = (
            "Answer the question using the compressed context.\n\n"
            f"Context:\n{compressed_context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

        return _CompressionStrategyResult(
            strategy=self.name,
            query=query,
            context=compressed_context,
            prompt=prompt,
            metadata=metadata,
        )

    def _target_chars(self, original_length: int) -> int:
        if original_length <= 0:
            return 0
        ratio_budget = math.ceil(original_length * self.compression_ratio)
        return min(original_length, max(ratio_budget, self.min_chars))

    def _split_context(self, context: str) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        for paragraph in re.split(r"\n\s*\n+", context):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            sentences = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(paragraph)]
            for sentence in sentences:
                if not sentence:
                    continue
                for chunk in self._split_long_text(sentence):
                    chunks.append({"index": len(chunks), "text": chunk})

        if chunks:
            return chunks

        return [
            {"index": index, "text": chunk}
            for index, chunk in enumerate(self._split_long_text(context.strip()))
        ]

    def _split_long_text(self, text: str) -> list[str]:
        if len(text) <= self.max_chunk_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chunk_chars, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
            while start < len(text) and text[start].isspace():
                start += 1
        return chunks

    def _select_chunks(
        self,
        chunks: list[dict[str, Any]],
        query: str,
        target_chars: int,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []

        query_terms = self._keywords(query)
        scored = [
            (self._score_chunk(chunk["text"], query_terms), chunk["index"], chunk)
            for chunk in chunks
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))

        selected: list[dict[str, Any]] = []
        selected_length = 0
        for _, _, chunk in scored:
            chunk_length = len(chunk["text"])
            join_cost = 2 if selected else 0
            if selected and selected_length + join_cost + chunk_length > target_chars:
                continue
            selected.append(chunk)
            selected_length += join_cost + chunk_length
            if selected_length >= target_chars:
                break

        if not selected:
            selected = [scored[0][2]]

        return sorted(selected, key=lambda chunk: chunk["index"])

    def _keywords(self, text: str) -> set[str]:
        return {
            token
            for token in (match.group(0).lower() for match in _WORD_RE.finditer(text))
            if len(token) > 2 and token not in _STOPWORDS
        }

    def _score_chunk(self, chunk: str, query_terms: set[str]) -> float:
        if not query_terms:
            return 0.0

        chunk_terms = [
            match.group(0).lower()
            for match in _WORD_RE.finditer(chunk)
            if len(match.group(0)) > 2
        ]
        if not chunk_terms:
            return 0.0

        counts = Counter(chunk_terms)
        overlap = query_terms.intersection(counts)
        coverage = len(overlap) / len(query_terms)
        frequency = sum(counts[term] for term in overlap)
        density = frequency / len(chunk_terms)

        return coverage + density
