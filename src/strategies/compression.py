"""Context compression strategies."""

from __future__ import annotations

import inspect
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
        method: str = "llmlingua2",
        max_chunk_chars: int = 1200,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device_map: str = "auto",
    ) -> None:
        if compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")
        if min_chars < 0:
            raise ValueError("min_chars must be non-negative")
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive")
        if method not in {"llmlingua2", "lexical"}:
            raise ValueError('method must be either "llmlingua2" or "lexical"')

        self.compression_ratio = min(compression_ratio, 1.0)
        self.min_chars = min_chars
        self.method = method
        self.max_chunk_chars = max_chunk_chars
        self.model_name = model_name
        self.device_map = self._resolve_device_map(device_map)
        self._prompt_compressor: Any | None = None

    def prepare(self, example: dict[str, Any]) -> StrategyResult:
        query = str(example.get("query") or "")
        context = str(example.get("context") or "")
        original_length = len(context)
        target_chars = self._target_chars(original_length)

        if self.method == "llmlingua2":
            self._get_prompt_compressor()

        if not context or original_length <= target_chars:
            compressed_context = context
            selected_chunks = 1 if context else 0
        elif self.method == "llmlingua2":
            compressed_context = self._compress_with_llmlingua2(
                context=context,
                query=query,
                target_chars=target_chars,
            )
            selected_chunks = None
        else:
            compressed_context, selected_chunks = self._compress_lexically(
                context=context,
                query=query,
                target_chars=target_chars,
            )

        compressed_length = len(compressed_context)
        metadata = {
            "method": self.method,
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
        if self.method == "llmlingua2":
            metadata["model_name"] = self.model_name
            metadata["device_map"] = self.device_map

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

    def _resolve_device_map(self, device_map: str) -> str:
        if device_map != "auto":
            return device_map

        try:
            import torch
        except ImportError:
            return "cpu"

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _compress_with_llmlingua2(
        self,
        context: str,
        query: str,
        target_chars: int,
    ) -> str:
        compressor = self._get_prompt_compressor()
        rate = max(0.01, min(1.0, target_chars / len(context)))

        kwargs: dict[str, Any] = {
            "rate": rate,
            "force_tokens": ["\n", "?"],
            "drop_consecutive": True,
        }
        if query:
            kwargs["question"] = query

        result = self._call_compress_prompt(compressor, context, kwargs)
        return self._extract_compressed_prompt(result)

    def _get_prompt_compressor(self) -> Any:
        if self._prompt_compressor is not None:
            return self._prompt_compressor

        try:
            from llmlingua import PromptCompressor
        except ImportError as exc:
            raise ImportError(
                "LLMLingua-2 compression requires the llmlingua package. "
                "Install it with: pip install llmlingua"
            ) from exc

        self._prompt_compressor = PromptCompressor(
            model_name=self.model_name,
            use_llmlingua2=True,
            device_map=self.device_map,
        )
        return self._prompt_compressor

    def _call_compress_prompt(
        self,
        compressor: Any,
        context: str,
        kwargs: dict[str, Any],
    ) -> Any:
        compress_prompt = compressor.compress_prompt
        accepted_kwargs = self._accepted_kwargs(compress_prompt, kwargs)

        try:
            return compress_prompt(context, **accepted_kwargs)
        except TypeError:
            accepted_kwargs.pop("question", None)
            return compress_prompt(context, **accepted_kwargs)

    def _accepted_kwargs(self, method: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return kwargs

        parameters = signature.parameters
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return kwargs

        return {key: value for key, value in kwargs.items() if key in parameters}

    def _extract_compressed_prompt(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            compressed = result.get("compressed_prompt")
            if compressed is None:
                compressed = result.get("compressed_prompt_list")
            if isinstance(compressed, list):
                return "\n\n".join(str(item) for item in compressed)
            if compressed is not None:
                return str(compressed)
        return str(result)

    def _compress_lexically(
        self,
        context: str,
        query: str,
        target_chars: int,
    ) -> tuple[str, int]:
        chunks = self._split_context(context)
        selected = self._select_chunks(chunks, query, target_chars)
        return "\n\n".join(chunk["text"] for chunk in selected), len(selected)

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
