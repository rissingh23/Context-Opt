"""Retrieval-based context strategy.

This is intentionally a simple lexical retriever first: no API calls, no
embedding service, and no hidden model dependency. It gives us a cheap baseline
that the later eval pipeline can run many times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.strategies.base import StrategyResult


DEFAULT_SUMMARY_QUERY = "summary key findings main points conclusions important details"


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start_char: int
    end_char: int


def chunk_text(text: str, chunk_chars: int = 2000, overlap_chars: int = 200) -> list[TextChunk]:
    """Split long context into overlapping character windows."""

    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    chunks: list[TextChunk] = []
    start = 0
    index = 0

    # Step 1: walk through the document with overlap so boundary facts are less
    # likely to disappear between adjacent chunks.
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(TextChunk(index=index, text=chunk, start_char=start, end_char=end))
            index += 1
        if end == len(text):
            break
        start = end - overlap_chars

    return chunks


def rank_chunks_tfidf(query: str, chunks: list[TextChunk], top_k: int) -> list[tuple[TextChunk, float]]:
    """Rank chunks by TF-IDF cosine similarity to the query."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not chunks:
        return []

    # Step 2: fit TF-IDF on the query plus this example's chunks. This makes the
    # retrieval completely local to one LongBench example and easy to reproduce.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50_000)
    try:
        matrix = vectorizer.fit_transform([query] + [chunk.text for chunk in chunks])
    except ValueError:
        return [(chunk, 0.0) for chunk in chunks[: min(top_k, len(chunks))]]

    # Step 3: compare the query vector against every chunk vector.
    try:
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    except NotFittedError:
        return [(chunk, 0.0) for chunk in chunks[: min(top_k, len(chunks))]]
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda item: item[1], reverse=True)
    return [(chunk, float(score)) for chunk, score in ranked[: min(top_k, len(ranked))]]


def build_retrieval_prompt(example: dict[str, Any], retrieved_context: str) -> str:
    """Create the prompt that will be sent to the model after retrieval."""

    query = example["query"]
    if example.get("task_type") == "summarization" or not query.strip():
        return f"Write a concise summary using the retrieved context.\n\nContext:\n{retrieved_context}\n\nSummary:"

    return (
        "Answer the question using the retrieved context.\n\n"
        f"Context:\n{retrieved_context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )


class RetrievalStrategy:
    name = "retrieval"

    def __init__(self, top_k: int = 5, chunk_chars: int = 2000, overlap_chars: int = 200) -> None:
        self.top_k = top_k
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars

    def prepare(self, example: dict) -> StrategyResult:
        query = example["query"]
        context = example["context"]

        # Step 1: split the long context into chunks that can be ranked.
        chunks = chunk_text(context, chunk_chars=self.chunk_chars, overlap_chars=self.overlap_chars)

        # Step 2: use the real question for QA tasks. Summarization tasks often
        # have an empty query, so give TF-IDF a small generic summary query.
        retrieval_query = query.strip() or DEFAULT_SUMMARY_QUERY

        # Step 3: rank chunks, then restore selected chunks to document order so
        # the final context is easier for the answer model to read.
        ranked_chunks = rank_chunks_tfidf(retrieval_query, chunks, top_k=self.top_k)
        rank_by_index = {chunk.index: rank for rank, (chunk, _score) in enumerate(ranked_chunks, start=1)}
        selected_in_order = sorted(ranked_chunks, key=lambda item: item[0].index)
        retrieved_context = "\n\n".join(chunk.text for chunk, _score in selected_in_order)

        # Step 4: build the final prompt from only the retrieved context.
        prompt = build_retrieval_prompt(example, retrieved_context)

        # Step 5: keep enough metadata to debug what retrieval selected later.
        metadata = {
            "top_k": self.top_k,
            "chunk_chars": self.chunk_chars,
            "overlap_chars": self.overlap_chars,
            "num_chunks": len(chunks),
            "retrieval_query": retrieval_query,
            "selected_chunks": [
                {
                    "index": chunk.index,
                    "retrieval_rank": rank_by_index[chunk.index],
                    "score": score,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "length_chars": len(chunk.text),
                }
                for chunk, score in selected_in_order
            ],
        }

        return StrategyResult(
            strategy=self.name,
            query=query,
            context=retrieved_context,
            prompt=prompt,
            metadata=metadata,
        )
