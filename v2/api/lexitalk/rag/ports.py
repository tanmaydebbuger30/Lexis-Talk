"""Interfaces for the heavy dependencies. Production adapters live in
``adapters.py``; tests use in-memory fakes. Nothing here imports torch."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    title: str
    page: int | None
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    similarity: float  # cosine similarity in [0, 1]


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    def count(self) -> int: ...
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...
    def query(self, embedding: Sequence[float], k: int) -> list[Hit]: ...


class LLM(Protocol):
    async def complete(self, messages: Sequence[ChatMessage]) -> str: ...
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]: ...


class ASR(Protocol):
    def transcribe(self, audio: bytes, lang_hint: str) -> tuple[str, float]:
        """Return (text, duration_seconds). Raises AudioError on bad input."""
        ...


class TTS(Protocol):
    def synthesize(self, text: str, lang: str) -> bytes: ...


class AudioError(ValueError):
    pass
