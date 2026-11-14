from __future__ import annotations

import hashlib
import math
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lexitalk.config import Settings
from lexitalk.main import Services, create_app
from lexitalk.rag.pipeline import RagPipeline
from lexitalk.rag.ports import AudioError, ChatMessage, Chunk, Hit
from lexitalk.rag.retriever import Retriever

DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words: deterministic, and similar texts get similar vectors."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.sha256(w.encode()).hexdigest(), 16) % DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class FakeIndex:
    def __init__(self) -> None:
        self.items: list[tuple[Chunk, list[float]]] = []

    def count(self) -> int:
        return len(self.items)

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        self.items.extend((c, list(e)) for c, e in zip(chunks, embeddings, strict=True))

    def query(self, embedding: Sequence[float], k: int) -> list[Hit]:
        scored = [Hit(c, sum(a * b for a, b in zip(embedding, e, strict=True))) for c, e in self.items]
        return sorted(scored, key=lambda h: h.similarity, reverse=True)[:k]


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []
        self.reply = "Under 3 U.S.C. § 102 the President receives compensation [S1]."
        self.fail_stream = False

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        if messages[0].content.startswith("Translate"):
            return "what is the salary of the president"
        return self.reply

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        # split into awkward 3-char pieces to exercise the stream masker
        for i in range(0, len(self.reply), 3):
            if self.fail_stream and i > 6:
                raise RuntimeError("upstream died")
            yield self.reply[i : i + 3]


class FakeASR:
    def transcribe(self, audio: bytes, lang_hint: str) -> tuple[str, float]:
        if audio == b"garbage":
            raise AudioError("Could not decode audio.")
        return "what does title 3 say about the president", 2.5


class FakeTTS:
    def synthesize(self, text: str, lang: str) -> bytes:
        return b"ID3fake"


CORPUS = [
    Chunk(
        "usc-title-03:00000",
        "usc-title-03",
        "U.S. Code Title 3",
        4,
        "The President shall receive in full for his services during the term for which he shall have been elected compensation.",
    ),
    Chunk(
        "usc-title-02:00000",
        "usc-title-02",
        "U.S. Code Title 2",
        10,
        "The Capitol Police shall protect the Members of Congress.",
    ),
    Chunk(
        "usc-title-01:00000",
        "usc-title-01",
        "U.S. Code Title 1",
        1,
        "In determining the meaning of any Act of Congress words importing the singular include the plural.",
    ),
]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        env="development",
        allowed_origins=["https://lexitalk.example"],
        rate_limit_per_minute=100,
        daily_request_budget=100,
        vector_store_dir=tmp_path / "vs",
        max_message_chars=500,
        max_audio_bytes=1024,
        tts_enabled=False,
    )


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def index() -> FakeIndex:
    idx = FakeIndex()
    idx.add(CORPUS, FakeEmbedder().encode([c.text for c in CORPUS]))
    return idx


def make_client(settings: Settings, llm: FakeLLM, index: FakeIndex, tts: FakeTTS | None = None) -> TestClient:
    retriever = Retriever(FakeEmbedder(), index, min_score=0.2)
    pipeline = RagPipeline(retriever, llm, top_k=3, max_context_chars=2000, snippet_chars=200, max_history_turns=4)
    svcs = Services(pipeline=pipeline, index_count=index.count, asr=FakeASR(), tts=tts)
    return TestClient(create_app(settings, svcs))


@pytest.fixture
def client(settings: Settings, llm: FakeLLM, index: FakeIndex):
    with make_client(settings, llm, index) as c:
        yield c
