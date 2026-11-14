"""Mock-mode server for frontend development: the real FastAPI app, routes,
masking, rate limits and SSE, with deterministic fakes instead of Legal-BERT,
Whisper and OpenAI. No API key, no torch, no network.

    LEXITALK_ALLOWED_ORIGINS=http://localhost:5173 uvicorn lexitalk.devserver:app --port 8000

Refuses to start when LEXITALK_ENV=production.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import AsyncIterator, Sequence

from .config import get_settings
from .main import Services, create_app
from .rag.pipeline import RagPipeline
from .rag.ports import AudioError, ChatMessage, Chunk, Hit
from .rag.retriever import Retriever

_DIM = 128

_CORPUS = [
    Chunk(
        "usc-title-03:0001",
        "usc-title-03",
        "U.S. Code Title 3 — The President",
        4,
        "§102. Compensation of the President. The President shall receive in full for his services during the term "
        "for which he shall have been elected compensation in the aggregate amount of $400,000 a year, to be paid "
        "monthly, and in addition an expense allowance of $50,000 to assist in defraying expenses relating to or "
        "resulting from the discharge of his official duties.",
    ),
    Chunk(
        "usc-title-03:0002",
        "usc-title-03",
        "U.S. Code Title 3 — The President",
        2,
        "§19. Vacancy in offices of both President and Vice President; officers eligible to act. (a)(1) If, by "
        "reason of death, resignation, removal from office, inability, or failure to qualify, there is neither a "
        "President nor Vice President to discharge the powers and duties of the office of President, then the "
        "Speaker of the House of Representatives shall, upon his resignation as Speaker and as Representative in "
        "Congress, act as President.",
    ),
    Chunk(
        "usc-title-01:0001",
        "usc-title-01",
        "U.S. Code Title 1 — General Provisions",
        1,
        "§1. Words denoting number, gender, and so forth. In determining the meaning of any Act of Congress, "
        "unless the context indicates otherwise— words importing the singular include and apply to several "
        "persons, parties, or things; words importing the plural include the singular.",
    ),
    Chunk(
        "usc-title-02:0001",
        "usc-title-02",
        "U.S. Code Title 2 — The Congress",
        112,
        "§1961. Capitol Police. The Capitol Police shall police the United States Capitol Buildings and Grounds "
        "under the direction of the Capitol Police Board.",
    ),
]


class _Embedder:
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * _DIM
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.sha256(w.encode()).hexdigest(), 16) % _DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class _Index:
    def __init__(self) -> None:
        self.items = list(zip(_CORPUS, _Embedder().encode([c.text for c in _CORPUS]), strict=True))

    def count(self) -> int:
        return len(self.items)

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    def query(self, embedding: Sequence[float], k: int) -> list[Hit]:
        hits = [Hit(c, sum(a * b for a, b in zip(embedding, e, strict=True))) for c, e in self.items]
        return sorted(hits, key=lambda h: h.similarity, reverse=True)[:k]


class _LLM:
    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        if messages[0].content.startswith("Translate"):
            return "What does the law say about the President's oath?"
        return "".join([d async for d in self.stream(messages)])

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        hindi = "Respond in Hindi" in messages[0].content
        has_sources = '<source id="S1"' in messages[-1].content
        if not has_sources:
            text = "The indexed documents do not cover this question."
        elif hindi:
            text = "(mock) स्रोत [S1] के अनुसार राष्ट्रपति को $400,000 प्रति वर्ष मिलते हैं।"
        else:
            text = (
                "(mock answer) According to the retrieved passage [S1], the President receives compensation of "
                "$400,000 a year, paid monthly, plus a $50,000 expense allowance.\n\nSee also [S2] for related "
                "provisions."
            )
        for i in range(0, len(text), 4):
            await asyncio.sleep(0.02)
            yield text[i : i + 4]


class _ASR:
    def transcribe(self, audio: bytes, lang_hint: str) -> tuple[str, float]:
        if len(audio) < 100:
            raise AudioError("Recording is too short.")
        return "What compensation does the President receive?", 2.0


def _build():
    s = get_settings()
    if s.is_production:
        raise RuntimeError("devserver must not run in production")
    retriever = Retriever(_Embedder(), _Index(), min_score=0.15)
    pipeline = RagPipeline(
        retriever,
        _LLM(),
        top_k=4,
        max_context_chars=s.max_context_chars,
        snippet_chars=s.snippet_chars,
        max_history_turns=s.max_history_turns,
    )
    return create_app(s, Services(pipeline=pipeline, index_count=lambda: len(_CORPUS), asr=_ASR(), tts=None))


app = _build()
