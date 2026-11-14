"""Public API contract. The frontend's zod schemas mirror these exactly."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

InputLang = Literal["auto", "en", "hi"]
OutputLang = Literal["en", "hi"]
SUPPORTED_INPUT: tuple[InputLang, ...] = ("auto", "en", "hi")
SUPPORTED_OUTPUT: tuple[OutputLang, ...] = ("en", "hi")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryTurn(Strict):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(Strict):
    message: str = Field(min_length=1)
    input_lang: InputLang = "auto"
    output_lang: OutputLang = "en"
    history: list[HistoryTurn] = Field(default_factory=list)


class Redaction(Strict):
    type: str
    count: int


class Source(Strict):
    id: str
    document_id: str
    title: str
    page: int | None
    snippet: str
    score: float = Field(description="Heuristic relevance, 0-1. Not a calibrated probability.")
    rank: int
    used_in_prompt: bool


class ChatResponse(Strict):
    answer: str
    detected_lang: Literal["en", "hi"]
    query_used_for_retrieval: str
    grounded: bool
    sources: list[Source]
    redactions: list[Redaction]
    request_id: str


class TranscribeResponse(Strict):
    text: str
    input_lang: InputLang
    duration_seconds: float


class TtsRequest(Strict):
    text: str = Field(min_length=1, max_length=4000)
    lang: OutputLang = "en"


class ConfigResponse(Strict):
    input_languages: list[InputLang]
    output_languages: list[OutputLang]
    tts_enabled: bool
    max_message_chars: int
    max_audio_seconds: int
    max_audio_bytes: int


class HealthResponse(Strict):
    status: Literal["ok", "degraded"]
    index_ready: bool
    chunks_indexed: int
    version: str


class StreamMeta(Strict):
    """First SSE event (`event: meta`)."""

    request_id: str
    detected_lang: Literal["en", "hi"]
    query_used_for_retrieval: str
    grounded: bool
    sources: list[Source]


class StreamToken(Strict):
    """`event: token` - one masked text delta."""

    delta: str


class StreamDone(Strict):
    """Final `event: done`."""

    redactions: list[Redaction]


class ErrorBody(Strict):
    code: str
    message: str
    request_id: str


class ErrorResponse(Strict):
    error: ErrorBody
