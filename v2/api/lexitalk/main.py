"""FastAPI application factory.

No `from __future__ import annotations` here: FastAPI resolves dependency
annotations at runtime.
"""

import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import __version__
from .config import Settings, get_settings
from .rag.pipeline import RagPipeline
from .rag.ports import ASR, TTS, AudioError
from .schemas import (
    SUPPORTED_INPUT,
    SUPPORTED_OUTPUT,
    ChatRequest,
    ChatResponse,
    ConfigResponse,
    HealthResponse,
    InputLang,
    StreamDone,
    StreamMeta,
    StreamToken,
    TranscribeResponse,
    TtsRequest,
)
from .security.audit import audit, configure_logging, hash_client
from .security.limits import DailyBudget, SlidingWindowLimiter
from .security.middleware import RequestContextMiddleware, client_ip
from .security.pii import StreamMasker, mask, merge_counts

log = logging.getLogger("lexitalk")

ALLOWED_AUDIO_TYPES = frozenset(
    {"audio/webm", "audio/ogg", "audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp4", "audio/x-m4a"}
)


@dataclass
class Services:
    pipeline: RagPipeline
    index_count: Callable[[], int]
    asr: ASR | None
    tts: TTS | None


def build_services(s: Settings) -> Services:  # pragma: no cover - needs models + network
    from .rag.adapters import ChromaIndex, LegalBertEmbedder, OpenAIChat, OpenAITTS, WhisperASR
    from .rag.retriever import Retriever

    if s.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set")
    key = s.openai_api_key.get_secret_value()
    index = ChromaIndex(s.vector_store_dir, s.collection_name)
    retriever = Retriever(LegalBertEmbedder(s.embedding_model), index)
    llm = OpenAIChat(key, s.llm_model, s.llm_temperature, s.llm_max_tokens)
    pipeline = RagPipeline(
        retriever,
        llm,
        top_k=s.top_k,
        max_context_chars=s.max_context_chars,
        snippet_chars=s.snippet_chars,
        max_history_turns=s.max_history_turns,
    )
    return Services(
        pipeline=pipeline,
        index_count=index.count,
        asr=WhisperASR(s.asr_model, s.max_audio_seconds),
        tts=OpenAITTS(key) if s.tts_enabled else None,
    )


def svc(request: Request) -> Services:
    return request.app.state.services  # type: ignore[no-any-return]


def _err(status: int, code: str, message: str, rid: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": rid}},
        headers=headers,
    )


def _sse(event: str, payload: str) -> bytes:
    return f"event: {event}\ndata: {payload}\n\n".encode()


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    s = settings or get_settings()
    configure_logging()
    limiter = SlidingWindowLimiter(s.rate_limit_per_minute)
    budget = DailyBudget(s.daily_request_budget)
    salt = s.audit_salt.get_secret_value()
    if s.is_production and salt == "change-me-in-production":
        raise RuntimeError("Set LEXITALK_AUDIT_SALT in production")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services or build_services(s)
        yield

    app = FastAPI(
        title="LexiTalk API",
        version=__version__,
        lifespan=lifespan,
        docs_url=None if s.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if s.is_production else "/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
        max_age=600,
    )
    app.add_middleware(RequestContextMiddleware, max_body_bytes=s.max_audio_bytes + 64 * 1024, hsts=s.is_production)

    # ---------- errors: never echo request bodies back (they may contain PII) ----------
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(p) for p in e.get("loc", ())[1:]) or "body" for e in exc.errors()})
        return _err(422, "invalid_request", f"Invalid fields: {', '.join(fields)}", request.state.request_id)

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException) -> JSONResponse:
        code, _, message = str(exc.detail).partition(":")
        return _err(
            exc.status_code,
            code,
            message.strip() or code,
            request.state.request_id,
            dict(exc.headers) if exc.headers else None,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error rid=%s", getattr(request.state, "request_id", "?"))
        return _err(500, "internal_error", "Something went wrong.", getattr(request.state, "request_id", ""))

    # ---------- abuse controls ----------
    def guard(spend_budget: bool) -> Callable[[Request], str]:
        def dep(request: Request) -> str:
            client = hash_client(client_ip(request, s.trust_proxy_headers), salt)
            ok, retry = limiter.check(client)
            if not ok:
                audit("rate_limited", request_id=request.state.request_id, client=client, route=request.url.path)
                raise HTTPException(
                    429, "rate_limited: Too many requests. Please wait a moment.", headers={"Retry-After": str(retry)}
                )
            if spend_budget and not budget.consume():
                audit("budget_exhausted", request_id=request.state.request_id, route=request.url.path)
                raise HTTPException(503, "budget_exhausted: Daily usage limit reached. Try again tomorrow.")
            return client

        return dep

    llm_guard = Depends(guard(spend_budget=True))

    def check_message(req: ChatRequest) -> None:
        if len(req.message) > s.max_message_chars:
            raise HTTPException(422, f"message_too_long: Messages are limited to {s.max_message_chars} characters.")
        if not req.message.strip():
            raise HTTPException(422, "empty_message: Message is empty.")

    # ---------- routes ----------
    @app.get("/health", response_model=HealthResponse)
    def health(sv: Annotated[Services, Depends(svc)]) -> HealthResponse:
        n = sv.index_count()
        return HealthResponse(
            status="ok" if n else "degraded", index_ready=n > 0, chunks_indexed=n, version=__version__
        )

    @app.get("/api/config", response_model=ConfigResponse)
    def config(sv: Annotated[Services, Depends(svc)]) -> ConfigResponse:
        return ConfigResponse(
            input_languages=list(SUPPORTED_INPUT),
            output_languages=list(SUPPORTED_OUTPUT),
            tts_enabled=sv.tts is not None,
            max_message_chars=s.max_message_chars,
            max_audio_seconds=s.max_audio_seconds,
            max_audio_bytes=s.max_audio_bytes,
        )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(
        req: ChatRequest, request: Request, sv: Annotated[Services, Depends(svc)], client: str = llm_guard
    ) -> ChatResponse:
        check_message(req)
        t0 = time.perf_counter()
        p = await sv.pipeline.prepare(req.message, req.input_lang, req.output_lang, req.history)
        answer, redactions = await sv.pipeline.answer(p)
        rid = request.state.request_id
        audit(
            "chat",
            request_id=rid,
            client=client,
            stream=False,
            input_lang=p.detected_lang,
            output_lang=req.output_lang,
            message_chars=len(req.message),
            history_turns=len(req.history),
            redactions={r.type: r.count for r in redactions},
            grounded=p.grounded,
            documents=sorted({x.document_id for x in p.sources if x.used_in_prompt}),
            latency_ms=round((time.perf_counter() - t0) * 1000),
        )
        return ChatResponse(
            answer=answer,
            detected_lang=p.detected_lang,  # type: ignore[arg-type]
            query_used_for_retrieval=p.query_used_for_retrieval,
            grounded=p.grounded,
            sources=p.sources,
            redactions=redactions,
            request_id=rid,
        )

    @app.post("/api/chat/stream")
    async def chat_stream(
        req: ChatRequest, request: Request, sv: Annotated[Services, Depends(svc)], client: str = llm_guard
    ) -> StreamingResponse:
        check_message(req)
        rid = request.state.request_id
        t0 = time.perf_counter()
        p = await sv.pipeline.prepare(req.message, req.input_lang, req.output_lang, req.history)
        masker = StreamMasker()

        async def events() -> AsyncIterator[bytes]:
            meta = StreamMeta(
                request_id=rid,
                detected_lang=p.detected_lang,  # type: ignore[arg-type]
                query_used_for_retrieval=p.query_used_for_retrieval,
                grounded=p.grounded,
                sources=p.sources,
            )
            yield _sse("meta", meta.model_dump_json())
            outcome = "ok"
            try:
                async for delta in sv.pipeline.stream(p, masker):
                    if await request.is_disconnected():
                        outcome = "client_disconnected"
                        return
                    yield _sse("token", StreamToken(delta=delta).model_dump_json())
                redactions = sv.pipeline.stream_redactions(p, masker)
                yield _sse("done", StreamDone(redactions=redactions).model_dump_json())
            except Exception:
                outcome = "error"
                log.exception("stream failed rid=%s", rid)
                yield _sse(
                    "error",
                    json.dumps(
                        {"code": "stream_failed", "message": "The answer could not be completed.", "request_id": rid}
                    ),
                )
            finally:
                audit(
                    "chat",
                    request_id=rid,
                    client=client,
                    stream=True,
                    outcome=outcome,
                    input_lang=p.detected_lang,
                    output_lang=req.output_lang,
                    message_chars=len(req.message),
                    history_turns=len(req.history),
                    redactions=merge_counts(masker.counts, p.inbound_redactions),
                    grounded=p.grounded,
                    documents=sorted({x.document_id for x in p.sources if x.used_in_prompt}),
                    latency_ms=round((time.perf_counter() - t0) * 1000),
                )

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
        )

    @app.post("/api/transcribe", response_model=TranscribeResponse)
    async def transcribe(
        request: Request,
        sv: Annotated[Services, Depends(svc)],
        audio: Annotated[UploadFile, File()],
        input_lang: Annotated[InputLang, Form()] = "auto",
        client: str = Depends(guard(spend_budget=False)),
    ) -> TranscribeResponse:
        if sv.asr is None:
            raise HTTPException(503, "asr_unavailable: Voice input is not available.")
        ctype = (audio.content_type or "").split(";")[0].strip().lower()
        if ctype not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(415, "unsupported_media_type: Unsupported audio format.")
        data = await audio.read(s.max_audio_bytes + 1)
        if len(data) > s.max_audio_bytes:
            raise HTTPException(413, "payload_too_large: Recording is too large.")
        if not data:
            raise HTTPException(422, "empty_audio: Recording is empty.")
        try:
            text, duration = await run_in_threadpool(sv.asr.transcribe, data, input_lang)
        except AudioError as e:
            raise HTTPException(422, f"bad_audio: {e}") from e
        finally:
            del data  # do not retain audio beyond the request
        audit(
            "transcribe",
            request_id=request.state.request_id,
            client=client,
            input_lang=input_lang,
            duration_s=duration,
            transcript_chars=len(text),
            pii_detected=mask(text).counts,
        )
        # The transcript is returned unmasked to the speaker for review; it is
        # masked when it is sent to /api/chat like any other message.
        return TranscribeResponse(text=text, input_lang=input_lang, duration_seconds=duration)

    @app.post("/api/tts")
    async def tts(
        req: TtsRequest, request: Request, sv: Annotated[Services, Depends(svc)], client: str = llm_guard
    ) -> Response:
        if sv.tts is None:
            raise HTTPException(404, "tts_disabled: Text-to-speech is disabled.")
        audio = await run_in_threadpool(sv.tts.synthesize, mask(req.text).text, req.lang)
        audit("tts", request_id=request.state.request_id, client=client, chars=len(req.text), lang=req.lang)
        return Response(content=audio, media_type="audio/mpeg")

    return app
