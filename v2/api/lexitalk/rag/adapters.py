"""Production implementations of the ports. Heavy imports happen inside
constructors so the API package (and its tests) import without torch."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from .ports import AudioError, ChatMessage, Chunk, Hit

SAMPLE_RATE = 16_000


class LegalBertEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        # legal-bert is not a sentence-transformer checkpoint; SentenceTransformer
        # wraps it with mean pooling. Normalising makes cosine == dot product.
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


class ChromaIndex:
    def __init__(self, persist_dir: Path, collection: str) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return int(self.collection.count())

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        for i in range(0, len(chunks), 500):
            batch = chunks[i : i + 500]
            self.collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=[list(e) for e in embeddings[i : i + 500]],
                metadatas=[
                    {"document_id": c.document_id, "title": c.title, "page": -1 if c.page is None else c.page}
                    for c in batch
                ],
            )

    def query(self, embedding: Sequence[float], k: int) -> list[Hit]:
        r: Any = self.collection.query(query_embeddings=[list(embedding)], n_results=k)
        hits: list[Hit] = []
        for cid, doc, meta, dist in zip(
            r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0], strict=True
        ):
            page = int(meta.get("page", -1))
            hits.append(
                Hit(
                    Chunk(cid, str(meta["document_id"]), str(meta["title"]), None if page < 0 else page, doc),
                    similarity=1.0 - float(dist),  # cosine distance → similarity
                )
            )
        return hits


class OpenAIChat:
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(
            api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens, timeout=60, max_retries=2
        )

    @staticmethod
    def _convert(messages: Sequence[ChatMessage]) -> list[tuple[str, str]]:
        return [
            ("human" if m.role == "user" else "ai" if m.role == "assistant" else "system", m.content) for m in messages
        ]

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        resp = await self.llm.ainvoke(self._convert(messages))
        return str(resp.content)

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        async for chunk in self.llm.astream(self._convert(messages)):
            if chunk.content:
                yield str(chunk.content)


class WhisperASR:
    def __init__(self, model_name: str, max_seconds: int) -> None:
        from transformers import pipeline

        self.max_seconds = max_seconds
        self.asr = pipeline("automatic-speech-recognition", model=model_name, device=-1)

    def transcribe(self, audio: bytes, lang_hint: str) -> tuple[str, float]:
        from transformers.pipelines.audio_utils import ffmpeg_read

        try:
            # Decoded in memory via an ffmpeg pipe; audio is never written to disk.
            wav = ffmpeg_read(audio, SAMPLE_RATE)
        except Exception as e:  # ffmpeg raises ValueError on undecodable input
            raise AudioError("Could not decode audio.") from e
        duration = len(wav) / SAMPLE_RATE
        if duration < 0.3:
            raise AudioError("Recording is too short.")
        if duration > self.max_seconds:
            raise AudioError(f"Recording is longer than {self.max_seconds} seconds.")
        kwargs: dict[str, Any] = {"task": "transcribe"}
        if lang_hint in ("en", "hi"):
            kwargs["language"] = lang_hint
        out = self.asr({"raw": wav, "sampling_rate": SAMPLE_RATE}, generate_kwargs=kwargs)
        return str(out.get("text", "")).strip(), round(duration, 2)


class OpenAITTS:
    def __init__(self, api_key: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=60)

    def synthesize(self, text: str, lang: str) -> bytes:
        voice = "nova" if lang == "hi" else "alloy"
        resp = self.client.audio.speech.create(model="tts-1", voice=voice, input=text[:4000])
        return bytes(resp.content)
