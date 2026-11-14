"""Runtime configuration, read from environment variables only.

Secrets (OPENAI_API_KEY) live in the server environment and are never sent to
the client. Every limit that protects cost or abuse is configurable here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

API_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEXITALK_", env_file=".env", extra="ignore")

    # --- secrets (no LEXITALK_ prefix: standard OpenAI variable) ---
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    # --- environment ---
    env: str = "development"  # development | production
    # NoDecode: accept "https://a.app,https://b.app" instead of requiring JSON.
    allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]
    trust_proxy_headers: bool = False  # True behind HF Spaces / a reverse proxy

    # --- models ---
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    embedding_model: str = "nlpaueb/legal-bert-base-uncased"
    asr_model: str = "openai/whisper-small"
    tts_enabled: bool = False

    # --- data / governance ---
    corpus_manifest: Path = API_ROOT / "corpus" / "manifest.json"
    corpus_dir: Path = API_ROOT.parent / "data" / "pdf"
    vector_store_dir: Path = API_ROOT / ".vector_store"
    collection_name: str = "lexitalk_public_v1"
    top_k: int = 8
    max_context_chars: int = 6000
    snippet_chars: int = 600

    # --- abuse & cost controls ---
    max_message_chars: int = 2000
    max_history_turns: int = 6
    max_audio_bytes: int = 5 * 1024 * 1024
    max_audio_seconds: int = 60
    rate_limit_per_minute: int = 10
    daily_request_budget: int = 500  # global cap on LLM-backed requests per UTC day
    audit_salt: SecretStr = SecretStr("change-me-in-production")

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):
                import json

                return json.loads(s)
            return [o.strip() for o in s.split(",") if o.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
