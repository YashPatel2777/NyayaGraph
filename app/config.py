"""Configuration: every environment variable in 1.3, loaded with pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )

    APP_ENV: Literal["dev", "prod"] = "dev"
    # Keep loopback while the service has no authentication (3.2).
    APP_HOST: str = "127.0.0.1"
    # The single user id for every request in v1 (3.2, R-02).
    DEFAULT_USER_ID: str = Field(default="local", min_length=1)

    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: SecretStr
    NEO4J_DATABASE: str = "neo4j"

    OLLAMA_URL: str = "http://localhost:11434"
    # Locked values, checked against the Meta node at startup (2.2, R-08).
    EMBED_MODEL: str = "nomic-embed-text"
    EMBED_DIM: int = 768

    GROQ_API_KEY: SecretStr
    # D-1 (docs/README.md): the operator replaced llama-3.3-70b-versatile, which Groq no
    # longer serves, with openai/gpt-oss-120b.
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # Empty means the FlashRank library default.
    FLASHRANK_MODEL: str = ""
    WHISPER_MODEL: str = "medium"
    OCR_LANGS: Annotated[list[str], NoDecode] = ["en", "hi"]
    UPLOAD_DIR: str = "./data/uploads"

    ENABLE_GUARDRAILS: bool = False
    # Initial values; calibrate on the evaluation set (R-24).
    LINK_MIN_SCORE: float = Field(default=0.60, ge=0.0, le=1.0)
    RERANK_HIGH: float = 0.5

    # Comma-separated; empty means CORS is off (1.3).
    CORS_ORIGINS: Annotated[list[str], NoDecode] = []
    LOGFIRE_TOKEN: SecretStr | None = None

    @field_validator("OCR_LANGS", "CORS_ORIGINS", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("LOGFIRE_TOKEN", mode="before")
    @classmethod
    def _empty_token_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def _no_wildcard_cors_in_prod(self) -> Settings:
        # 1.3: no wildcard origin when APP_ENV is prod.
        if self.APP_ENV == "prod" and "*" in self.CORS_ORIGINS:
            raise ValueError("CORS_ORIGINS may not contain '*' when APP_ENV is prod")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # required values come from the environment
