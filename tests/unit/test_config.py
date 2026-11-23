"""Settings (1.3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

REQUIRED = {"NEO4J_PASSWORD": "pw", "GROQ_API_KEY": "key"}
ALL_VARS = [
    "APP_ENV", "APP_HOST", "DEFAULT_USER_ID", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
    "NEO4J_DATABASE", "OLLAMA_URL", "EMBED_MODEL", "EMBED_DIM", "GROQ_API_KEY", "GROQ_MODEL",
    "FLASHRANK_MODEL", "WHISPER_MODEL", "OCR_LANGS", "UPLOAD_DIR", "ENABLE_GUARDRAILS",
    "LINK_MIN_SCORE", "RERANK_HIGH", "CORS_ORIGINS", "LOGFIRE_TOKEN",
]


@pytest.fixture
def clean_env(monkeypatch):
    for name in ALL_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _settings(**env) -> Settings:
    return Settings(_env_file=None, **{**REQUIRED, **env})  # type: ignore[call-arg]


def test_every_spec_variable_exists() -> None:
    assert set(ALL_VARS) <= set(Settings.model_fields)


def test_defaults(clean_env) -> None:
    s = _settings()
    assert s.APP_ENV == "dev"
    assert s.APP_HOST == "127.0.0.1"
    assert s.DEFAULT_USER_ID == "local"
    assert s.NEO4J_URI == "bolt://localhost:7687"
    assert s.NEO4J_USER == "neo4j" and s.NEO4J_DATABASE == "neo4j"
    assert s.OLLAMA_URL == "http://localhost:11434"
    assert s.EMBED_MODEL == "nomic-embed-text" and s.EMBED_DIM == 768
    assert s.WHISPER_MODEL == "medium" and s.OCR_LANGS == ["en", "hi"]
    assert s.UPLOAD_DIR == "./data/uploads"
    assert s.ENABLE_GUARDRAILS is False
    assert s.LINK_MIN_SCORE == 0.60 and s.RERANK_HIGH == 0.5
    assert s.CORS_ORIGINS == [] and s.LOGFIRE_TOKEN is None
    assert s.FLASHRANK_MODEL == ""


@pytest.mark.parametrize("missing", ["NEO4J_PASSWORD", "GROQ_API_KEY"])
def test_required(clean_env, missing) -> None:
    env = {k: v for k, v in REQUIRED.items() if k != missing}
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **env)  # type: ignore[call-arg]


def test_csv_parsing_from_environment(clean_env) -> None:
    clean_env.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    clean_env.setenv("OCR_LANGS", "en")
    clean_env.setenv("LOGFIRE_TOKEN", "")
    s = _settings()
    assert s.CORS_ORIGINS == ["http://a.test", "http://b.test"]
    assert s.OCR_LANGS == ["en"]
    assert s.LOGFIRE_TOKEN is None


def test_no_wildcard_cors_in_prod(clean_env) -> None:
    with pytest.raises(ValidationError, match="CORS_ORIGINS"):
        _settings(APP_ENV="prod", CORS_ORIGINS="*")
    assert _settings(APP_ENV="dev", CORS_ORIGINS="*").CORS_ORIGINS == ["*"]


def test_secrets_not_in_repr(clean_env) -> None:
    s = _settings()
    assert "pw" not in repr(s) and "key" not in repr(s.GROQ_API_KEY)


def test_docs_disabled_in_prod(clean_env) -> None:
    from app.main import create_app

    assert create_app(_settings(APP_ENV="prod")).openapi_url is None
    assert create_app(_settings(APP_ENV="dev")).openapi_url == "/openapi.json"
