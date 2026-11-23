"""Error envelope (3.3) and X-Request-ID (3.1)."""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.errors import AppError, ErrorCode, install_exception_handlers
from app.middleware import RequestIDMiddleware


class Body(BaseModel):
    title: str = Field(min_length=1, max_length=5)


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    install_exception_handlers(app)

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    @app.get("/missing")
    async def missing():
        raise AppError(ErrorCode.CASE_NOT_FOUND)

    @app.get("/limited")
    async def limited():
        raise AppError(
            ErrorCode.RATE_LIMITED,
            details={"retry_after_seconds": 12},
            headers={"Retry-After": "12"},
        )

    @app.post("/body")
    async def body(payload: Body):
        return payload

    @app.get("/boom")
    async def boom():
        raise RuntimeError("secret case text must not leak")

    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_app(), raise_server_exceptions=False)


def test_codes_map_to_spec_statuses() -> None:
    expected = {
        "CASE_NOT_FOUND": 404, "DOCUMENT_NOT_FOUND": 404, "JOB_NOT_FOUND": 404,
        "PROVISION_NOT_FOUND": 404, "CASE_DELETING": 409, "NO_READY_DOCUMENTS": 409,
        "FILE_TOO_LARGE": 413, "UNSUPPORTED_MEDIA_TYPE": 415, "VALIDATION_ERROR": 422,
        "RATE_LIMITED": 429, "INTERNAL": 500, "LLM_UPSTREAM_ERROR": 502,
        "DEPENDENCY_UNAVAILABLE": 503, "QUERY_TIMEOUT": 504,
    }
    assert {c.value: c.http_status for c in ErrorCode} == expected


def test_app_error_envelope(client) -> None:
    resp = client.get("/missing", headers={"X-Request-ID": "req-123"})
    assert resp.status_code == 404
    assert resp.headers["X-Request-ID"] == "req-123"
    assert resp.json() == {
        "error": {
            "code": "CASE_NOT_FOUND",
            "message": "Case not found.",
            "details": {},
            "request_id": "req-123",
        }
    }


def test_app_error_details_and_headers(client) -> None:
    resp = client.get("/limited")
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "12"
    assert resp.json()["error"]["details"] == {"retry_after_seconds": 12}


def test_validation_error_lists_fields_without_echoing_input(client) -> None:
    resp = client.post("/body", json={"title": "far too long a title"})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "VALIDATION_ERROR"
    assert err["details"]["fields"][0]["loc"] == ["body", "title"]
    assert "msg" in err["details"]["fields"][0]
    assert "far too long" not in resp.text


def test_unhandled_exception_is_generic_500(client) -> None:
    resp = client.get("/boom", headers={"X-Request-ID": "req-500"})
    assert resp.status_code == 500
    assert resp.json() == {
        "error": {
            "code": "INTERNAL",
            "message": "Internal server error.",
            "details": {},
            "request_id": "req-500",
        }
    }
    assert "secret" not in resp.text
    assert resp.headers["X-Request-ID"] == "req-500"


def test_request_id_generated_when_absent(client) -> None:
    resp = client.get("/ok")
    generated = resp.headers["X-Request-ID"]
    assert uuid.UUID(generated).version == 4


@pytest.mark.parametrize("bad", ["x" * 65, "has space", "tab\tchar"])
def test_invalid_request_id_is_replaced(client, bad) -> None:
    resp = client.get("/missing", headers={"X-Request-ID": bad})
    rid = resp.headers["X-Request-ID"]
    assert rid != bad and uuid.UUID(rid).version == 4
    assert resp.json()["error"]["request_id"] == rid


def test_request_id_of_64_chars_is_kept(client) -> None:
    rid = "a" * 64
    assert client.get("/ok", headers={"X-Request-ID": rid}).headers["X-Request-ID"] == rid
