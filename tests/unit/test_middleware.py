"""Per-IP sliding-window rate limit (3.2, A-44) and its 429 envelope (3.3)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import RateLimitMiddleware, RequestIDMiddleware, rate_category


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _client(clock: Clock) -> TestClient:
    app = FastAPI()

    @app.post("/v1/query")
    async def query():
        return {"ok": True}

    @app.post("/v1/cases/{case_id}/documents")
    async def upload(case_id: str):
        return {"ok": True}

    @app.get("/v1/cases")
    async def cases():
        return {"ok": True}

    app.add_middleware(RateLimitMiddleware, clock=clock)
    app.add_middleware(RequestIDMiddleware)
    return TestClient(app)


@pytest.mark.parametrize(("method", "path", "expected"), [
    ("POST", "/v1/query", ("query", 30)),
    ("POST", "/v1/query/", ("query", 30)),
    ("POST", "/v1/cases/abc/documents", ("upload", 10)),
    ("GET", "/v1/cases/abc/documents", ("default", 120)),
    ("GET", "/v1/query", ("default", 120)),
    ("GET", "/health", ("default", 120)),
    ("POST", "/v1/admin/legal/ingest", ("default", 120)),
])
def test_categories(method, path, expected) -> None:
    assert rate_category(method, path) == expected


def test_31st_query_in_a_minute_is_429_with_envelope_and_retry_after() -> None:
    clock = Clock()
    client = _client(clock)
    for i in range(30):
        clock.now += 1.0
        assert client.post("/v1/query").status_code == 200, i
    resp = client.post("/v1/query", headers={"X-Request-ID": "req-429"})
    assert resp.status_code == 429
    body = resp.json()["error"]
    assert body["code"] == "RATE_LIMITED" and body["request_id"] == "req-429"
    # The oldest counted request was at t=1001; now is t=1030, so it leaves the window in 31 s.
    assert body["details"] == {"retry_after_seconds": 31}
    assert resp.headers["Retry-After"] == "31" and resp.headers["X-Request-ID"] == "req-429"


def test_window_slides_and_rejected_requests_are_not_counted() -> None:
    clock = Clock()
    client = _client(clock)
    for _ in range(10):
        assert client.post("/v1/cases/c1/documents").status_code == 200
    assert client.post("/v1/cases/c1/documents").status_code == 429
    assert client.post("/v1/cases/c1/documents").status_code == 429
    clock.now += 60.0  # every counted upload is now older than the window
    assert client.post("/v1/cases/c1/documents").status_code == 200


def test_categories_have_separate_buckets() -> None:
    clock = Clock()
    client = _client(clock)
    for _ in range(10):
        client.post("/v1/cases/c1/documents")
    assert client.post("/v1/cases/c1/documents").status_code == 429
    assert client.post("/v1/query").status_code == 200  # a different bucket
    assert client.get("/v1/cases").status_code == 200


def test_default_limit_is_120() -> None:
    clock = Clock()
    client = _client(clock)
    for _ in range(120):
        assert client.get("/v1/cases").status_code == 200
    assert client.get("/v1/cases").status_code == 429


def test_clients_are_keyed_by_ip() -> None:
    clock = Clock()
    middleware = RateLimitMiddleware(app=None, clock=clock)  # type: ignore[arg-type]
    for _ in range(10):
        assert middleware._retry_after(("10.0.0.1", "upload"), 10) == 0
    assert middleware._retry_after(("10.0.0.1", "upload"), 10) > 0
    assert middleware._retry_after(("10.0.0.2", "upload"), 10) == 0
