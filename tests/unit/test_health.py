"""GET /health (3.4, A-4) and the dependency checks behind it (app/main.py)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main
from app.deps import get_dependency_check
from app.errors import AppError, ErrorCode
from app.services.retrieval import graph_queries

ALL_OK = {"neo4j": "ok", "indexes": "ok", "embedding_meta": "ok", "ollama": "ok"}
PROPS = ["embedding", "case_id", "user_id"]


def _client(checks: dict) -> TestClient:
    app = main.create_app()

    async def check():
        return checks

    app.dependency_overrides[get_dependency_check] = lambda: check
    return TestClient(app)


def test_all_ok_is_200() -> None:
    resp = _client(ALL_OK).get("/health")
    assert resp.status_code == 200 and resp.json() == {"status": "ok", "checks": ALL_OK}


@pytest.mark.parametrize("failing", list(ALL_OK))
def test_any_failure_is_503_with_the_same_shape(failing: str) -> None:
    checks = {**ALL_OK, failing: "fail"}
    resp = _client(checks).get("/health")
    assert resp.status_code == 503 and resp.json() == {"status": "fail", "checks": checks}
    assert resp.headers["X-Request-ID"]


class FakeNeo4j:
    def __init__(self, reachable=True, offline=(), props=None, meta_ok=True) -> None:
        self.reachable, self.offline, self.props, self.meta_ok = reachable, offline, props, meta_ok

    async def verify_connectivity(self):
        if not self.reachable:
            raise AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, details={"dependency": "neo4j"})

    async def read(self, query, params=None):
        assert query == graph_queries.INDEX_STATES
        return [{"name": n, "state": "POPULATING" if n in self.offline else "ONLINE",
                 "properties": self.props if self.props is not None else PROPS} for n in params["names"]]


class FakeEmbeddings:
    def __init__(self, served=True, meta_ok=True) -> None:
        self.served, self.meta_ok = served, meta_ok

    async def check_model_served(self):
        if not self.served:
            raise AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, details={"dependency": "ollama"})

    async def verify_meta(self, neo4j):
        if not self.meta_ok:
            raise RuntimeError("meta mismatch")


@pytest.mark.parametrize(("neo4j", "embeddings", "expected"), [
    (FakeNeo4j(), FakeEmbeddings(), ALL_OK),
    (FakeNeo4j(reachable=False), FakeEmbeddings(),
     {"neo4j": "fail", "indexes": "fail", "embedding_meta": "fail", "ollama": "ok"}),
    (FakeNeo4j(offline=("case_text_ft",)), FakeEmbeddings(), {**ALL_OK, "indexes": "fail"}),
    (FakeNeo4j(props=["embedding"]), FakeEmbeddings(), {**ALL_OK, "indexes": "fail"}),  # SC-15
    (FakeNeo4j(), FakeEmbeddings(meta_ok=False), {**ALL_OK, "embedding_meta": "fail"}),
    (FakeNeo4j(), FakeEmbeddings(served=False), {**ALL_OK, "ollama": "fail"}),
], ids=["ok", "neo4j-down", "index-offline", "index-props", "meta", "ollama"])
async def test_dependency_status(neo4j, embeddings, expected) -> None:
    assert await main.dependency_status(neo4j, embeddings) == expected  # type: ignore[arg-type]


def test_serve_binds_app_host(monkeypatch) -> None:
    # Phase 5 exit: the service listens on APP_HOST (default 127.0.0.1).
    import uvicorn

    seen: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    main.serve(main.get_settings())
    assert seen == {"app": "app.main:app", "host": "127.0.0.1", "port": 8000}
    main.serve(main.get_settings().model_copy(update={"APP_HOST": "127.0.0.2"}))
    assert seen["host"] == "127.0.0.2"
