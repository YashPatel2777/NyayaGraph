"""Contract tests: every error code in 3.3 over ASGI (Phase 5 exit).

Each test triggers one code through the real app (create_app) and checks the HTTP status,
the code, the envelope shape {error: {code, message, details, request_id}}, the 3.3 details
keys, and that request_id matches the X-Request-ID header (3.1). A throwaway Neo4j database
backs the store; only the LLM and the failures that cannot be produced for real (an unhandled
exception, an upstream LLM failure) are stubbed. test_every_code_has_a_contract_test fails
when a code is added to 3.3 without a test here.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from app.agents import graph as graph_module
from app.agents.graph import QueryAgent, build_graph
from app.agents.nodes.graph_expander import ConceptIndex
from app.agents.state import QueryDeps
from app.config import get_settings
from app.deps import get_concepts, get_jobs, get_neo4j, get_query_agent
from app.errors import AppError, ErrorCode
from app.main import create_app
from app.services.retrieval.neo4j_service import Neo4jService
from tests.integration.test_case_processor import FIXTURES, db, embeddings  # noqa: F401
from tests.integration.test_jobs import make_queue

pytestmark = [pytest.mark.neo4j, pytest.mark.ollama]

MISSING = str(uuid.uuid4())


class FailingLLM:
    """Raises what the Groq client raises after its retries (1.2, E-01)."""

    def __init__(self, error: AppError) -> None:
        self.error = error
        self.calls = 0

    async def complete_json(self, *args, **kwargs):
        self.calls += 1
        raise self.error


class Api:
    def __init__(self, app, client: httpx.AsyncClient, db: Neo4jService, embeddings) -> None:
        self.app, self.client, self.db, self.embeddings = app, client, db, embeddings

    async def case(self) -> str:
        resp = await self.client.post("/v1/cases", json={"title": "contract test"})
        assert resp.status_code == 201, resp.text
        return resp.json()["case_id"]

    def agent(self, llm, **kw) -> QueryAgent:
        deps = QueryDeps(neo4j=self.db, embeddings=self.embeddings, llm=llm, ranker=None,
                         concepts=ConceptIndex([]), rerank_high=0.5)
        return QueryAgent(build_graph(deps), deps, **kw)


@pytest.fixture
async def api(db, embeddings, tmp_path: Path) -> AsyncIterator[Api]:
    queue = make_queue(db, embeddings, tmp_path / "uploads")
    await queue.start()
    app = create_app()
    settings = get_settings().model_copy(update={"UPLOAD_DIR": str(tmp_path / "uploads")})
    app.dependency_overrides[get_neo4j] = lambda: db
    app.dependency_overrides[get_jobs] = lambda: queue
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_concepts] = lambda: ConceptIndex([])
    # raise_app_exceptions=False: a 500 is answered by the app, not re-raised into the test.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield Api(app, client, db, embeddings)
    await queue.stop()


def envelope(resp: httpx.Response, code: ErrorCode, details_keys: set[str] | None = None) -> dict:
    assert resp.status_code == code.http_status, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    err = body["error"]
    assert set(err) == {"code", "message", "details", "request_id"}
    assert err["code"] == code.value and isinstance(err["message"], str) and err["message"]
    assert isinstance(err["details"], dict)
    if details_keys is not None:
        assert set(err["details"]) == details_keys
    assert err["request_id"] and err["request_id"] == resp.headers["X-Request-ID"]
    return err


# 404


async def test_case_not_found(api: Api) -> None:
    envelope(await api.client.get(f"/v1/cases/{MISSING}"), ErrorCode.CASE_NOT_FOUND, set())


async def test_document_not_found(api: Api) -> None:
    case_id = await api.case()
    resp = await api.client.get(f"/v1/cases/{case_id}/documents/{MISSING}")
    envelope(resp, ErrorCode.DOCUMENT_NOT_FOUND, set())


async def test_job_not_found(api: Api) -> None:
    envelope(await api.client.get(f"/v1/jobs/{MISSING}"), ErrorCode.JOB_NOT_FOUND, set())


async def test_provision_not_found(api: Api) -> None:
    for pid in ("sec:no-such-act:1", "cl:cpa-2019:2:47-p1"):  # unknown, and a split part (A-45)
        envelope(await api.client.get(f"/v1/legal/provisions/{pid}"), ErrorCode.PROVISION_NOT_FOUND, set())


# 409


async def test_case_deleting(api: Api) -> None:
    case_id = await api.case()
    await api.db.write("MATCH (c:Case {case_id: $case_id}) SET c.status = 'deleting'", {"case_id": case_id})
    path = FIXTURES / "clean" / "chat_seller.txt"
    resp = await api.client.post(f"/v1/cases/{case_id}/documents",
                                 files={"file": (path.name, path.read_bytes(), "text/plain")})
    envelope(resp, ErrorCode.CASE_DELETING, set())
    # The query path checks the same rule before any model call (C1).
    llm = FailingLLM(AppError(ErrorCode.INTERNAL))
    api.app.dependency_overrides[get_query_agent] = lambda: api.agent(llm)
    resp = await api.client.post("/v1/query", json={"question": "Can I get a refund?", "case_id": case_id})
    envelope(resp, ErrorCode.CASE_DELETING, set())
    assert llm.calls == 0


async def test_no_ready_documents(api: Api) -> None:
    case_id = await api.case()
    llm = FailingLLM(AppError(ErrorCode.INTERNAL))
    api.app.dependency_overrides[get_query_agent] = lambda: api.agent(llm)
    resp = await api.client.post("/v1/query", json={"question": "When did I buy it?", "case_id": case_id,
                                                    "mode": "case_only"})
    envelope(resp, ErrorCode.NO_READY_DOCUMENTS, set())
    assert llm.calls == 0  # E-06: refused before the planner call


# 413, 415, 422


async def test_file_too_large(api: Api, monkeypatch) -> None:
    from app.ingestion.loaders import router

    small = router.ALLOWED["text/plain"].__class__(**{**router.ALLOWED["text/plain"].__dict__, "max_bytes": 10})
    monkeypatch.setitem(router.ALLOWED, "text/plain", small)
    case_id = await api.case()
    path = FIXTURES / "clean" / "chat_seller.txt"
    resp = await api.client.post(f"/v1/cases/{case_id}/documents",
                                 files={"file": (path.name, path.read_bytes(), "text/plain")})
    err = envelope(resp, ErrorCode.FILE_TOO_LARGE, {"max_bytes"})
    assert err["details"]["max_bytes"] == 10


async def test_unsupported_media_type(api: Api) -> None:
    case_id = await api.case()
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # declared as text, signature says PNG (E-13)
    resp = await api.client.post(f"/v1/cases/{case_id}/documents",
                                 files={"file": ("notes.txt", png, "text/plain")})
    err = envelope(resp, ErrorCode.UNSUPPORTED_MEDIA_TYPE, {"allowed"})
    assert "application/pdf" in err["details"]["allowed"]
    resp = await api.client.post(f"/v1/cases/{case_id}/documents",
                                 files={"file": ("a.docx", b"PK\x03\x04", "application/msword")})
    envelope(resp, ErrorCode.UNSUPPORTED_MEDIA_TYPE, {"allowed"})


async def test_validation_error(api: Api) -> None:
    resp = await api.client.post("/v1/cases", json={"title": ""})
    err = envelope(resp, ErrorCode.VALIDATION_ERROR, {"fields"})
    assert err["details"]["fields"][0]["loc"] == ["body", "title"]
    # The test app skips the lifespan, so the query agent dependency is supplied here.
    api.app.dependency_overrides[get_query_agent] = lambda: api.agent(FailingLLM(AppError(ErrorCode.INTERNAL)))
    resp = await api.client.post("/v1/query", json={"question": "hi"})  # under 3 characters
    envelope(resp, ErrorCode.VALIDATION_ERROR, {"fields"})


# 429


async def test_rate_limited(api: Api) -> None:
    for i in range(120):  # 3.2: everything else is 120 per minute per IP
        assert (await api.client.get(f"/v1/jobs/{MISSING}")).status_code == 404, i
    resp = await api.client.get(f"/v1/jobs/{MISSING}")
    err = envelope(resp, ErrorCode.RATE_LIMITED, {"retry_after_seconds"})
    assert int(resp.headers["Retry-After"]) == err["details"]["retry_after_seconds"] >= 1


# 500


async def test_internal(api: Api) -> None:
    class Broken:
        async def answer(self, **kwargs):
            raise RuntimeError("secret case text must not leak")

    api.app.dependency_overrides[get_query_agent] = lambda: Broken()
    resp = await api.client.post("/v1/query", json={"question": "Can I get a refund?"})
    err = envelope(resp, ErrorCode.INTERNAL, set())
    assert err["message"] == ErrorCode.INTERNAL.default_message and "secret" not in resp.text


# 502, 503, 504


async def test_llm_upstream_error(api: Api) -> None:
    llm = FailingLLM(AppError(ErrorCode.LLM_UPSTREAM_ERROR, details={"reason": "retries_exhausted"}))
    api.app.dependency_overrides[get_query_agent] = lambda: api.agent(llm)
    resp = await api.client.post("/v1/query", json={"question": "What is the time limit?"})
    err = envelope(resp, ErrorCode.LLM_UPSTREAM_ERROR, {"reason"})
    assert err["details"]["reason"] == "retries_exhausted" and llm.calls == 1


async def test_dependency_unavailable(api: Api) -> None:
    s = get_settings()
    down = Neo4jService("bolt://127.0.0.1:1", s.NEO4J_USER, s.NEO4J_PASSWORD.get_secret_value(), s.NEO4J_DATABASE)
    api.app.dependency_overrides[get_neo4j] = lambda: down
    try:
        resp = await api.client.get(f"/v1/cases/{MISSING}")  # E-04: after the retries
    finally:
        down.close()
    err = envelope(resp, ErrorCode.DEPENDENCY_UNAVAILABLE, {"dependency"})
    assert err["details"]["dependency"] == "neo4j"


async def test_query_timeout(api: Api, monkeypatch) -> None:
    import asyncio

    class SlowLLM:
        async def complete_json(self, *args, **kwargs):
            await asyncio.sleep(5)

    monkeypatch.setitem(graph_module.run_query.__kwdefaults__, "timeout_s", 0.1)  # E-12 budget
    api.app.dependency_overrides[get_query_agent] = lambda: api.agent(SlowLLM())
    resp = await api.client.post("/v1/query", json={"question": "What is the time limit?"})
    envelope(resp, ErrorCode.QUERY_TIMEOUT, set())


# Coverage of the 3.3 table


TESTED = {
    ErrorCode.CASE_NOT_FOUND: test_case_not_found,
    ErrorCode.DOCUMENT_NOT_FOUND: test_document_not_found,
    ErrorCode.JOB_NOT_FOUND: test_job_not_found,
    ErrorCode.PROVISION_NOT_FOUND: test_provision_not_found,
    ErrorCode.CASE_DELETING: test_case_deleting,
    ErrorCode.NO_READY_DOCUMENTS: test_no_ready_documents,
    ErrorCode.FILE_TOO_LARGE: test_file_too_large,
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: test_unsupported_media_type,
    ErrorCode.VALIDATION_ERROR: test_validation_error,
    ErrorCode.RATE_LIMITED: test_rate_limited,
    ErrorCode.INTERNAL: test_internal,
    ErrorCode.LLM_UPSTREAM_ERROR: test_llm_upstream_error,
    ErrorCode.DEPENDENCY_UNAVAILABLE: test_dependency_unavailable,
    ErrorCode.QUERY_TIMEOUT: test_query_timeout,
}


def test_every_code_has_a_contract_test() -> None:
    assert set(TESTED) == set(ErrorCode), sorted(set(ErrorCode) - set(TESTED))
    module = sys.modules[__name__]
    assert all(getattr(module, fn.__name__) is fn for fn in TESTED.values())
