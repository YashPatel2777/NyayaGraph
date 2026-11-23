"""POST /v1/query (3.4): request mapping, thread ids, validation, the error envelope, the
3.5 response with and without trace, and the C2 guard path."""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.agents import graph as graph_module
from app.agents.graph import QueryAgent, to_response
from app.deps import get_query_agent
from app.errors import AppError, ErrorCode
from app.guardrails.hook import GuardResult
from app.main import create_app
from app.schemas.output import DISCLAIMER

CASE = str(uuid.uuid4())

ANALYSIS = {
    "summary": "File within two years.",
    "relevant_facts": [],
    "applicable_provisions": [{
        "provision_id": "sec:cpa-2019:69", "act": "Consumer Protection Act, 2019", "act_as_on": date(2026, 9, 1),
        "citation": "Section 69, Consumer Protection Act, 2019", "status": "in_force", "relation": "direct",
        "relevance": "time limit", "excerpt": "within two years",
    }],
    "precedents": [],
    "analysis": "Two years.",
    "gaps_or_conflicts": [],
    "confidence": {"level": "medium", "reasons": ["1 direct provision", "0 supporting case facts"]},
    "disclaimer": DISCLAIMER,
}
STATE = {
    "intent": "LEGAL_ONLY", "search_query": "time limit", "issue_statement": "time limit",
    "analysis": ANALYSIS,
    "trace": {"downgraded": False, "notes": [], "counts": {"case_candidates": 0, "case_final": 0,
              "legal_candidates": 20, "legal_final": 8, "expanded_nodes": 30, "dropped_citations": 0},
              "timings_ms": {"planner": 400, "legal": 300}, "llm": {"prompt_tokens": 2800, "completion_tokens": 300}},
}


class StubAgent:
    def __init__(self, error: AppError | None = None) -> None:
        self.calls: list[dict] = []
        self.error = error

    async def answer(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return to_response(STATE, request_id=kwargs["request_id"], case_id=kwargs["case_id"],
                           thread_id=kwargs["thread_id"], include_trace=kwargs["include_trace"], total_ms=1234)


@pytest.fixture
def client_and_agent():
    agent = StubAgent()
    app = create_app()
    app.dependency_overrides[get_query_agent] = lambda: agent
    return TestClient(app), agent


def test_request_is_mapped_to_the_agent(client_and_agent) -> None:
    client, agent = client_and_agent
    resp = client.post("/v1/query", json={
        "question": "What is the time limit?", "case_id": CASE, "thread_id": "t-001", "mode": "case_and_legal",
        "options": {"max_case_chunks": 3, "max_provisions": 6, "include_trace": False},
        "user_id": "intruder",  # R-02: ignored
    })
    assert resp.status_code == 200
    call = agent.calls[0]
    assert call["user_id"] == "local" and call["case_id"] == CASE and call["thread_id"] == "t-001"
    assert (call["mode"], call["max_case_chunks"], call["max_provisions"]) == ("case_and_legal", 3, 6)
    body = resp.json()
    assert body["request_id"] == resp.headers["X-Request-ID"] == call["request_id"]
    assert body["thread_id"] == "t-001" and body["trace"] is None
    assert body["applicable_provisions"][0]["act_as_on"] == "2026-09-01"
    assert body["disclaimer"] == DISCLAIMER


def test_null_thread_id_is_generated_and_returned(client_and_agent) -> None:
    client, agent = client_and_agent
    body = client.post("/v1/query", json={"question": "Time limit?", "mode": "legal_only"}).json()
    assert str(uuid.UUID(body["thread_id"])) == body["thread_id"] == agent.calls[0]["thread_id"]
    assert body["case_id"] is None and agent.calls[0]["case_id"] is None


def test_trace_when_requested(client_and_agent) -> None:
    client, _ = client_and_agent
    body = client.post("/v1/query", json={"question": "Time limit?", "options": {"include_trace": True}}).json()
    trace = body["trace"]
    assert trace["search_query"] == "time limit" and trace["counts"]["legal_final"] == 8
    assert trace["timings_ms"] == {"planner": 400, "legal": 300, "total": 1234}
    assert trace["llm"] == {"prompt_tokens": 2800, "completion_tokens": 300} and "notes" not in trace


@pytest.mark.parametrize("payload", [
    {"question": "hi"},  # under 3 characters
    {"question": "x" * 2001},
    {"question": "Time limit?", "mode": "case_only"},  # case_id required
    {"question": "Time limit?", "case_id": "not-a-uuid"},
    {"question": "Time limit?", "options": {"max_provisions": 13}},
    {"question": "Time limit?", "thread_id": "x" * 65},
])
def test_invalid_requests_are_422(client_and_agent, payload) -> None:
    client, agent = client_and_agent
    resp = client.post("/v1/query", json=payload)
    assert resp.status_code == 422 and resp.json()["error"]["code"] == "VALIDATION_ERROR"
    assert agent.calls == []


@pytest.mark.parametrize("code", [ErrorCode.CASE_NOT_FOUND, ErrorCode.CASE_DELETING, ErrorCode.NO_READY_DOCUMENTS,
                                  ErrorCode.QUERY_TIMEOUT, ErrorCode.LLM_UPSTREAM_ERROR])
def test_agent_errors_use_the_envelope(code: ErrorCode) -> None:
    app = create_app()
    app.dependency_overrides[get_query_agent] = lambda: StubAgent(AppError(code))
    resp = TestClient(app).post("/v1/query", json={"question": "Time limit?", "case_id": CASE})
    assert resp.status_code == code.http_status and resp.json()["error"]["code"] == code.value


async def test_guard_block_reply_skips_the_graph(monkeypatch) -> None:
    monkeypatch.setattr(graph_module, "guard", lambda text, user: GuardResult(blocked=True, reply="Not allowed."))

    class NoGraph:
        async def ainvoke(self, *a, **k):
            raise AssertionError("the graph must not run")

    agent = QueryAgent(NoGraph(), deps=None, enable_guardrails=True)  # type: ignore[arg-type]
    resp = await agent.answer(request_id="r", user_id="local", case_id=None, question="q?", thread_id="t",
                              mode="auto", max_case_chunks=5, max_provisions=8, include_trace=False)
    assert resp.summary == "Not allowed." and resp.applicable_provisions == [] and resp.disclaimer == DISCLAIMER


async def test_guard_is_not_called_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(graph_module, "guard", lambda *a: pytest.fail("guard must not run"))

    async def fake_run_query(*a, **k):
        return STATE

    monkeypatch.setattr(graph_module, "run_query", fake_run_query)
    agent = QueryAgent(object(), deps=None)  # type: ignore[arg-type]
    resp = await agent.answer(request_id="r", user_id="local", case_id=None, question="q?", thread_id="t",
                              mode="auto", max_case_chunks=5, max_provisions=8, include_trace=True)
    assert resp.intent == "LEGAL_ONLY" and resp.trace.timings_ms["total"] >= 0
