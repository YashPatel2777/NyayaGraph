"""Log-scrubbing test (Phase 6, R-04): no case text, question text or transcript appears in
logs or Logfire spans.

A text document full of unique marker strings goes through real case ingestion (real Groq
extraction and linking), then a question containing markers goes through the real query
graph (real Groq, Ollama, FlashRank) over the API. Logging is set up by the production
configure_observability(); the only change is an extra in-memory span exporter, so the check
covers the configuration that runs in production. Logs are captured at INFO, the level
production sets. Positive controls make sure both captures are not empty.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx
import logfire
import pytest
from logfire.testing import TestExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from app.agents.graph import QueryAgent, build_graph
from app.agents.nodes.graph_expander import ConceptIndex
from app.agents.state import QueryDeps
from app.config import get_settings
from app.deps import get_jobs, get_neo4j, get_query_agent
from app.main import create_app
from app.observability import configure_observability
from app.services.llm.groq_client import GroqClient
from app.services.retrieval.ranking_service import RankingService
from tests.integration.test_case_processor import db, embeddings  # noqa: F401
from tests.integration.test_jobs import make_queue

pytestmark = [pytest.mark.neo4j, pytest.mark.ollama, pytest.mark.groq]

MARKERS = ["ZEBRAQUARTZ", "VELVETMANGO", "PLUMBRIDGE", "77,341", "QX-55813"]
DOCUMENT = (
    "Complaint notes of Anil ZEBRAQUARTZ, 12 Lotus Road, PLUMBRIDGE.\n\n"
    "On 3 March 2026 I bought a mixer grinder from VELVETMANGO Traders, invoice QX-55813, "
    "and paid Rs. 77,341. The motor burnt out on 9 March 2026. VELVETMANGO Traders refused to "
    "replace it or refund the price and said all sales are final."
)
QUESTION = "Can ZEBRAQUARTZ get a refund of Rs. 77,341 from VELVETMANGO Traders for invoice QX-55813?"


@pytest.fixture
def captured(monkeypatch):
    """Production logging with an extra in-memory span exporter; restored afterwards."""
    exporter = TestExporter()
    real_configure = logfire.configure

    def configure(**kwargs):
        kwargs.setdefault("additional_span_processors", []).append(SimpleSpanProcessor(exporter))
        return real_configure(**kwargs)

    monkeypatch.setattr(logfire, "configure", configure)
    settings = get_settings().model_copy(update={"LOGFIRE_TOKEN": None})
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    configure_observability(settings)
    try:
        yield exporter
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


async def test_no_case_or_question_text_in_logs_or_spans(db, embeddings, captured, caplog, tmp_path: Path) -> None:
    s = get_settings()
    llm = GroqClient(s.GROQ_API_KEY.get_secret_value(), s.GROQ_MODEL)
    queue = make_queue(db, embeddings, tmp_path / "uploads", llm=llm)
    await queue.start()
    deps = QueryDeps(neo4j=db, embeddings=embeddings, llm=llm, ranker=RankingService(s.FLASHRANK_MODEL),
                     concepts=ConceptIndex([]), rerank_high=s.RERANK_HIGH)
    app = create_app()
    app.dependency_overrides[get_neo4j] = lambda: db
    app.dependency_overrides[get_jobs] = lambda: queue
    app.dependency_overrides[get_settings] = lambda: s.model_copy(update={"UPLOAD_DIR": str(tmp_path / "uploads")})
    app.dependency_overrides[get_query_agent] = lambda: QueryAgent(build_graph(deps), deps)
    caplog.set_level(logging.INFO)  # the production level (app/observability.py)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            case_id = (await c.post("/v1/cases", json={"title": "scrubbing test"})).json()["case_id"]
            up = await c.post(f"/v1/cases/{case_id}/documents",
                              files={"file": ("notes.txt", DOCUMENT.encode("utf-8"), "text/plain")})
            assert up.status_code == 202, up.text
            await asyncio.wait_for(queue.join(), timeout=600)
            job = (await c.get(f"/v1/jobs/{up.json()['job_id']}")).json()
            assert job["status"] in ("succeeded", "succeeded_with_warnings"), job
            resp = await c.post("/v1/query", json={"question": QUESTION, "case_id": case_id}, timeout=120)
            assert resp.status_code == 200, resp.text
    finally:
        await queue.stop()

    # The markers really went through the pipeline: the answer may quote them.
    assert any(m in resp.text for m in MARKERS), "the answer used none of the case text"

    logs = caplog.text + "\n" + "\n".join(repr(r.args) for r in caplog.records)
    spans = json.dumps(captured.exported_spans_as_dict(), default=str)
    # Positive controls: both sinks captured the run.
    assert "case_doc_done" in logs and "query_done" in logs
    assert "query_done" in spans and "case_doc_done" in spans
    for marker in MARKERS:
        assert marker not in logs, f"{marker!r} found in logs"
        assert marker not in spans, f"{marker!r} found in spans"
