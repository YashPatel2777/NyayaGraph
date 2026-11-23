"""Soak test (Phase 6): 20 concurrent queries and one concurrent ingestion against a real
server process, under Groq rate limiting.

Opt-in (`pytest -m soak`, A-46). Exit criterion (05 Phase 6 as changed by SC-14 c2): no 500
responses, and every response is 200, 502 LLM_UPSTREAM_ERROR or 504 QUERY_TIMEOUT with the
error envelope; the count of 200s is recorded. On the Groq free tier most queries are
expected to time out (8,000 tokens per minute, O-6).

The server uses the configured database, because the queries need the Legal KB and the
synthetic evaluation cases (case questions q01 to q20 of tests/eval/questions.jsonl). The
only write is the uploaded document, which goes into a temporary soak case deleted at the
end. The server log is also checked for the question texts (R-04).
"""

from __future__ import annotations

import asyncio
import statistics
import time
from pathlib import Path

import httpx
import pytest

from app.config import get_settings
from app.services.retrieval.neo4j_service import Neo4jService
from tests.eval.run_eval import case_ids, load_questions
from tests.integration.test_case_processor import FIXTURES
from tests.integration.test_recovery import TERMINAL, ServerProcess

pytestmark = [pytest.mark.soak, pytest.mark.neo4j, pytest.mark.ollama, pytest.mark.groq]

CONCURRENT_QUERIES = 20
ALLOWED = {200: None, 502: "LLM_UPSTREAM_ERROR", 504: "QUERY_TIMEOUT"}


async def _query(client: httpx.AsyncClient, body: dict) -> tuple[int, dict, float]:
    start = time.perf_counter()
    resp = await client.post("/v1/query", json=body)
    return resp.status_code, resp.json(), time.perf_counter() - start


async def _ingest(client: httpx.AsyncClient) -> tuple[str, str, dict]:
    case_id = (await client.post("/v1/cases", json={"title": "soak test (temporary)"})).json()["case_id"]
    path = FIXTURES / "clean" / "complaint_notes.txt"
    up = await client.post(f"/v1/cases/{case_id}/documents",
                           files={"file": (path.name, path.read_bytes(), "text/plain")})
    return case_id, up.status_code, up.json()


def test_20_concurrent_queries_and_one_ingestion(tmp_path: Path) -> None:
    s = get_settings()
    neo4j = Neo4jService(s.NEO4J_URI, s.NEO4J_USER, s.NEO4J_PASSWORD.get_secret_value(), s.NEO4J_DATABASE)
    try:
        cases = case_ids(neo4j, s.DEFAULT_USER_ID)
    finally:
        neo4j.close()
    questions = [q for q in load_questions() if q["case"] in cases][:CONCURRENT_QUERIES]
    assert len(questions) == CONCURRENT_QUERIES, "the synthetic evaluation cases are not loaded"

    server = ServerProcess(tmp_path / "server.log")
    server.start()
    soak_case = None
    try:
        async def run() -> tuple[list, tuple]:
            async with httpx.AsyncClient(base_url=server.base, timeout=120) as client:
                bodies = [{"question": q["question"], "case_id": cases[q["case"]]} for q in questions]
                return await asyncio.gather(asyncio.gather(*(_query(client, b) for b in bodies)), _ingest(client))

        started = time.perf_counter()
        results, (soak_case, up_status, up_body) = asyncio.run(run())
        wall = time.perf_counter() - started

        statuses = [status for status, _, _ in results]
        for status, body, _ in results:
            assert status in ALLOWED, f"status {status} is not allowed: {body}"
            if status != 200:
                err = body["error"]
                assert err["code"] == ALLOWED[status] and set(err) == {"code", "message", "details", "request_id"}
        assert 500 not in statuses
        assert up_status == 202, up_body

        # The concurrent ingestion still finishes (E-01: the job waits out rate limits).
        job_id, deadline = up_body["job_id"], time.monotonic() + 1200
        while (job := httpx.get(f"{server.base}/v1/jobs/{job_id}", timeout=10).json())["status"] not in TERMINAL:
            assert time.monotonic() < deadline, f"ingestion did not finish: {job}"
            time.sleep(5)

        ok = [lat for st, _, lat in results if st == 200]
        print(f"\nsoak: {len(results)} queries in {wall:.0f} s wall time; "
              f"200: {statuses.count(200)}, 502: {statuses.count(502)}, 504: {statuses.count(504)}, 500: 0")
        if ok:
            print(f"soak: latency of the 200s: min {min(ok):.1f} s, median {statistics.median(ok):.1f} s, max {max(ok):.1f} s")
        print(f"soak: concurrent ingestion job {job['status']}")

        log = (tmp_path / "server.log").read_text(encoding="utf-8", errors="replace")
        leaked = [q["id"] for q in questions if q["question"] in log]
        assert not leaked, f"question text found in the server log (R-04): {leaked}"
    finally:
        if soak_case:
            httpx.delete(f"{server.base}/v1/cases/{soak_case}", timeout=30)
            deadline = time.monotonic() + 120
            while httpx.get(f"{server.base}/v1/cases/{soak_case}", timeout=10).status_code != 404:
                if time.monotonic() > deadline:
                    break
                time.sleep(2)
        server.stop()
