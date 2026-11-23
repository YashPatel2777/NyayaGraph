"""Admin legal endpoints (3.4): validation, signature check, 202 with a job, acts listing."""

from __future__ import annotations

import json
import uuid
from datetime import date

import pytest
from fastapi.testclient import TestClient

import app.api.admin_legal as admin
import app.ingestion.jobs as jobs_module
from app.deps import get_jobs, get_neo4j
from app.main import create_app

META = {
    "act_id": "cpa-2019", "title": "The Consumer Protection Act, 2019",
    "short_title": "Consumer Protection Act, 2019", "year": 2019, "number": "35 of 2019",
    "jurisdiction": "IN", "as_on_date": "2026-09-01", "kind": "act", "expected_sections": 107,
}
PDF = b"%PDF-1.7\n%fake body\n"


@pytest.fixture
def ctx(monkeypatch, tmp_path):
    calls: dict = {"jobs": [], "runs": []}

    async def fake_create(neo4j, job_uuid, user_id):
        calls["jobs"].append((job_uuid, user_id))

    class FakeQueue:
        async def enqueue(self, job_uuid):
            calls["runs"].append(job_uuid)

    async def fake_list(neo4j, limit, cursor):
        calls["list"] = (limit, cursor)
        return [{"act_id": "cpa-2019", "kind": "act", "title": "T", "short_title": "S",
                 "year": 2019, "number": "35 of 2019", "jurisdiction": "IN",
                 "as_on_date": date(2026, 9, 1), "status": "in_force", "external": False,
                 "issued_under_act_id": None, "issued_under_section": None}], "act:cpa-2019"

    monkeypatch.setattr(admin, "create_legal_job", fake_create)
    monkeypatch.setattr(admin, "list_acts", fake_list)
    monkeypatch.setattr(jobs_module, "LEGAL_UPLOAD_DIR", tmp_path)
    queue = FakeQueue()
    application = create_app()
    application.dependency_overrides[get_neo4j] = lambda: object()
    application.dependency_overrides[get_jobs] = lambda: queue
    return TestClient(application), calls, tmp_path


def _post(client, metadata, body=PDF):
    return client.post(
        "/v1/admin/legal/ingest",
        files={"file": ("cpa.pdf", body, "application/pdf")},
        data={"metadata": metadata if isinstance(metadata, str) else json.dumps(metadata)},
    )


def test_valid_upload_returns_202_and_queues_the_job(ctx) -> None:
    client, calls, tmp = ctx
    resp = _post(client, META)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert uuid.UUID(job_id).version == 4
    assert calls["jobs"] == [(job_id, "local")]  # user id from deps (3.2)
    assert calls["runs"] == [job_id]  # on the job queue (1.2)
    assert (tmp / f"{job_id}.pdf").read_bytes() == PDF
    # Stored so an interrupted job can resume (E-11).
    assert json.loads((tmp / f"{job_id}.json").read_text(encoding="utf-8"))["act_id"] == "cpa-2019"


def test_invalid_json_metadata_is_422(ctx) -> None:
    client, calls, _ = ctx
    resp = _post(client, "{not json")
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "VALIDATION_ERROR" and err["details"]["fields"][0]["loc"] == ["body", "metadata"]
    assert calls["jobs"] == []  # nothing is written before validation passes (E-13)


def test_metadata_schema_error_names_the_field(ctx) -> None:
    client, _, _ = ctx
    resp = _post(client, {**META, "kind": "statute"})
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["fields"][0]["loc"] == ["body", "metadata", "kind"]


def test_missing_file_is_422(ctx) -> None:
    client, _, _ = ctx
    resp = client.post("/v1/admin/legal/ingest", data={"metadata": json.dumps(META)})
    assert resp.status_code == 422


def test_non_pdf_signature_is_415(ctx) -> None:
    client, calls, _ = ctx
    resp = _post(client, META, body=b"hello, not a pdf")
    assert resp.status_code == 415
    assert resp.json()["error"]["details"] == {"allowed": ["application/pdf"]}
    assert calls["jobs"] == []


def test_oversized_file_is_413(ctx, monkeypatch) -> None:
    client, _, _ = ctx
    monkeypatch.setattr(admin, "MAX_PDF_BYTES", 10)
    resp = _post(client, META)
    assert resp.status_code == 413
    assert resp.json()["error"]["details"] == {"max_bytes": 10}


def test_list_acts(ctx) -> None:
    client, calls, _ = ctx
    resp = client.get("/v1/admin/legal/acts", params={"limit": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["next_cursor"] == "act:cpa-2019"
    assert body["items"][0]["act_id"] == "cpa-2019" and body["items"][0]["as_on_date"] == "2026-09-01"
    assert calls["list"] == (1, None)
    assert client.get("/v1/admin/legal/acts", params={"limit": 101}).status_code == 422
