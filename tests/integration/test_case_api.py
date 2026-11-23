"""Cases, documents and jobs endpoints (3.4; 07 T3.11) over ASGI, with a real JobQueue on a
throwaway Neo4j database: E-09, E-13, R-05 and the error envelope (3.3)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from app.config import Settings, get_settings
from app.deps import get_jobs, get_neo4j
from app.ingestion.jobs import JobQueue
from app.main import create_app
from tests.integration.test_case_processor import FIXTURES, db, embeddings  # noqa: F401
from tests.integration.test_jobs import make_queue

pytestmark = [pytest.mark.neo4j, pytest.mark.ollama]


class Api:
    def __init__(self, client: httpx.AsyncClient, queue: JobQueue, uploads: Path) -> None:
        self.client, self.queue, self.uploads = client, queue, uploads

    async def case(self, title: str = "Zenith X2 refund") -> str:
        resp = await self.client.post("/v1/cases", json={"title": title})
        assert resp.status_code == 201, resp.text
        return resp.json()["case_id"]

    async def upload(self, case_id: str, path: Path, mime: str, **form) -> httpx.Response:
        return await self.client.post(
            f"/v1/cases/{case_id}/documents",
            files={"file": (path.name, path.read_bytes(), mime)}, data=form,
        )


@pytest.fixture
async def api(db, embeddings, tmp_path: Path) -> AsyncIterator[Api]:
    uploads = tmp_path / "uploads"
    queue = make_queue(db, embeddings, uploads)
    await queue.start()
    app = create_app()
    settings = get_settings().model_copy(update={"UPLOAD_DIR": str(uploads)})
    app.dependency_overrides[get_neo4j] = lambda: db
    app.dependency_overrides[get_jobs] = lambda: queue
    app.dependency_overrides[get_settings] = lambda: settings
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield Api(client, queue, uploads)
    await queue.stop()


def error(resp: httpx.Response) -> dict:
    return resp.json()["error"]


async def test_case_lifecycle_and_lists(api: Api) -> None:
    c = api.client
    case_id = await api.case()
    assert uuid.UUID(case_id).version == 4
    got = (await c.get(f"/v1/cases/{case_id}")).json()
    assert got["status"] == "active" and got["created_at"].endswith("Z")
    second = await api.case("Second")
    page = (await c.get("/v1/cases", params={"limit": 1})).json()
    assert len(page["items"]) == 1 and page["next_cursor"]
    rest = (await c.get("/v1/cases", params={"limit": 100, "cursor": page["next_cursor"]})).json()
    assert {i["case_id"] for i in page["items"] + rest["items"]} >= {case_id, second}
    assert rest["next_cursor"] is None

    missing = await c.get(f"/v1/cases/{uuid.uuid4()}")
    assert missing.status_code == 404 and error(missing)["code"] == "CASE_NOT_FOUND"
    bad = await c.post("/v1/cases", json={"title": ""})
    assert bad.status_code == 422 and error(bad)["code"] == "VALIDATION_ERROR"
    assert (await c.get("/v1/cases/not-a-uuid")).status_code == 422


async def test_upload_process_duplicate_and_job(api: Api) -> None:
    c = api.client
    case_id = await api.case()
    pdf = FIXTURES / "clean" / "invoice_and_letter.pdf"
    resp = await api.upload(case_id, pdf, "application/pdf")
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["processing_status"] == "queued" and body["duplicate"] is False
    await asyncio.wait_for(api.queue.join(), timeout=300)

    job = (await c.get(f"/v1/jobs/{body['job_id']}")).json()
    assert job["type"] == "case_document" and job["stage"] == "done" and job["progress"] == 100
    assert job["status"] in {"succeeded", "succeeded_with_warnings"} and job["error"] is None
    doc = (await c.get(f"/v1/cases/{case_id}/documents/{body['doc_id']}")).json()
    assert doc["media_type"] == "pdf" and doc["page_count"] == 2 and doc["mime"] == "application/pdf"
    assert doc["processing_status"] in {"ready", "ready_with_warnings"}
    assert "stored_path" not in doc and "user_id" not in doc  # A-6
    stored = api.uploads / "local" / case_id / body["doc_id"] / "original.pdf"
    assert stored.read_bytes() == pdf.read_bytes()  # B2 layout

    # E-09: the same bytes again: 200, the existing document, no job.
    dup = await api.upload(case_id, pdf, "application/pdf")
    assert dup.status_code == 200
    assert dup.json() == {"doc_id": body["doc_id"], "job_id": None,
                          "processing_status": doc["processing_status"], "duplicate": True}
    listing = (await c.get(f"/v1/cases/{case_id}/documents")).json()
    assert [d["doc_id"] for d in listing["items"]] == [body["doc_id"]]

    assert error(await c.get(f"/v1/jobs/{uuid.uuid4()}"))["code"] == "JOB_NOT_FOUND"
    assert error(await c.get(f"/v1/cases/{case_id}/documents/{uuid.uuid4()}"))["code"] == "DOCUMENT_NOT_FOUND"


async def test_upload_validation_writes_nothing(api: Api, tmp_path: Path) -> None:
    c = api.client
    case_id = await api.case()
    png = FIXTURES / "noisy" / "service_job_sheet.png"
    wrong = await api.upload(case_id, png, "application/pdf")  # signature disagrees
    assert wrong.status_code == 415 and "application/pdf" in error(wrong)["details"]["allowed"]
    docx = tmp_path / "a.docx"
    docx.write_bytes(b"PK\x03\x04" + b"\x00" * 300)
    other = await api.upload(case_id, docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert other.status_code == 415
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    assert error(await api.upload(case_id, empty, "text/plain"))["code"] == "VALIDATION_ERROR"
    lang = await api.upload(case_id, FIXTURES / "clean" / "chat_seller.txt", "text/plain", language="English")
    assert lang.status_code == 422
    nowhere = await api.upload(str(uuid.uuid4()), FIXTURES / "clean" / "chat_seller.txt", "text/plain")
    assert nowhere.status_code == 404 and error(nowhere)["code"] == "CASE_NOT_FOUND"
    # E-13: nothing was stored or created.
    assert (await c.get(f"/v1/cases/{case_id}/documents")).json()["items"] == []
    assert not (api.uploads / "local" / case_id).exists()


async def test_oversized_upload_is_413(api: Api, monkeypatch, tmp_path: Path) -> None:
    from app.ingestion.loaders import router

    small = router.ALLOWED["text/plain"].__class__(**{**router.ALLOWED["text/plain"].__dict__, "max_bytes": 10})
    monkeypatch.setitem(router.ALLOWED, "text/plain", small)
    case_id = await api.case()
    resp = await api.upload(case_id, FIXTURES / "clean" / "chat_seller.txt", "text/plain")
    assert resp.status_code == 413 and error(resp)["details"] == {"max_bytes": 10}


async def test_delete_case_removes_nodes_and_files(api: Api, db) -> None:
    c = api.client
    case_id = await api.case()
    resp = await api.upload(case_id, FIXTURES / "clean" / "chat_seller.txt", "text/plain")
    job_id = resp.json()["job_id"]
    await asyncio.wait_for(api.queue.join(), timeout=300)
    assert (api.uploads / "local" / case_id).exists()

    deleted = await c.delete(f"/v1/cases/{case_id}")
    assert deleted.status_code == 202 and deleted.json()["status"] == "deleting"
    # R-05: while deleting, uploads get 409 (until the purge finishes, then 404).
    during = await api.upload(case_id, FIXTURES / "clean" / "complaint_notes.txt", "text/plain")
    assert during.status_code in {409, 404}
    await asyncio.wait_for(api.queue.join(), timeout=120)

    left = await db.read("MATCH (n) WHERE n.case_id = $case_id RETURN count(n) AS n", {"case_id": case_id})
    assert left == [{"n": 0}]
    assert not (api.uploads / "local" / case_id).exists()
    assert (await c.get(f"/v1/cases/{case_id}")).status_code == 404
    assert (await c.get(f"/v1/jobs/{job_id}")).status_code == 404
    assert (await c.delete(f"/v1/cases/{case_id}")).status_code == 404


async def test_upload_to_a_deleting_case_is_409(api: Api, db) -> None:
    case_id = await api.case()
    await db.write("MATCH (c:Case {case_id: $case_id}) SET c.status = 'deleting'", {"case_id": case_id})
    resp = await api.upload(case_id, FIXTURES / "clean" / "chat_seller.txt", "text/plain")
    assert resp.status_code == 409 and error(resp)["code"] == "CASE_DELETING"
