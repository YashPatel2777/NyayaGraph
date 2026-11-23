"""Job queue edge cases through the real JobQueue and a throwaway Neo4j database
(07 T3.10, T3.12): E-07, E-08, E-10, E-11. The LLM is scripted; no GPU model is loaded.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pymupdf
import pytest

from app.ingestion.case.case_processor import CaseProcessor
from app.ingestion.jobs import JobQueue
from app.ingestion.legal.legal_processor import LegalProcessor
from app.schemas.extraction import ExtractionResult
from app.schemas.graph_schema import document_node_id, job_node_id
from app.services.media.transcription_service import TranscriptionService
from app.services.retrieval import graph_queries
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.neo4j_service import Neo4jService
from app.services.retrieval.scope import ScopedGraph
from tests.integration.test_case_processor import (  # noqa: F401 (fixtures)
    FIXTURES,
    NoGpuModels,
    ScriptedLLM,
    add_document,
    counts,
    db,
    embeddings,
    new_case,
)
from app.config import get_settings

pytestmark = [pytest.mark.neo4j, pytest.mark.ollama]


def make_queue(db: Neo4jService, emb: EmbeddingService, tmp: Path, llm=None, transcriber=None) -> JobQueue:
    processor = CaseProcessor(
        db, emb, NoGpuModels(), transcriber or NoGpuModels(), asyncio.Semaphore(1),
        llm or ScriptedLLM(), get_settings().LINK_MIN_SCORE,
    )
    return JobQueue(db, processor, LegalProcessor(db, emb), tmp)


@pytest.fixture
async def started() -> AsyncIterator[list[JobQueue]]:
    queues: list[JobQueue] = []
    yield queues
    for q in queues:
        await q.stop()


async def job_row(db: Neo4jService, job_uuid: str) -> dict:
    rows = await db.read(
        "MATCH (j:Job {id: $id}) RETURN j.status AS status, j.error_code AS code, "
        "j.attempts AS attempts, j.stage AS stage, j.progress AS progress, j.warnings AS warnings",
        {"id": job_node_id(job_uuid)},
    )
    return rows[0]


async def doc_row(graph: ScopedGraph, doc_uuid: str) -> dict:
    rows = await graph.read(graph_queries.DOC_GET, {"doc_node_id": document_node_id(doc_uuid)})
    return rows[0]


async def run(queue: JobQueue, started: list[JobQueue], *job_uuids: str) -> None:
    started.append(queue)
    await queue.start()
    for j in job_uuids:
        await queue.enqueue(j)
    await asyncio.wait_for(queue.join(), timeout=300)


async def test_success_records_stages_progress_and_status(db, embeddings, started, tmp_path: Path) -> None:
    graph = await new_case(db)
    job = await add_document(graph, tmp_path, FIXTURES / "clean" / "chat_seller.txt", "text/plain", "text")
    await run(make_queue(db, embeddings, tmp_path), started, job.job_uuid)
    row = await job_row(db, job.job_uuid)
    assert (row["status"], row["stage"], row["progress"]) == ("succeeded", "done", 100)
    assert (await doc_row(graph, job.doc_uuid))["processing_status"] == "ready"


async def test_e07_encrypted_pdf_fails_the_job(db, embeddings, started, tmp_path: Path) -> None:
    pdf = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Invoice QK-55120-8841 for one Zenith X2 smartphone.")
    doc.save(pdf, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    graph = await new_case(db)
    job = await add_document(graph, tmp_path / "up", pdf, "application/pdf", "pdf")
    await run(make_queue(db, embeddings, tmp_path), started, job.job_uuid)
    assert (await job_row(db, job.job_uuid))["status"] == "failed"
    assert (await job_row(db, job.job_uuid))["code"] == "PDF_ENCRYPTED"
    d = await doc_row(graph, job.doc_uuid)
    assert (d["processing_status"], d["error_code"]) == ("failed", "PDF_ENCRYPTED")


async def test_e07_image_only_pdf_with_blank_ocr_is_ocr_empty(db, embeddings, started, tmp_path: Path) -> None:
    class BlankOcr:
        def ocr_pdf_page(self, path, index):
            return ""

    pdf = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf)
    graph = await new_case(db)
    job = await add_document(graph, tmp_path / "up", pdf, "application/pdf", "pdf")
    queue = make_queue(db, embeddings, tmp_path)
    queue.case_processor.ocr = BlankOcr()
    await run(queue, started, job.job_uuid)
    assert (await job_row(db, job.job_uuid))["code"] == "OCR_EMPTY"


async def test_e08_undecodable_audio_fails_the_job(db, embeddings, started, tmp_path: Path) -> None:
    bad = tmp_path / "voice.ogg"
    bad.write_bytes(b"OggS\x00" + b"\x13" * 4000)
    graph = await new_case(db)
    job = await add_document(graph, tmp_path / "up", bad, "audio/ogg", "audio")
    # The real service decodes before it loads any model, so no GPU is used.
    queue = make_queue(db, embeddings, tmp_path, transcriber=TranscriptionService("medium"))
    await run(queue, started, job.job_uuid)
    assert (await job_row(db, job.job_uuid))["code"] == "AUDIO_DECODE_FAILED"
    assert (await doc_row(graph, job.doc_uuid))["error_code"] == "AUDIO_DECODE_FAILED"


async def test_e10_case_deleted_while_the_job_runs(db, embeddings, started, tmp_path: Path) -> None:
    graph = await new_case(db)
    job = await add_document(graph, tmp_path, FIXTURES / "clean" / "chat_seller.txt", "text/plain", "text")

    class DeleteDuringExtraction(ScriptedLLM):
        async def complete_json_resuming(self, messages, schema, **kwargs):
            if schema is ExtractionResult:  # chunks are written by now (B5)
                await graph.write("MATCH (c:Case {case_id: $case_id, user_id: $user_id}) SET c.status = 'deleting'")
            return await super().complete_json_resuming(messages, schema, **kwargs)

    await run(make_queue(db, embeddings, tmp_path, llm=DeleteDuringExtraction()), started, job.job_uuid)
    assert (await job_row(db, job.job_uuid))["status"] == "cancelled"
    left = await counts(graph)
    assert not {"CaseChunk", "Party", "Event", "Claim", "Evidence"} & set(left)  # partial nodes gone


async def test_e10_deleting_a_case_cancels_its_queued_jobs(db, embeddings, started, tmp_path: Path) -> None:
    graph = await new_case(db)
    job = await add_document(graph, tmp_path, FIXTURES / "clean" / "chat_seller.txt", "text/plain", "text")
    queue = make_queue(db, embeddings, tmp_path)
    started.append(queue)
    await queue.start()  # recover() enqueues the queued job, but delete_case runs first
    record = await queue.delete_case(graph.scope.user_id, graph.scope.case_id)
    assert record["status"] == "deleting"
    await asyncio.wait_for(queue.join(), timeout=120)
    assert await counts(graph) == {}  # R-05: nothing with this case_id is left, not even the Job
    assert not (tmp_path / graph.scope.user_id / graph.scope.case_id).exists()


async def test_e11_restart_requeues_running_jobs_and_fails_after_3_attempts(db, embeddings, started, tmp_path: Path) -> None:
    graph = await new_case(db)
    resumed = await add_document(graph, tmp_path, FIXTURES / "clean" / "chat_seller.txt", "text/plain", "text")
    exhausted = await add_document(graph, tmp_path, FIXTURES / "clean" / "complaint_notes.txt", "text/plain", "text")
    # Simulate a process killed mid-job: both Jobs are left 'running'.
    for job, attempts in ((resumed, 0), (exhausted, 3)):
        await db.write("MATCH (j:Job {id: $id}) SET j.status = 'running', j.attempts = $a",
                       {"id": job_node_id(job.job_uuid), "a": attempts})
    queue = make_queue(db, embeddings, tmp_path)
    started.append(queue)
    await queue.start()  # recovery runs here (E-11)
    await asyncio.wait_for(queue.join(), timeout=300)
    first = await job_row(db, resumed.job_uuid)
    assert (first["status"], first["attempts"]) == ("succeeded", 1)
    second = await job_row(db, exhausted.job_uuid)
    assert (second["status"], second["code"], second["attempts"]) == ("failed", "JOB_ATTEMPTS_EXCEEDED", 4)
    assert (await doc_row(graph, exhausted.doc_uuid))["error_code"] == "JOB_ATTEMPTS_EXCEEDED"
