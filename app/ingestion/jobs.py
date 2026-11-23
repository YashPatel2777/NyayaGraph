"""Job queue and worker (1.2, E-10, E-11; 07 T3.10).

- One asyncio.Queue and one worker task in the Uvicorn process (1.2). Jobs run one at a
  time; the GPU semaphore of 1 (held by CaseProcessor for OCR and Whisper) guards GPU stages.
- The queue holds job ids only. The Job node in Neo4j is the record, so nothing is lost on
  a restart: recover() sets running jobs back to queued with one more attempt, fails those
  past 3 attempts with JOB_ATTEMPTS_EXCEEDED, and re-enqueues every queued job in creation
  order (E-11). Uploads stay on disk and all writes MERGE, so resuming is safe.
- case_document jobs run CaseProcessor (B3 to B10). A case that is deleting or gone stops
  the job at the next check, which is cancelled and its partial nodes removed (E-10).
- Deleting a case (R-05): delete_case() marks it deleting and cancels its queued jobs; a
  background task waits for a running job of that case to stop at its next E-10 check,
  then removes every node with the case_id and the case's upload folder. start() resumes
  deletions a restart interrupted.
- legal_ingest jobs run the Phase 2 legal processor. The admin endpoint stores the PDF and
  its metadata as data/legal/uploads/{job_uuid}.pdf and .json (A-16), so a restarted job
  can resume.
- R-04: logs carry job ids, statuses, codes and timings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio

from app.errors import AppError
from app.errors import ErrorCode
from app.ingestion.case.case_processor import (
    CaseGone,
    CaseJob,
    CaseProcessor,
    DocumentFailed,
    dependency_error_code,
)
from app.ingestion.legal.legal_processor import MAX_JOB_WARNINGS, LegalProcessor, run_legal_job
from app.schemas.api import LegalIngestMetadata
from app.schemas.graph_schema import JOB_PREFIX, bare_uuid, document_node_id, job_node_id
from app.services.retrieval import graph_queries
from app.services.retrieval.neo4j_service import Neo4jService
from app.services.retrieval.scope import CaseScope, ScopedGraph

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3  # E-11
DELETE_BATCH = 1000
# R-05: how long a deletion waits for the case's running job to stop at an E-10 check.
DELETE_WAIT_S = 300.0
DELETE_POLL_S = 0.2
# A-16: uploaded statutes and their metadata (06: data/legal/).
LEGAL_UPLOAD_DIR = Path("data/legal/uploads")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _native(value: object) -> object:
    """neo4j.time values to Python datetimes."""
    to_native = getattr(value, "to_native", None)
    return to_native() if callable(to_native) else value


def legal_job_files(job_uuid: str) -> tuple[Path, Path]:
    return LEGAL_UPLOAD_DIR / f"{job_uuid}.pdf", LEGAL_UPLOAD_DIR / f"{job_uuid}.json"


class JobQueue:
    def __init__(
        self,
        neo4j: Neo4jService,
        case_processor: CaseProcessor,
        legal_processor: LegalProcessor,
        upload_dir: Path,
    ) -> None:
        self.neo4j = neo4j
        self.case_processor = case_processor
        self.legal_processor = legal_processor
        self.upload_dir = upload_dir
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._purges: set[asyncio.Task[None]] = set()
        self.current: str | None = None  # job uuid being run
        self.current_case: str | None = None  # its case uuid

    # Lifecycle

    async def start(self) -> None:
        await self.recover()
        for row in await self.neo4j.read(graph_queries.CASES_DELETING):
            self._schedule_purge(row["case_id"], row["user_id"])
        self._worker = asyncio.create_task(self._run(), name="job-worker")

    async def stop(self) -> None:
        """Stop the worker. A job cut off here is still 'running' and resumes through E-11;
        an unfinished case deletion resumes in start()."""
        for task in list(self._purges):
            task.cancel()
        await asyncio.gather(*self._purges, return_exceptions=True)
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def enqueue(self, job_uuid: str) -> None:
        await self._queue.put(job_uuid)

    async def join(self) -> None:
        """Wait until every enqueued job and case deletion has finished (tests, scripts)."""
        await self._queue.join()
        if self._purges:
            await asyncio.gather(*list(self._purges), return_exceptions=True)

    # Case deletion (R-05)

    async def delete_case(self, user_id: str, case_uuid: str) -> dict:
        """Mark the case deleting, cancel its queued jobs and start the removal. Returns the
        case record (A-5); 404 if the case does not exist."""
        graph = ScopedGraph(self.neo4j, CaseScope(case_uuid, user_id))
        rows = await graph.write(graph_queries.CASE_MARK_DELETING)
        if not rows:
            raise AppError(ErrorCode.CASE_NOT_FOUND)
        await graph.write(graph_queries.CASE_JOBS_CANCEL, {"now": _now()})
        self._schedule_purge(graph.scope.case_id, user_id)
        return {k: _native(v) for k, v in rows[0].items()}

    def _schedule_purge(self, case_uuid: str, user_id: str) -> None:
        task = asyncio.create_task(self._purge_case(case_uuid, user_id), name="case-purge")
        self._purges.add(task)
        task.add_done_callback(self._purges.discard)

    async def _purge_case(self, case_uuid: str, user_id: str) -> None:
        start = time.perf_counter()
        waited = 0.0
        while self.current_case == case_uuid and waited < DELETE_WAIT_S:
            await asyncio.sleep(DELETE_POLL_S)  # E-10: the job stops at its next check
            waited += DELETE_POLL_S
        graph = ScopedGraph(self.neo4j, CaseScope(case_uuid, user_id))
        total = 0
        try:
            while True:
                rows = await graph.write(graph_queries.CASE_DELETE_BATCH, {"batch": DELETE_BATCH})
                deleted = rows[0]["deleted"] if rows else 0
                total += deleted
                if deleted == 0:
                    break
            folder = self.upload_dir / user_id / case_uuid
            await anyio.to_thread.run_sync(lambda: shutil.rmtree(folder, ignore_errors=True))
        except AppError:
            # Neo4j unavailable (E-04): the case stays 'deleting' and start() resumes it.
            logger.warning("case_purge_failed case_id=%s deleted=%d", case_uuid, total)
            return
        logger.info("case_purged case_id=%s nodes=%d ms=%d", case_uuid, total,
                    (time.perf_counter() - start) * 1000)

    async def recover(self) -> list[dict]:
        """E-11. Returns the rows of the jobs that were running."""
        rows = await self.neo4j.write(
            graph_queries.JOBS_RECOVER, {"max_attempts": MAX_ATTEMPTS, "now": _now()}
        )
        for row in rows:
            logger.warning(
                "job_recovered job_id=%s status=%s attempts=%s", row["id"], row["status"], row["attempts"]
            )
            if row["status"] == "failed" and row["type"] == "case_document":
                await self._doc_status(row, "failed", "JOB_ATTEMPTS_EXCEEDED")
        for row in await self.neo4j.read(graph_queries.JOBS_QUEUED):
            await self.enqueue(bare_uuid(row["id"], JOB_PREFIX))
        return rows

    # Worker

    async def _run(self) -> None:
        while True:
            job_uuid = await self._queue.get()
            self.current = job_uuid
            try:
                await self.run_job(job_uuid)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let one job stop the worker
                logger.exception("job_worker_error job_id=%s type=%s", job_uuid, type(exc).__name__)
            finally:
                self.current = None
                self._queue.task_done()

    async def _update(self, job_uuid: str, **props: Any) -> None:
        await self.neo4j.write(graph_queries.UPDATE_JOB, {"id": job_node_id(job_uuid), "props": props})

    async def _doc_status(self, job: dict, status: str, code: str | None) -> None:
        graph = ScopedGraph(self.neo4j, CaseScope(job["case_id"], job["user_id"]))
        await graph.write(
            graph_queries.DOC_UPDATE,
            {"doc_node_id": document_node_id(job["doc_id"]),
             "props": {"processing_status": status, "error_code": code}},
        )

    async def run_job(self, job_uuid: str) -> None:
        rows = await self.neo4j.read(graph_queries.JOB_FOR_WORKER, {"id": job_node_id(job_uuid)})
        if not rows or rows[0]["status"] != "queued":
            return  # cancelled (R-05) or already finished
        job = rows[0]
        if job["type"] == "legal_ingest":
            await self._run_legal(job_uuid)
        else:
            await self._run_case(job_uuid, job)

    async def _run_legal(self, job_uuid: str) -> None:
        pdf, meta_file = legal_job_files(job_uuid)
        try:
            raw = await anyio.to_thread.run_sync(meta_file.read_text, "utf-8")
            meta = LegalIngestMetadata.model_validate_json(raw)
        except (OSError, ValueError):
            await self._update(job_uuid, status="failed", finished_at=_now(),
                               error_code="INTERNAL", error_message="The job's stored upload is missing.")
            return
        await run_legal_job(self.legal_processor, job_uuid, str(pdf), meta)

    async def _run_case(self, job_uuid: str, job: dict) -> None:
        case_job = CaseJob(job_uuid, job["case_id"], job["doc_id"], job["user_id"])
        self.current_case = job["case_id"]
        try:
            await self._run_case_job(job_uuid, job, case_job)
        finally:
            self.current_case = None

    async def _run_case_job(self, job_uuid: str, job: dict, case_job: CaseJob) -> None:
        start = time.perf_counter()
        await self._update(job_uuid, status="running", started_at=_now(),
                           stage="text_extraction", progress=0, error_code=None, error_message=None)

        async def progress(stage: str, pct: int) -> None:
            await self._update(job_uuid, stage=stage, progress=pct)

        try:
            result = await self.case_processor.process(case_job, progress)
        except CaseGone:
            # E-10: stop, cancel, remove the document's partial nodes.
            await self._update(job_uuid, status="cancelled", finished_at=_now())
            try:
                await self.case_processor.purge_document(case_job)
            except AppError:
                logger.warning("job_purge_failed job_id=%s", job_uuid)
            logger.info("job_cancelled job_id=%s", job_uuid)
            return
        except DocumentFailed as exc:
            await self._update(job_uuid, status="failed", finished_at=_now(),
                               error_code=exc.code, error_message=exc.message[:2000])
            logger.warning("job_failed job_id=%s code=%s", job_uuid, exc.code)
            return
        except AppError as exc:  # raised before the document could be marked
            code = dependency_error_code(exc)
            await self._update(job_uuid, status="failed", finished_at=_now(),
                               error_code=code, error_message=exc.message)
            return
        except Exception as exc:
            logger.exception("job_error job_id=%s type=%s", job_uuid, type(exc).__name__)
            await self._update(job_uuid, status="failed", finished_at=_now(),
                               error_code="INTERNAL", error_message="Internal error.")
            try:
                await self._doc_status(job, "failed", "INTERNAL")
            except AppError:
                pass
            return
        status = "succeeded_with_warnings" if result.warnings else "succeeded"
        await self._update(job_uuid, status=status, finished_at=_now(), stage="done", progress=100,
                           warnings=result.warnings[:MAX_JOB_WARNINGS])
        logger.info("job_done job_id=%s status=%s chunks=%d ms=%d", job_uuid, status,
                    result.chunk_count, (time.perf_counter() - start) * 1000)


def write_legal_job_files(job_uuid: str, pdf: bytes, meta: LegalIngestMetadata) -> None:
    """Blocking: the admin endpoint calls it through a worker thread."""
    pdf_path, meta_path = legal_job_files(job_uuid)
    LEGAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf)
    meta_path.write_text(json.dumps(meta.model_dump(mode="json")), encoding="utf-8")


async def get_job(neo4j: Neo4jService, user_id: str, job_uuid: str) -> dict:
    """GET /v1/jobs/{job_id} (3.4): the caller's own job, else 404 JOB_NOT_FOUND."""
    rows = await neo4j.read(
        graph_queries.JOB_FOR_USER, {"id": job_node_id(job_uuid), "user_id": user_id}
    )
    if not rows:
        raise AppError(ErrorCode.JOB_NOT_FOUND)
    row = {k: _native(v) for k, v in rows[0].items()}
    code, message = row.pop("error_code"), row.pop("error_message")
    row["job_id"] = bare_uuid(row.pop("id"), JOB_PREFIX)
    row["error"] = {"code": code, "message": message or code} if code else None
    row["warnings"] = row["warnings"] or []
    row["progress"] = row["progress"] or 0
    return row
