"""Admin legal endpoints (3.4). Open in v1 like every endpoint (3.2): the service binds to
127.0.0.1. Legal KB nodes are written only by the ingest job (R-18).

POST /v1/admin/legal/ingest   multipart: file (PDF) + metadata (JSON string)   202 {job_id}
GET  /v1/admin/legal/acts     limit, cursor                                    200 Page[ActResponse]

The ingest job runs on the job queue (1.2, app/ingestion/jobs.py). The PDF and its
metadata are stored first, so a job interrupted by a restart resumes (E-11).
"""

from __future__ import annotations

import json
import uuid

import anyio
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from pydantic import ValidationError

from app.deps import get_current_user, get_jobs, get_neo4j
from app.errors import AppError, ErrorCode, error_responses
from app.ingestion.jobs import JobQueue, write_legal_job_files
from app.ingestion.legal.legal_processor import create_legal_job, list_acts
from app.schemas.api import ActResponse, JobAccepted, LegalIngestMetadata, Page
from app.services.retrieval.neo4j_service import Neo4jService

router = APIRouter(prefix="/v1/admin/legal", tags=["admin"], responses=error_responses('FILE_TOO_LARGE', 'UNSUPPORTED_MEDIA_TYPE'))

MAX_PDF_BYTES = 50 * 1024 * 1024  # 3.4 PDF limit
PDF_SIGNATURE = b"%PDF-"


def _metadata(raw: str) -> LegalIngestMetadata:
    try:
        return LegalIngestMetadata.model_validate(json.loads(raw))
    except json.JSONDecodeError as exc:
        fields = [{"loc": ["body", "metadata"], "msg": f"invalid JSON: {exc.msg}"}]
    except ValidationError as exc:
        fields = [
            {"loc": ["body", "metadata", *err["loc"]], "msg": err["msg"]}
            for err in exc.errors(include_input=False, include_url=False)
        ]
    raise AppError(ErrorCode.VALIDATION_ERROR, details={"fields": fields})


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED, response_model=JobAccepted)
async def ingest(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    user_id: str = Depends(get_current_user),
    neo4j: Neo4jService = Depends(get_neo4j),
    jobs: JobQueue = Depends(get_jobs),
) -> JobAccepted:
    meta = _metadata(metadata)
    data = await file.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise AppError(ErrorCode.FILE_TOO_LARGE, details={"max_bytes": MAX_PDF_BYTES})
    if not data.startswith(PDF_SIGNATURE):  # 3.4: the signature must agree with the type
        raise AppError(ErrorCode.UNSUPPORTED_MEDIA_TYPE, details={"allowed": ["application/pdf"]})

    job_uuid = str(uuid.uuid4())
    await anyio.to_thread.run_sync(write_legal_job_files, job_uuid, data, meta)
    await create_legal_job(neo4j, job_uuid, user_id)  # A2: queued, then 202
    await jobs.enqueue(job_uuid)
    return JobAccepted(job_id=job_uuid)


@router.get("/acts", response_model=Page[ActResponse])
async def acts(
    limit: int = Query(default=25, ge=1, le=100),
    cursor: str | None = Query(default=None, min_length=1, max_length=512),
    neo4j: Neo4jService = Depends(get_neo4j),
) -> Page[ActResponse]:
    items, next_cursor = await list_acts(neo4j, limit, cursor)
    return Page[ActResponse](items=[ActResponse(**i) for i in items], next_cursor=next_cursor)
