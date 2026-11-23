"""Case documents (3.4, B1, B2, E-09, E-13).

POST /v1/cases/{case_id}/documents            multipart: file, language   202 (200 duplicate)
GET  /v1/cases/{case_id}/documents            limit, cursor               200 Page[DocumentResponse]
GET  /v1/cases/{case_id}/documents/{doc_id}                               200 DocumentResponse
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.deps import get_current_user, get_jobs, get_neo4j
from app.errors import error_responses
from app.ingestion.case.case_processor import get_document, list_documents, store_document
from app.ingestion.jobs import JobQueue
from app.ingestion.loaders.router import upload_type
from app.schemas.api import DocumentResponse, DocumentUploadResponse, Page
from app.services.retrieval.neo4j_service import Neo4jService

router = APIRouter(prefix="/v1/cases/{case_id}/documents", tags=["documents"], responses=error_responses('CASE_NOT_FOUND', 'DOCUMENT_NOT_FOUND', 'CASE_DELETING', 'FILE_TOO_LARGE', 'UNSUPPORTED_MEDIA_TYPE'))


@router.post(
    "",
    status_code=202,
    response_model=DocumentUploadResponse,
    responses={200: {"model": DocumentUploadResponse, "description": "Duplicate (E-09)"}},
)
async def upload(
    case_id: UUID,
    file: UploadFile = File(...),
    language: str | None = Form(default=None, pattern=r"^[a-z]{2}$"),  # ISO 639-1 hint
    user_id: str = Depends(get_current_user),
    neo4j: Neo4jService = Depends(get_neo4j),
    jobs: JobQueue = Depends(get_jobs),
    settings: Settings = Depends(get_settings),
):
    kind = upload_type(file.content_type)  # 415 before reading the body
    data = await file.read(kind.max_bytes + 1)  # one byte over is enough to answer 413
    accepted = await store_document(
        neo4j, user_id, str(case_id), file.filename, kind, data, language, Path(settings.UPLOAD_DIR)
    )
    body = DocumentUploadResponse(
        doc_id=accepted.doc_id, job_id=accepted.job_id,
        processing_status=accepted.processing_status, duplicate=accepted.duplicate,
    )
    if accepted.duplicate:
        return JSONResponse(status_code=200, content=body.model_dump(mode="json"))
    await jobs.enqueue(accepted.job_id)
    return body


@router.get("", response_model=Page[DocumentResponse])
async def list_(
    case_id: UUID,
    limit: int = Query(default=25, ge=1, le=100),
    cursor: UUID | None = Query(default=None),
    user_id: str = Depends(get_current_user),
    neo4j: Neo4jService = Depends(get_neo4j),
) -> Page[DocumentResponse]:
    items, next_cursor = await list_documents(
        neo4j, user_id, str(case_id), limit, str(cursor) if cursor else None
    )
    return Page[DocumentResponse](items=[DocumentResponse(**i) for i in items], next_cursor=next_cursor)


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get(
    case_id: UUID,
    doc_id: UUID,
    user_id: str = Depends(get_current_user),
    neo4j: Neo4jService = Depends(get_neo4j),
) -> DocumentResponse:
    return DocumentResponse(**await get_document(neo4j, user_id, str(case_id), str(doc_id)))
