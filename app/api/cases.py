"""Cases (3.4).

POST   /v1/cases             {title}         201 CaseResponse
GET    /v1/cases             limit, cursor   200 Page[CaseResponse]
GET    /v1/cases/{case_id}                   200 CaseResponse
DELETE /v1/cases/{case_id}                   202 CaseResponse with status 'deleting' (A-5, R-05)
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from app.deps import get_current_user, get_jobs, get_neo4j
from app.errors import error_responses
from app.ingestion.case.case_processor import create_case, get_case, list_cases
from app.ingestion.jobs import JobQueue
from app.schemas.api import CaseCreateRequest, CaseResponse, Page
from app.services.retrieval.neo4j_service import Neo4jService

router = APIRouter(prefix="/v1/cases", tags=["cases"], responses=error_responses('CASE_NOT_FOUND', 'CASE_DELETING'))


@router.post("", status_code=status.HTTP_201_CREATED, response_model=CaseResponse)
async def create(
    body: CaseCreateRequest,
    user_id: str = Depends(get_current_user),
    neo4j: Neo4jService = Depends(get_neo4j),
) -> CaseResponse:
    return CaseResponse(**await create_case(neo4j, user_id, body.title))


@router.get("", response_model=Page[CaseResponse])
async def list_(
    limit: int = Query(default=25, ge=1, le=100),
    cursor: UUID | None = Query(default=None),
    user_id: str = Depends(get_current_user),
    neo4j: Neo4jService = Depends(get_neo4j),
) -> Page[CaseResponse]:
    items, next_cursor = await list_cases(neo4j, user_id, limit, str(cursor) if cursor else None)
    return Page[CaseResponse](items=[CaseResponse(**i) for i in items], next_cursor=next_cursor)


@router.get("/{case_id}", response_model=CaseResponse)
async def get(
    case_id: UUID,
    user_id: str = Depends(get_current_user),
    neo4j: Neo4jService = Depends(get_neo4j),
) -> CaseResponse:
    return CaseResponse(**await get_case(neo4j, user_id, str(case_id)))


@router.delete("/{case_id}", status_code=status.HTTP_202_ACCEPTED, response_model=CaseResponse)
async def delete(
    case_id: UUID,
    user_id: str = Depends(get_current_user),
    jobs: JobQueue = Depends(get_jobs),
) -> CaseResponse:
    return CaseResponse(**await jobs.delete_case(user_id, str(case_id)))
