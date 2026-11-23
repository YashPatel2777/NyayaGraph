"""Jobs (3.4).

GET /v1/jobs/{job_id}   200 JobResponse
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends

from app.deps import get_current_user, get_neo4j
from app.errors import error_responses
from app.ingestion.jobs import get_job
from app.schemas.api import JobResponse
from app.services.retrieval.neo4j_service import Neo4jService

router = APIRouter(prefix="/v1/jobs", tags=["jobs"], responses=error_responses('JOB_NOT_FOUND'))


@router.get("/{job_id}", response_model=JobResponse)
async def get(
    job_id: UUID,
    user_id: str = Depends(get_current_user),
    neo4j: Neo4jService = Depends(get_neo4j),
) -> JobResponse:
    return JobResponse(**await get_job(neo4j, user_id, str(job_id)))
