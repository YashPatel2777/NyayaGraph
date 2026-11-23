"""Query (3.4, 1.4 C).

POST /v1/query   QueryRequest   200 AnalysisResponse (3.5)

The router only validates and delegates: QueryAgent checks the case (C1), runs the graph
with the 45 s budget (E-12) and returns the verified analysis. A null thread_id is replaced
by a new UUID, which is returned (3.4).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request

from app.agents.graph import QueryAgent
from app.deps import get_current_user, get_query_agent
from app.errors import error_responses
from app.schemas.api import QueryRequest
from app.schemas.output import AnalysisResponse

router = APIRouter(prefix="/v1", tags=["query"], responses=error_responses('CASE_NOT_FOUND', 'CASE_DELETING', 'NO_READY_DOCUMENTS', 'LLM_UPSTREAM_ERROR', 'QUERY_TIMEOUT'))


@router.post("/query", response_model=AnalysisResponse)
async def query(
    body: QueryRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
    agent: QueryAgent = Depends(get_query_agent),
) -> AnalysisResponse:
    return await agent.answer(
        request_id=request.state.request_id,
        user_id=user_id,
        case_id=str(body.case_id) if body.case_id else None,
        question=body.question,
        thread_id=body.thread_id or str(uuid.uuid4()),
        mode=body.mode,
        max_case_chunks=body.options.max_case_chunks,
        max_provisions=body.options.max_provisions,
        include_trace=body.options.include_trace,
    )
