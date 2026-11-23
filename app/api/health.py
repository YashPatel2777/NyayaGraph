"""Health (3.4).

GET /health   200 HealthResponse with status 'ok' when Neo4j answers, all four search indexes
              are ONLINE (with the SC-15 filter properties), the Meta check passes and Ollama
              serves EMBED_MODEL; otherwise 503 with the same body shape and status 'fail'
              (A-4). The checks themselves live in app/main.py (dependency_status).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.deps import get_dependency_check
from app.errors import error_responses
from app.schemas.api import HealthChecks, HealthResponse

router = APIRouter(tags=["health"], responses=error_responses())


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
)
async def health(
    check: Callable[[], Awaitable[dict[str, str]]] = Depends(get_dependency_check),
) -> HealthResponse | JSONResponse:
    checks = HealthChecks(**await check())
    body = HealthResponse(
        status="ok" if all(v == "ok" for v in checks.model_dump().values()) else "fail",
        checks=checks,
    )
    if body.status == "ok":
        return body
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump())
