"""AppError, error codes, the error envelope and the exception handlers (3.3).

Every non-2xx response produced by these handlers has the body
{"error": {"code", "message", "details", "request_id"}}. A raw exception message is never
returned; unhandled exceptions become a generic 500 INTERNAL and the stack goes to logs only.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Codes from the 3.3 table, each with its HTTP status and default message."""

    CASE_NOT_FOUND = ("CASE_NOT_FOUND", 404, "Case not found.")
    DOCUMENT_NOT_FOUND = ("DOCUMENT_NOT_FOUND", 404, "Document not found.")
    JOB_NOT_FOUND = ("JOB_NOT_FOUND", 404, "Job not found.")
    PROVISION_NOT_FOUND = ("PROVISION_NOT_FOUND", 404, "Provision not found.")
    CASE_DELETING = ("CASE_DELETING", 409, "Case is being deleted.")
    NO_READY_DOCUMENTS = ("NO_READY_DOCUMENTS", 409, "The case has no ready document.")
    FILE_TOO_LARGE = ("FILE_TOO_LARGE", 413, "File is too large.")
    UNSUPPORTED_MEDIA_TYPE = ("UNSUPPORTED_MEDIA_TYPE", 415, "Unsupported media type.")
    VALIDATION_ERROR = ("VALIDATION_ERROR", 422, "Request validation failed.")
    RATE_LIMITED = ("RATE_LIMITED", 429, "Rate limit exceeded.")
    INTERNAL = ("INTERNAL", 500, "Internal server error.")
    LLM_UPSTREAM_ERROR = ("LLM_UPSTREAM_ERROR", 502, "The language model service failed.")
    DEPENDENCY_UNAVAILABLE = ("DEPENDENCY_UNAVAILABLE", 503, "A required service is unavailable.")
    QUERY_TIMEOUT = ("QUERY_TIMEOUT", 504, "The query took too long.")

    def __new__(cls, code: str, http_status: int, default_message: str) -> ErrorCode:
        obj = str.__new__(cls, code)
        obj._value_ = code
        obj.http_status = http_status  # type: ignore[attr-defined]
        obj.default_message = default_message  # type: ignore[attr-defined]
        return obj

    http_status: int
    default_message: str


class AppError(Exception):
    """The only exception type that reaches the API with a non-500 status (06 section 4)."""

    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.code = code
        self.message = message or code.default_message
        self.details = details or {}
        self.headers = headers or {}
        super().__init__(f"{code.value}: {self.message}")

    @property
    def http_status(self) -> int:
        return self.code.http_status


def _request_id(request: Request) -> str | None:
    # Set by RequestIDMiddleware (3.1).
    return getattr(request.state, "request_id", None)


def error_body(
    code: ErrorCode, message: str, details: dict[str, Any], request_id: str | None
) -> dict[str, Any]:
    return {
        "error": {
            "code": code.value,
            "message": message,
            "details": details,
            "request_id": request_id,
        }
    }


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    logger.info(
        "app_error code=%s status=%s request_id=%s",
        exc.code.value,
        exc.http_status,
        _request_id(request),
    )
    return JSONResponse(
        status_code=exc.http_status,
        content=error_body(exc.code, exc.message, exc.details, _request_id(request)),
        headers=exc.headers,
    )


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # 3.3: details.fields is a list of {loc, msg}. The rejected input value is not echoed,
    # because it may be case or question text (R-04).
    fields = [
        {"loc": list(err.get("loc", ())), "msg": str(err.get("msg", ""))} for err in exc.errors()
    ]
    code = ErrorCode.VALIDATION_ERROR
    return JSONResponse(
        status_code=code.http_status,
        content=error_body(code, code.default_message, {"fields": fields}, _request_id(request)),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # The stack goes to logs only; the response message is generic (3.3).
    logger.exception(
        "unhandled_error type=%s request_id=%s", type(exc).__name__, _request_id(request)
    )
    code = ErrorCode.INTERNAL
    request_id = _request_id(request)
    # Starlette sends this response from its outermost middleware, outside
    # RequestIDMiddleware, so the request id header is echoed here (3.1).
    return JSONResponse(
        status_code=code.http_status,
        content=error_body(code, code.default_message, {}, request_id),
        headers={"X-Request-ID": request_id} if request_id else None,
    )


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)


# OpenAPI (Phase 5 review): the 3.3 envelope, documented on every router.


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = {}
    request_id: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


# Every route can return these: invalid input, rate limit (3.2), unhandled error, and Neo4j or
# Ollama unreachable after retries (E-03, E-04).
COMMON_ERRORS = ("VALIDATION_ERROR", "RATE_LIMITED", "INTERNAL", "DEPENDENCY_UNAVAILABLE")


def error_responses(*codes: str) -> dict[int | str, dict[str, Any]]:
    """OpenAPI `responses` for the given 3.3 codes plus COMMON_ERRORS, grouped by status."""
    by_status: dict[int, list[str]] = {}
    for name in dict.fromkeys([*codes, *COMMON_ERRORS]):
        code = ErrorCode[name]
        by_status.setdefault(code.http_status, []).append(code.value)
    return {
        status: {"model": ErrorEnvelope, "description": ", ".join(names)}
        for status, names in sorted(by_status.items())
    }
