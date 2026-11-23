"""HTTP middleware: X-Request-ID (3.1) and the per-IP sliding-window rate limit (3.2, A-44)."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import deque
from collections.abc import Callable

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import ErrorCode, error_body

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_MAX_LEN = 64


def _accept_request_id(raw: str | None) -> str | None:
    # 3.1: 1 to 64 characters. Printable ASCII only, so the value is safe to echo in a header.
    if raw is None or not 1 <= len(raw) <= REQUEST_ID_MAX_LEN:
        return None
    if not all(33 <= ord(ch) <= 126 for ch in raw):
        return None
    return raw


class RequestIDMiddleware:
    """Echo a valid X-Request-ID, or generate a UUID4 when it is absent or invalid (3.1).

    The id is stored in request.state.request_id for the error envelope (3.3).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = None
        for name, value in scope.get("headers", []):
            if name.lower() == b"x-request-id":
                incoming = value.decode("latin-1")
                break
        request_id = _accept_request_id(incoming) or str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
            await send(message)

        await self.app(scope, receive, send_with_id)


# 3.2: requests per minute per client IP.
RATE_WINDOW_S = 60.0
QUERY_LIMIT = 30
UPLOAD_LIMIT = 10
DEFAULT_LIMIT = 120
_UPLOAD_PATH = re.compile(r"^/v1/cases/[^/]+/documents/?$")


def rate_category(method: str, path: str) -> tuple[str, int]:
    """(bucket name, limit) for a request (3.2)."""
    if method == "POST" and path.rstrip("/") == "/v1/query":
        return "query", QUERY_LIMIT
    if method == "POST" and _UPLOAD_PATH.match(path):
        return "upload", UPLOAD_LIMIT
    return "default", DEFAULT_LIMIT


class RateLimitMiddleware:
    """Sliding window per client IP and category, in memory (3.2, A-44).

    Each (ip, category) keeps the timestamps of its requests in the last RATE_WINDOW_S
    seconds. A request that would exceed the limit gets 429 RATE_LIMITED with Retry-After
    and details.retry_after_seconds (3.3): the whole seconds until the oldest counted request
    leaves the window. Rejected requests are not counted. Logs carry no request content (R-04).
    """

    def __init__(self, app: ASGIApp, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.app = app
        self._clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = {}

    def _retry_after(self, key: tuple[str, str], limit: int) -> int:
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - RATE_WINDOW_S:
            hits.popleft()
        if len(hits) < limit:
            hits.append(now)
            return 0
        return max(1, math.ceil(hits[0] + RATE_WINDOW_S - now))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        ip = client[0] if client else "unknown"
        category, limit = rate_category(scope["method"], scope["path"])
        retry_after = self._retry_after((ip, category), limit)
        if not retry_after:
            await self.app(scope, receive, send)
            return
        code = ErrorCode.RATE_LIMITED
        request_id = scope.get("state", {}).get("request_id")
        body = json.dumps(
            error_body(code, code.default_message, {"retry_after_seconds": retry_after}, request_id)
        ).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": code.http_status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"retry-after", str(retry_after).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})
