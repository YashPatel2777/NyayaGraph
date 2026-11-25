"""HTTP client for the NyayaGraph REST API (SC-24).

The UI talks to the backend only through this module and only over HTTP (3.4), so the
backend layering (api -> agents/ingestion -> services -> schemas) is untouched. Nothing
here logs case text, file contents or question text (R-04).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Optional

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"  # 3.2: the API binds to 127.0.0.1

# 3.4 allowed upload types, keyed by file extension. The API checks the file signature
# itself; the UI only picks the declared MIME type the API expects.
MIME_BY_EXTENSION: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".mp4": "audio/mp4",
    ".m4a": "audio/x-m4a",
    ".ogg": "audio/ogg",
    ".txt": "text/plain",
}
UPLOAD_EXTENSIONS = [ext.lstrip(".") for ext in MIME_BY_EXTENSION]
QUERY_MODES = ("auto", "case_and_legal", "legal_only", "case_only")  # 3.4
TERMINAL_JOB_STATUSES = {"succeeded", "succeeded_with_warnings", "failed", "cancelled"}


def mime_for(filename: str) -> Optional[str]:
    """Declared MIME type for an upload, or None when 3.4 does not allow the extension."""
    return MIME_BY_EXTENSION.get(PurePath(filename).suffix.lower())


@dataclass
class ApiError(Exception):
    """A non-2xx response carrying the 3.3 error envelope."""

    status: int
    code: str
    message: str
    details: dict = field(default_factory=dict)
    request_id: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class ApiUnreachable(Exception):
    """The API did not answer (not started, wrong address, or the request timed out)."""


class ApiClient:
    """Thin synchronous wrapper over the 3.4 endpoints used by the UI."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout_s: float = 15.0,
        query_timeout_s: float = 75.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("UI_API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        # E-12: the API stops a query at 45 s; the UI waits longer so the 504 arrives.
        self.query_timeout_s = query_timeout_s
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout_s, transport=transport)

    # -- plumbing -----------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiUnreachable(f"The API at {self.base_url} timed out.") from exc
        except httpx.TransportError as exc:
            raise ApiUnreachable(f"Cannot reach the API at {self.base_url}.") from exc
        return response

    @staticmethod
    def _raise_for_error(response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            body = response.json()["error"]
            raise ApiError(
                status=response.status_code,
                code=str(body.get("code", "INTERNAL")),
                message=str(body.get("message", "")),
                details=body.get("details") or {},
                request_id=body.get("request_id"),
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ApiError(status=response.status_code, code="INTERNAL",
                           message=f"Unexpected HTTP {response.status_code} from the API.") from None

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request(method, path, **kwargs)
        self._raise_for_error(response)
        return response.json()

    def _all_pages(self, path: str) -> list[dict]:
        items: list[dict] = []
        cursor: Optional[str] = None
        while True:
            params: dict[str, Any] = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            page = self._json("GET", path, params=params)
            items.extend(page.get("items", []))
            cursor = page.get("next_cursor")
            if not cursor:
                return items

    # -- endpoints (3.4) ----------------------------------------------------------------
    def health(self) -> dict:
        """GET /health. A 503 carries the same body shape, so it is returned, not raised."""
        response = self._request("GET", "/health")
        if response.status_code in (200, 503):
            return response.json()
        self._raise_for_error(response)
        return response.json()

    def list_cases(self) -> list[dict]:
        return self._all_pages("/v1/cases")

    def create_case(self, title: str) -> dict:
        return self._json("POST", "/v1/cases", json={"title": title})

    def delete_case(self, case_id: str) -> dict:
        return self._json("DELETE", f"/v1/cases/{case_id}")

    def list_documents(self, case_id: str) -> list[dict]:
        return self._all_pages(f"/v1/cases/{case_id}/documents")

    def upload_document(self, case_id: str, filename: str, content: bytes,
                        language: Optional[str] = None) -> dict:
        mime = mime_for(filename)
        if mime is None:
            raise ApiError(status=415, code="UNSUPPORTED_MEDIA_TYPE",
                           message=f"File type not allowed: {PurePath(filename).suffix or 'none'}.",
                           details={"allowed": sorted(set(MIME_BY_EXTENSION.values()))})
        data = {"language": language} if language else None
        return self._json(
            "POST", f"/v1/cases/{case_id}/documents",
            files={"file": (filename, content, mime)}, data=data, timeout=120.0,
        )

    def get_job(self, job_id: str) -> dict:
        return self._json("GET", f"/v1/jobs/{job_id}")

    def query(self, question: str, *, case_id: Optional[str], thread_id: Optional[str] = None,
              mode: str = "auto", include_trace: bool = False,
              max_case_chunks: Optional[int] = None, max_provisions: Optional[int] = None) -> dict:
        options: dict[str, Any] = {"include_trace": include_trace}
        if max_case_chunks is not None:
            options["max_case_chunks"] = max_case_chunks
        if max_provisions is not None:
            options["max_provisions"] = max_provisions
        body: dict[str, Any] = {
            "question": question,
            "case_id": case_id,
            "thread_id": thread_id,
            "mode": mode,
            "options": options,
        }
        return self._json("POST", "/v1/query", json=body, timeout=self.query_timeout_s)

    def get_provision(self, provision_id: str) -> dict:
        return self._json("GET", f"/v1/legal/provisions/{provision_id}")

    def close(self) -> None:
        self._http.close()
