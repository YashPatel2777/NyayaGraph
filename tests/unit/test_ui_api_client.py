"""SC-24: the UI's HTTP client, against a mocked API (no server, no Neo4j)."""

from __future__ import annotations

import json

import httpx
import pytest

from ui.api_client import ApiClient, ApiError, ApiUnreachable, mime_for


def make(handler) -> ApiClient:
    return ApiClient("http://api.test", transport=httpx.MockTransport(handler))


def envelope(status: int, code: str, **details) -> httpx.Response:
    return httpx.Response(status, json={"error": {
        "code": code, "message": "msg", "details": details, "request_id": "r-1"}})


def test_error_envelope_becomes_api_error():
    c = make(lambda req: envelope(404, "CASE_NOT_FOUND"))
    with pytest.raises(ApiError) as info:
        c.list_documents("abc")
    assert info.value.status == 404
    assert info.value.code == "CASE_NOT_FOUND"
    assert info.value.request_id == "r-1"


def test_non_envelope_error_is_internal():
    c = make(lambda req: httpx.Response(500, text="boom"))
    with pytest.raises(ApiError) as info:
        c.get_job("j")
    assert info.value.code == "INTERNAL"


def test_connection_failure_is_unreachable():
    def handler(req):
        raise httpx.ConnectError("refused", request=req)
    with pytest.raises(ApiUnreachable):
        make(handler).health()


def test_health_503_is_returned_not_raised():
    body = {"status": "fail", "checks": {"neo4j": "ok", "indexes": "ok",
                                         "embedding_meta": "ok", "ollama": "fail"}}
    c = make(lambda req: httpx.Response(503, json=body))
    assert c.health() == body


def test_list_cases_follows_cursor():
    seen = []

    def handler(req):
        cursor = req.url.params.get("cursor")
        seen.append(cursor)
        assert req.url.params["limit"] == "100"
        if cursor is None:
            return httpx.Response(200, json={"items": [{"case_id": "a"}], "next_cursor": "c2"})
        return httpx.Response(200, json={"items": [{"case_id": "b"}], "next_cursor": None})

    assert [c["case_id"] for c in make(handler).list_cases()] == ["a", "b"]
    assert seen == [None, "c2"]


def test_query_body_matches_3_4():
    captured = {}

    def handler(req):
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"thread_id": "t-1"})

    make(handler).query("Is this a defect?", case_id="c1", thread_id=None,
                        mode="case_and_legal", include_trace=True)
    assert captured == {"question": "Is this a defect?", "case_id": "c1", "thread_id": None,
                        "mode": "case_and_legal", "options": {"include_trace": True}}
    assert "user_id" not in captured  # 3.2: a request never sets user_id


def test_query_options_are_sent_when_given():
    captured = {}

    def handler(req):
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"thread_id": "t-1"})

    make(handler).query("Is this a defect?", case_id="c1", max_case_chunks=3, max_provisions=4)
    assert captured["options"] == {"include_trace": False, "max_case_chunks": 3, "max_provisions": 4}


def test_upload_sends_declared_mime_and_language():
    captured = {}

    def handler(req):
        captured["ctype"] = req.headers["content-type"]
        captured["body"] = req.content
        return httpx.Response(202, json={"doc_id": "d", "job_id": "j",
                                         "processing_status": "queued", "duplicate": False})

    res = make(handler).upload_document("c1", "Voice.WAV", b"RIFF....", language="hi")
    assert res["job_id"] == "j"
    assert captured["ctype"].startswith("multipart/form-data")
    assert b"Content-Type: audio/wav" in captured["body"]
    assert b'name="language"' in captured["body"] and b"hi" in captured["body"]


def test_upload_rejects_disallowed_extension_without_calling_api():
    def handler(req):
        raise AssertionError("must not be called")
    with pytest.raises(ApiError) as info:
        make(handler).upload_document("c1", "letter.docx", b"PK")
    assert info.value.code == "UNSUPPORTED_MEDIA_TYPE"


@pytest.mark.parametrize("name,mime", [
    ("a.pdf", "application/pdf"), ("b.JPG", "image/jpeg"), ("c.tiff", "image/tiff"),
    ("d.m4a", "audio/x-m4a"), ("e.txt", "text/plain"), ("f.docx", None), ("noext", None),
])
def test_mime_for(name, mime):
    assert mime_for(name) == mime


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("UI_API_BASE_URL", "http://127.0.0.1:9000/")
    assert ApiClient().base_url == "http://127.0.0.1:9000"
