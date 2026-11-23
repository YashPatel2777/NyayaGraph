"""Upload routing: 3.4 table, signature agreement, E-13, IN-11 (07 T3.1)."""

from __future__ import annotations

import io
import wave

import pytest

from app.errors import AppError, ErrorCode
from app.ingestion.loaders import router
from app.ingestion.loaders.router import check_content_limits, validate_upload

PAD = b"\x00" * 300

SAMPLES = {
    "application/pdf": b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n" + b"1 0 obj\n" * 40,
    "image/png": b"\x89PNG\r\n\x1a\n" + PAD,
    "image/jpeg": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + PAD,
    "image/tiff": b"II*\x00\x08\x00\x00\x00" + PAD,
    "audio/mpeg": b"ID3\x04\x00\x00\x00\x00\x00\x00" + PAD,
    "audio/mp4": b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00" + PAD,
    "audio/x-m4a": b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00" + PAD,
    "audio/ogg": b"OggS\x00\x02" + PAD,
    "text/plain": "12/01/2026, 14:33 - Ravi: नमस्ते, the screen broke\n".encode("utf-8"),
}


def _wav_bytes(seconds: float = 0.5) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


@pytest.mark.parametrize("mime", sorted(SAMPLES))
def test_each_allowed_type_is_accepted(mime: str) -> None:
    assert validate_upload(mime, SAMPLES[mime]).mime == mime


def test_wav_is_accepted_as_audio() -> None:
    assert validate_upload("audio/wav", _wav_bytes()).media_type == "audio"


@pytest.mark.parametrize(
    ("mime", "media"),
    [("application/pdf", "pdf"), ("image/png", "image"), ("audio/ogg", "audio"), ("text/plain", "text")],
)
def test_media_types(mime: str, media: str) -> None:
    assert validate_upload(mime, SAMPLES[mime]).media_type == media


def test_mime_parameters_and_case_are_ignored() -> None:
    assert validate_upload("Text/Plain; charset=UTF-8", SAMPLES["text/plain"]).mime == "text/plain"


def _code(exc: pytest.ExceptionInfo[AppError]) -> ErrorCode:
    return exc.value.code


def test_png_declared_as_pdf_is_rejected() -> None:
    with pytest.raises(AppError) as exc:
        validate_upload("application/pdf", SAMPLES["image/png"])
    assert _code(exc) == ErrorCode.UNSUPPORTED_MEDIA_TYPE
    assert exc.value.details["allowed"] == router.ALLOWED_MIMES


def test_docx_is_rejected_by_type_and_by_signature() -> None:
    docx = b"PK\x03\x04\x14\x00\x06\x00" + b"[Content_Types].xml" + PAD
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    with pytest.raises(AppError) as exc:
        validate_upload(docx_mime, docx)
    assert _code(exc) == ErrorCode.UNSUPPORTED_MEDIA_TYPE
    with pytest.raises(AppError) as exc:
        validate_upload("application/pdf", docx)
    assert _code(exc) == ErrorCode.UNSUPPORTED_MEDIA_TYPE


def test_binary_declared_as_text_is_rejected() -> None:
    for data in (SAMPLES["image/png"], b"plain start\x00\x01binary"):
        with pytest.raises(AppError) as exc:
            validate_upload("text/plain", data)
        assert _code(exc) == ErrorCode.UNSUPPORTED_MEDIA_TYPE


def test_missing_or_unknown_mime_is_rejected() -> None:
    for mime in (None, "", "application/json", "video/mp4"):
        with pytest.raises(AppError) as exc:
            validate_upload(mime, SAMPLES["text/plain"])
        assert _code(exc) == ErrorCode.UNSUPPORTED_MEDIA_TYPE


def test_empty_file_is_a_validation_error() -> None:
    # A-18
    with pytest.raises(AppError) as exc:
        validate_upload("application/pdf", b"")
    assert _code(exc) == ErrorCode.VALIDATION_ERROR
    assert exc.value.details["fields"][0]["loc"] == ["body", "file"]


def test_oversized_file_is_413_with_max_bytes() -> None:
    kind = router.ALLOWED["text/plain"]
    with pytest.raises(AppError) as exc:
        router.check_size(kind, kind.max_bytes + 1)
    assert _code(exc) == ErrorCode.FILE_TOO_LARGE
    assert exc.value.details == {"max_bytes": 5 * 1024 * 1024}
    router.check_size(kind, kind.max_bytes)  # exactly at the limit is allowed


def test_limits_match_the_3_4_table() -> None:
    mb = 1024 * 1024
    assert router.ALLOWED["application/pdf"].max_bytes == 50 * mb
    assert all(router.ALLOWED[m].max_bytes == 20 * mb for m in ("image/png", "image/jpeg", "image/tiff"))
    audio = ("audio/mpeg", "audio/wav", "audio/mp4", "audio/x-m4a", "audio/ogg")
    assert all(router.ALLOWED[m].max_bytes == 200 * mb for m in audio)
    assert router.ALLOWED["text/plain"].max_bytes == 5 * mb
    assert set(router.ALLOWED) == set(audio) | {
        "application/pdf", "image/png", "image/jpeg", "image/tiff", "text/plain"
    }


def test_pdf_page_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    kind = router.ALLOWED["application/pdf"]
    monkeypatch.setattr(router, "_pdf_pages", lambda data: 501)
    with pytest.raises(AppError) as exc:
        check_content_limits(kind, b"%PDF-")
    assert _code(exc) == ErrorCode.FILE_TOO_LARGE
    assert exc.value.details == {"max_pages": 500}
    monkeypatch.setattr(router, "_pdf_pages", lambda data: 500)
    check_content_limits(kind, b"%PDF-")


def test_audio_duration_limit_reads_real_header(monkeypatch: pytest.MonkeyPatch) -> None:
    kind = router.ALLOWED["audio/wav"]
    assert router._audio_seconds(_wav_bytes(0.5)) == pytest.approx(0.5, abs=0.05)
    check_content_limits(kind, _wav_bytes(0.5))
    monkeypatch.setattr(router, "MAX_AUDIO_SEC", 0.25)
    with pytest.raises(AppError) as exc:
        check_content_limits(kind, _wav_bytes(0.5))
    assert exc.value.details == {"max_duration_sec": 0.25}


def test_unopenable_content_is_left_to_the_job() -> None:
    # A corrupt PDF or undecodable audio is stored and fails in the job (E-07, E-08).
    check_content_limits(router.ALLOWED["application/pdf"], b"%PDF-garbage")
    check_content_limits(router.ALLOWED["audio/ogg"], b"OggS\x00garbage")


def test_route_stored_rechecks_the_signature(tmp_path) -> None:
    path = tmp_path / "original.pdf"
    path.write_bytes(SAMPLES["image/png"])
    with pytest.raises(AppError):
        router.route_stored("application/pdf", str(path))
    path.write_bytes(SAMPLES["application/pdf"])
    assert router.route_stored("application/pdf", str(path)).media_type == "pdf"
