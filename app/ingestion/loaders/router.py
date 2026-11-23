"""Upload type routing: declared MIME and file signature to a media type (3.4, E-13, IN-11).

The allowed types and limits are the 3.4 table. The declared MIME type picks the type; the
file signature (first bytes, read with filetype) must agree with it, otherwise 415. Routing
never looks at the file name or extension (IN-01, IN-11).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Final

import filetype

from app.errors import AppError, ErrorCode
from app.ingestion.loaders import MediaType

MB: Final = 1024 * 1024
# Bytes the signature check reads; filetype needs at most 262.
SIGNATURE_BYTES: Final = 8192


@dataclass(frozen=True)
class UploadType:
    mime: str
    media_type: MediaType
    extension: str  # stored file name is original.{extension} (B2)
    max_bytes: int
    # Signatures (filetype MIME names) accepted for this declared type. Empty for text,
    # which has no signature.
    signatures: frozenset[str]


def _t(mime: str, media: MediaType, ext: str, max_mb: int, *sigs: str) -> UploadType:
    return UploadType(mime, media, ext, max_mb * MB, frozenset(sigs))


# 3.4 table. An MP4 audio file may carry the generic MP4 signature, so audio/mp4 and
# audio/x-m4a accept both M4A and MP4 containers.
ALLOWED: Final[dict[str, UploadType]] = {
    t.mime: t
    for t in (
        _t("application/pdf", "pdf", "pdf", 50, "application/pdf"),
        _t("image/png", "image", "png", 20, "image/png"),
        _t("image/jpeg", "image", "jpg", 20, "image/jpeg"),
        _t("image/tiff", "image", "tif", 20, "image/tiff"),
        _t("audio/mpeg", "audio", "mp3", 200, "audio/mpeg"),
        _t("audio/wav", "audio", "wav", 200, "audio/x-wav"),
        _t("audio/mp4", "audio", "m4a", 200, "audio/mp4", "video/mp4"),
        _t("audio/x-m4a", "audio", "m4a", 200, "audio/mp4", "video/mp4"),
        _t("audio/ogg", "audio", "ogg", 200, "audio/ogg"),
        _t("text/plain", "text", "txt", 5),
    )
}
ALLOWED_MIMES: Final = sorted(ALLOWED)

# Content limits in 3.4 that need the file opened.
MAX_PDF_PAGES: Final = 500
MAX_AUDIO_SEC: Final = 120 * 60


def normalise_mime(declared: str | None) -> str:
    """'Text/Plain; charset=utf-8' -> 'text/plain'."""
    return (declared or "").split(";", 1)[0].strip().lower()


def _unsupported() -> AppError:
    return AppError(ErrorCode.UNSUPPORTED_MEDIA_TYPE, details={"allowed": ALLOWED_MIMES})


def upload_type(declared_mime: str | None) -> UploadType:
    """The 3.4 entry for a declared MIME type, or 415."""
    found = ALLOWED.get(normalise_mime(declared_mime))
    if found is None:
        raise _unsupported()
    return found


def check_size(kind: UploadType, size: int) -> None:
    if size > kind.max_bytes:
        raise AppError(ErrorCode.FILE_TOO_LARGE, details={"max_bytes": kind.max_bytes})
    if size == 0:
        # A-18: an empty file is a missing file field.
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            details={"fields": [{"loc": ["body", "file"], "msg": "file is empty"}]},
        )


def check_signature(kind: UploadType, head: bytes) -> None:
    """The file's first bytes must agree with the declared MIME type (3.4, E-13)."""
    guessed = filetype.guess_mime(head[:SIGNATURE_BYTES])
    if kind.media_type == "text":
        # Plain text has no signature: any known binary signature, or a NUL byte, is not text.
        if guessed is not None or b"\x00" in head[:SIGNATURE_BYTES]:
            raise _unsupported()
        return
    if guessed not in kind.signatures:
        raise _unsupported()


def validate_upload(declared_mime: str | None, data: bytes) -> UploadType:
    """Type, size and signature checks of B1 and E-13. Nothing is written before this passes."""
    kind = upload_type(declared_mime)
    check_size(kind, len(data))
    check_signature(kind, data)
    return kind


def check_content_limits(kind: UploadType, data: bytes) -> None:
    """The page and duration limits of 3.4 (A-18). Blocking: call through a worker thread.

    A file that cannot be opened here is not rejected: it is stored and the job fails with
    the E-07 or E-08 code, so the user sees why.
    """
    if kind.media_type == "pdf":
        pages = _pdf_pages(data)
        if pages is not None and pages > MAX_PDF_PAGES:
            raise AppError(ErrorCode.FILE_TOO_LARGE, details={"max_pages": MAX_PDF_PAGES})
    elif kind.media_type == "audio":
        seconds = _audio_seconds(data)
        if seconds is not None and seconds > MAX_AUDIO_SEC:
            raise AppError(ErrorCode.FILE_TOO_LARGE, details={"max_duration_sec": MAX_AUDIO_SEC})


def _pdf_pages(data: bytes) -> int | None:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return None
        return len(reader.pages)
    except Exception:
        return None


def _audio_seconds(data: bytes) -> float | None:
    import av

    try:
        with av.open(io.BytesIO(data)) as container:
            if container.duration is not None:
                return container.duration / av.time_base
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
    except Exception:
        return None
    return None


def route_stored(declared_mime: str, path: str) -> UploadType:
    """Re-check a stored upload before a job reads it (the job may run after a restart)."""
    kind = upload_type(declared_mime)
    with open(path, "rb") as fh:
        check_signature(kind, fh.read(SIGNATURE_BYTES))
    return kind
