"""Image uploads: PNG, JPEG or TIFF read by EasyOCR (B3, T3.4).

The page is 1. A multi-page TIFF is read from its first frame only. The EXIF orientation
of phone photos is applied before OCR. Output under 40 non-space characters fails the job
with OCR_EMPTY (E-07 threshold, A-20). Devanagari digits, and a visarga after Latin text,
are mapped to ASCII (D-7, SC-10).

Blocking, and uses the GPU: the caller holds the GPU semaphore and runs this through
anyio.to_thread.run_sync (1.2).
"""

from __future__ import annotations

from app.ingestion.loaders import (
    MIN_TEXT_CHARS,
    LoadedDocument,
    LoaderError,
    TextUnit,
    non_space_len,
)
from app.services.media.ocr_service import OcrService, normalise_case_ocr


def read_image(path: str):  # -> numpy.ndarray (RGB)
    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            img.seek(0)
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            return np.asarray(rgb)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise LoaderError("IMAGE_UNREADABLE", "The image could not be read.") from exc


def load_image(path: str, ocr: OcrService) -> LoadedDocument:
    text = normalise_case_ocr(ocr.ocr_image(read_image(path)))
    if non_space_len(text) < MIN_TEXT_CHARS:
        raise LoaderError("OCR_EMPTY", "No readable text was found in the image.")
    return LoadedDocument(
        media_type="image", kind="prose", units=[TextUnit(text, page=1)], page_count=1
    )
