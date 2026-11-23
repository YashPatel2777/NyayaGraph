"""OCR: PyMuPDF renders a page at 200 DPI, EasyOCR reads it (section 0, 1.4 A3, B3).

The FastAPI lifespan calls load() once (1.2); scripts load the reader on first use. Callers
hold the GPU semaphore (app/ingestion/jobs.py). Calls are blocking: run them through
anyio.to_thread.run_sync. Output keeps line structure: boxes are grouped into lines by
vertical position, so line-based parsing rules still see line starts.
"""

from __future__ import annotations

import re
import threading
from typing import Any

RENDER_DPI = 200

# D-7: with en,hi EasyOCR reads ASCII digits in English text as Devanagari digits. Case
# document OCR maps them back; legal ingestion does not use this.
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# SC-10: it also reads a colon after an English word as the visarga (U+0903). Only a visarga
# after a Latin letter or ASCII digit is replaced, so Hindi text is unchanged.
_LATIN_VISARGA = re.compile(r"(?<=[A-Za-z0-9])ः")


def map_devanagari_digits(text: str) -> str:
    return text.translate(_DEVANAGARI_DIGITS)


def normalise_case_ocr(text: str) -> str:
    """D-7 and SC-10 clean-up for case-document OCR output."""
    return _LATIN_VISARGA.sub(":", map_devanagari_digits(text))


class OcrService:
    def __init__(self, langs: list[str], gpu: bool = True) -> None:
        self.langs = langs
        self.gpu = gpu
        self._reader: Any = None
        self._lock = threading.Lock()

    def _get_reader(self) -> Any:
        with self._lock:
            if self._reader is None:
                import easyocr

                self._reader = easyocr.Reader(self.langs, gpu=self.gpu, verbose=False)
            return self._reader

    def load(self) -> None:
        self._get_reader()

    def ocr_pdf_page(self, pdf_path: str, page_index: int) -> str:
        import numpy as np
        import pymupdf

        with pymupdf.open(pdf_path) as doc:
            pix = doc[page_index].get_pixmap(dpi=RENDER_DPI)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return self.ocr_image(image)

    def ocr_image(self, image: Any) -> str:
        boxes = self._get_reader().readtext(image, detail=1, paragraph=False)
        items = sorted(
            (
                sum(p[1] for p in box) / 4,  # vertical centre
                min(p[0] for p in box),  # left edge
                max(p[1] for p in box) - min(p[1] for p in box),  # height
                text,
            )
            for box, text, _conf in boxes
        )
        lines: list[list[tuple[float, str]]] = []
        centre: float | None = None
        for y, x, h, text in items:
            if centre is None or abs(y - centre) > max(h * 0.5, 8):
                lines.append([])
                centre = y
            lines[-1].append((x, text))
        return "\n".join(" ".join(t for _, t in sorted(line)) for line in lines)
