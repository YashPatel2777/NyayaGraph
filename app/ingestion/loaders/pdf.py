"""PDF uploads (B3, E-07, IN-05).

pypdf reads each page. A page with fewer than 40 non-space characters is rendered at 200 DPI
and read by EasyOCR. The two steps are separate so the job holds the GPU semaphore only for
OCR (1.2). The page number travels with the text as structure, never as a marker in the text
(IN-05).

E-07: encrypted fails with PDF_ENCRYPTED, unreadable with PDF_UNREADABLE, and under 40
characters for the whole file after OCR with OCR_EMPTY. Blank pages are skipped with a
warning. A-19: a PDF encrypted with an empty user password opens without one in any reader,
so it is decrypted and read.

All functions here are blocking: run them through anyio.to_thread.run_sync.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.ingestion.loaders import (
    MIN_TEXT_CHARS,
    LoadedDocument,
    LoaderError,
    TextUnit,
    non_space_len,
)
from app.services.media.ocr_service import OcrService, normalise_case_ocr

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    page: int  # 1-based
    text: str
    ocr: bool = False

    @property
    def needs_ocr(self) -> bool:
        return not self.ocr and non_space_len(self.text) < MIN_TEXT_CHARS


def read_pdf_text(path: str) -> list[PageText]:
    """pypdf text of every page. Raises LoaderError for encrypted or unreadable files."""
    from pypdf import PdfReader
    from pypdf.errors import DependencyError, PyPdfError

    try:
        try:
            reader = PdfReader(path)
        except DependencyError as exc:  # an AES file and no crypto provider
            raise LoaderError("PDF_ENCRYPTED", "The PDF is password protected.") from exc
        if reader.is_encrypted:
            try:
                opened = reader.decrypt("")
            except Exception:  # includes a missing AES dependency
                opened = 0
            if not opened:
                raise LoaderError("PDF_ENCRYPTED", "The PDF is password protected.")
        pages = reader.pages
        count = len(pages)
    except LoaderError:
        raise
    except (PyPdfError, OSError, ValueError, KeyError, TypeError) as exc:
        raise LoaderError("PDF_UNREADABLE", "The PDF could not be read.") from exc
    if count == 0:
        raise LoaderError("PDF_UNREADABLE", "The PDF has no pages.")

    result: list[PageText] = []
    for index in range(count):
        try:
            text = pages[index].extract_text() or ""
        except Exception:  # a broken page content stream: let OCR try the rendered page
            text = ""
        result.append(PageText(page=index + 1, text=text))
    return result


def ocr_pages(path: str, pages: list[PageText], ocr: OcrService) -> int:
    """OCR every page that needs it, in place. Returns the number of pages OCR'd.

    The caller holds the GPU semaphore (1.2).
    """
    done = 0
    for page in pages:
        if page.needs_ocr:
            try:
                text = ocr.ocr_pdf_page(path, page.page - 1)
            except Exception as exc:
                raise LoaderError("PDF_UNREADABLE", "A PDF page could not be rendered.") from exc
            page.text = normalise_case_ocr(text)  # D-7, SC-10
            page.ocr = True
            done += 1
    return done


def to_document(pages: list[PageText]) -> LoadedDocument:
    """Skip blank pages with a warning; fail OCR_EMPTY if the whole file has under 40
    characters (E-07)."""
    if sum(non_space_len(p.text) for p in pages) < MIN_TEXT_CHARS:
        raise LoaderError("OCR_EMPTY", "No readable text was found in the PDF.")
    blank = [p.page for p in pages if non_space_len(p.text) == 0]
    warnings = []
    if blank:
        listed = ", ".join(str(n) for n in blank[:20]) + (" ..." if len(blank) > 20 else "")
        warnings.append(f"pdf: {len(blank)} blank page(s) skipped: {listed}")
    units = [TextUnit(p.text, page=p.page) for p in pages if non_space_len(p.text) > 0]
    return LoadedDocument(
        media_type="pdf", kind="prose", units=units, page_count=len(pages), warnings=warnings
    )


def needs_ocr(pages: list[PageText]) -> bool:
    return any(p.needs_ocr for p in pages)


def load_pdf(path: str, ocr: OcrService | None) -> LoadedDocument:
    """All three steps in one call, for scripts and tests that need no semaphore."""
    pages = read_pdf_text(path)
    if ocr is not None and needs_ocr(pages):
        ocr_pages(path, pages, ocr)
    return to_document(pages)
