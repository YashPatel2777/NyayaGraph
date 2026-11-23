"""PDF loader: pypdf per page, OCR under 40 characters, E-07, A-19, D-7, IN-05 (07 T3.3).

OCR is a fake here; tests/integration/test_media_models.py runs the real EasyOCR.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.ingestion.loaders import LoaderError
from app.ingestion.loaders.pdf import load_pdf, read_pdf_text

PAGE_TEXT = "Invoice QK-55120-8841 for one Zenith X2 smartphone, total paid Rs. 18,499."


class FakeOcr:
    def __init__(self, text: str = "Service centre job sheet: display defect ०१५ confirmed.") -> None:
        self.text = text
        self.calls: list[int] = []

    def ocr_pdf_page(self, pdf_path: str, page_index: int) -> str:
        self.calls.append(page_index)
        return self.text

    def ocr_image(self, image) -> str:  # pragma: no cover - not used by the PDF loader
        return self.text


def make_pdf(path: Path, pages: list[str | None], **save_kwargs) -> Path:
    """None makes an image-only page (a 'scan' with no text layer); '' a blank page."""
    doc = pymupdf.open()
    for content in pages:
        page = doc.new_page()
        if content:
            page.insert_text((72, 100), content, fontsize=11)
        elif content is None:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 50), False)
            pix.clear_with(200)
            page.insert_image(pymupdf.Rect(72, 72, 272, 122), pixmap=pix)
    doc.save(path, **save_kwargs)
    return path


def test_digital_pdf_needs_no_ocr(tmp_path: Path) -> None:
    ocr = FakeOcr()
    doc = load_pdf(str(make_pdf(tmp_path / "a.pdf", [PAGE_TEXT, PAGE_TEXT + " Page two."])), ocr)
    assert ocr.calls == []
    assert doc.page_count == 2
    assert [u.page for u in doc.units] == [1, 2]  # IN-05: page is structure, not guessed
    assert "Page two." in doc.units[1].text
    assert doc.warnings == []


def test_scanned_pdf_goes_to_ocr_and_maps_devanagari_digits(tmp_path: Path) -> None:
    ocr = FakeOcr()
    doc = load_pdf(str(make_pdf(tmp_path / "s.pdf", [None])), ocr)
    assert ocr.calls == [0]
    assert doc.units[0].page == 1
    assert "015 confirmed" in doc.units[0].text  # D-7


def test_mixed_pdf_ocrs_only_the_image_page(tmp_path: Path) -> None:
    ocr = FakeOcr()
    doc = load_pdf(str(make_pdf(tmp_path / "m.pdf", [PAGE_TEXT, None, PAGE_TEXT])), ocr)
    assert ocr.calls == [1]
    assert [u.page for u in doc.units] == [1, 2, 3]


def test_blank_page_is_skipped_with_a_warning(tmp_path: Path) -> None:
    ocr = FakeOcr(text="  ")
    doc = load_pdf(str(make_pdf(tmp_path / "b.pdf", [PAGE_TEXT, "", PAGE_TEXT])), ocr)
    assert ocr.calls == [1]
    assert [u.page for u in doc.units] == [1, 3]
    assert doc.page_count == 3
    assert doc.warnings == ["pdf: 1 blank page(s) skipped: 2"]


def test_whole_file_under_40_characters_is_ocr_empty(tmp_path: Path) -> None:
    with pytest.raises(LoaderError) as exc:
        load_pdf(str(make_pdf(tmp_path / "e.pdf", [None, ""])), FakeOcr(text="12 34"))
    assert exc.value.code == "OCR_EMPTY"


def test_password_protected_pdf_is_pdf_encrypted(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path / "p.pdf", [PAGE_TEXT],
        encryption=pymupdf.PDF_ENCRYPT_RC4_128, user_pw="secret", owner_pw="owner",
    )
    with pytest.raises(LoaderError) as exc:
        load_pdf(str(path), FakeOcr())
    assert exc.value.code == "PDF_ENCRYPTED"


def test_aes_pdfs(tmp_path: Path) -> None:
    locked = make_pdf(
        tmp_path / "aes.pdf", [PAGE_TEXT],
        encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner",
    )
    with pytest.raises(LoaderError) as exc:
        read_pdf_text(str(locked))
    assert exc.value.code == "PDF_ENCRYPTED"
    # A-19: AES with an empty user password (common for invoices) opens.
    open_aes = make_pdf(
        tmp_path / "aes-open.pdf", [PAGE_TEXT],
        encryption=pymupdf.PDF_ENCRYPT_AES_128, user_pw="", owner_pw="owner",
    )
    assert "QK-55120-8841" in read_pdf_text(str(open_aes))[0].text


def test_missing_crypto_provider_is_pdf_encrypted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pypdf import _reader
    from pypdf.errors import DependencyError

    def no_crypto(*args, **kwargs):
        raise DependencyError("cryptography>=3.1 is required for AES algorithm")

    monkeypatch.setattr(_reader.PdfReader, "__init__", no_crypto)
    with pytest.raises(LoaderError) as exc:
        read_pdf_text(str(tmp_path / "any.pdf"))
    assert exc.value.code == "PDF_ENCRYPTED"


def test_owner_password_only_pdf_is_read(tmp_path: Path) -> None:
    # A-19: an empty user password opens in any reader, so the file is read.
    path = make_pdf(
        tmp_path / "o.pdf", [PAGE_TEXT],
        encryption=pymupdf.PDF_ENCRYPT_RC4_128, user_pw="", owner_pw="owner",
    )
    doc = load_pdf(str(path), FakeOcr())
    assert "QK-55120-8841" in doc.units[0].text


def test_truncated_and_garbage_pdfs_are_unreadable(tmp_path: Path) -> None:
    good = make_pdf(tmp_path / "g.pdf", [PAGE_TEXT, PAGE_TEXT]).read_bytes()
    truncated = tmp_path / "t.pdf"
    truncated.write_bytes(good[: len(good) // 3])
    garbage = tmp_path / "x.pdf"
    garbage.write_bytes(b"%PDF-1.7\nthis is not a pdf at all\n")
    for path in (truncated, garbage):
        with pytest.raises(LoaderError) as exc:
            load_pdf(str(path), FakeOcr())
        assert exc.value.code in {"PDF_UNREADABLE", "OCR_EMPTY"}, path.name
    with pytest.raises(LoaderError) as exc:
        load_pdf(str(garbage), FakeOcr())
    assert exc.value.code == "PDF_UNREADABLE"


def test_fixture_invoice_pdf() -> None:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "case" / "clean" / "invoice_and_letter.pdf"
    doc = load_pdf(str(path), FakeOcr())
    assert doc.page_count == 2
    assert "Rs. 18,499.00" in doc.units[0].text
