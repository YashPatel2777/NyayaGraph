"""Image loader (B3, T3.4, A-20, D-7). OCR is a fake here."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.ingestion.loaders import LoaderError
from app.ingestion.loaders.image import load_image, read_image


class FakeOcr:
    def __init__(self, text: str) -> None:
        self.text = text
        self.shapes: list[tuple[int, ...]] = []

    def ocr_image(self, image) -> str:
        self.shapes.append(image.shape)
        return self.text


def test_png_is_read_as_page_one_with_digits_mapped(tmp_path: Path) -> None:
    path = tmp_path / "original.png"
    Image.new("L", (120, 80), 255).save(path)
    ocr = FakeOcr("Job sheet ZSC-२०२६०११५-०७७, display defect confirmed")
    doc = load_image(str(path), ocr)
    assert ocr.shapes == [(80, 120, 3)]  # converted to RGB
    assert doc.units[0].page == 1 and doc.page_count == 1
    assert "ZSC-20260115-077" in doc.units[0].text


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    path = tmp_path / "original.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    Image.new("RGB", (120, 80), "white").save(path, exif=exif)
    assert read_image(str(path)).shape == (120, 80, 3)


def test_multipage_tiff_uses_the_first_frame(tmp_path: Path) -> None:
    path = tmp_path / "original.tif"
    first, second = Image.new("RGB", (50, 40)), Image.new("RGB", (90, 60))
    first.save(path, save_all=True, append_images=[second])
    assert read_image(str(path)).shape == (40, 50, 3)


def test_short_ocr_output_is_ocr_empty(tmp_path: Path) -> None:
    path = tmp_path / "original.png"
    Image.new("L", (10, 10), 255).save(path)
    with pytest.raises(LoaderError) as exc:
        load_image(str(path), FakeOcr("Rs 500"))
    assert exc.value.code == "OCR_EMPTY"


def test_unreadable_image(tmp_path: Path) -> None:
    path = tmp_path / "original.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    with pytest.raises(LoaderError) as exc:
        load_image(str(path), FakeOcr("x" * 50))
    assert exc.value.code == "IMAGE_UNREADABLE"
