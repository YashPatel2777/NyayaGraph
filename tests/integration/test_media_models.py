"""Real EasyOCR and faster-whisper on the synthetic fixtures (B3, E-07, 07 T3.3 to T3.5).

Slow (model load; the first Whisper run downloads WHISPER_MODEL). Skipped without CUDA.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.config import get_settings
from app.ingestion.loaders.audio import load_audio
from app.ingestion.loaders.image import load_image
from app.ingestion.loaders.pdf import load_pdf
from app.services.media.ocr_service import OcrService
from app.services.media.transcription_service import TranscriptionService

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "case"


@pytest.fixture(scope="module")
def ocr() -> OcrService:
    service = OcrService(get_settings().OCR_LANGS)
    service.load()
    return service


def test_scanned_image(ocr: OcrService) -> None:
    doc = load_image(str(FIXTURES / "noisy" / "service_job_sheet.png"), ocr)
    text = doc.units[0].text
    for expected in ("Ravi Mehta", "manufacturing defect", "15/01/2026", "6,200"):
        assert expected in text
    assert not any("०" <= ch <= "९" for ch in text)  # D-7


def test_scanned_and_mixed_pdf(ocr: OcrService, tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Digital page: order QK-55120-8841 for one Zenith X2 phone.")
    scan = doc.new_page()
    scan.insert_image(scan.rect, filename=str(FIXTURES / "noisy" / "service_job_sheet.png"))
    doc.save(pdf)
    loaded = load_pdf(str(pdf), ocr)
    assert [u.page for u in loaded.units] == [1, 2]
    assert "QK-55120-8841" in loaded.units[0].text
    assert "manufacturing defect" in loaded.units[1].text


def test_voice_note_transcribes_on_the_gpu() -> None:
    service = TranscriptionService(get_settings().WHISPER_MODEL)
    doc = load_audio(str(FIXTURES / "clean" / "voice_note.wav"), service)
    text = " ".join(u.text for u in doc.units).lower()
    assert "manufacturing defect" in text and "refund" in text
    assert doc.language == "en"
    assert doc.warnings == []  # no CPU fallback, confident language detection
    assert 15 < doc.duration_sec < 60
    assert doc.units[0].start_sec == pytest.approx(0.0, abs=1.0)
    assert doc.units[-1].end_sec <= doc.duration_sec + 0.5
