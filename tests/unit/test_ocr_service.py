"""Case OCR clean-up (D-7, SC-10)."""

from __future__ import annotations

from app.services.media.ocr_service import map_devanagari_digits, normalise_case_ocr


def test_devanagari_digits_become_ascii() -> None:
    assert map_devanagari_digits("Rs. १८,४९९ on ०२/०१/२०२६") == "Rs. 18,499 on 02/01/2026"


def test_visarga_after_latin_text_becomes_a_colon() -> None:
    assert normalise_case_ocr("Customerः Ravi Mehta") == "Customer: Ravi Mehta"
    assert normalise_case_ocr("Time १४ः३०") == "Time 14:30"  # digits first, then SC-10


def test_visarga_in_hindi_is_unchanged() -> None:
    hindi = "दुःख और अतः"
    assert normalise_case_ocr(hindi) == hindi
    assert normalise_case_ocr("Statusः दुःख") == "Status: दुःख"
