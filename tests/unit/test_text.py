"""Text uploads: UTF-8 decoding (IN-13, A-1) and chat detection (SC-1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.loaders import LoaderError
from app.ingestion.loaders.text import NOT_UTF8_WARNING, decode_text, load_text


def test_utf8_with_bom_and_hindi() -> None:
    text, warnings = decode_text("﻿शिकायत: phone".encode("utf-8"))
    assert text == "शिकायत: phone"
    assert warnings == []


def test_non_utf8_bytes_are_replaced_with_a_warning() -> None:
    text, warnings = decode_text("smart “quote”".encode("cp1252"))
    assert text == "smart �quote�"  # nothing silently dropped
    assert warnings == [NOT_UTF8_WARNING]


def test_plain_text_is_one_prose_unit_without_page(tmp_path: Path) -> None:
    path = tmp_path / "original.txt"
    path.write_text("Notes\n\nThe seller refused a refund on 12 January.", encoding="utf-8")
    doc = load_text(str(path))
    assert (doc.media_type, doc.kind) == ("text", "prose")
    assert len(doc.units) == 1
    assert doc.units[0].page is None


def test_chat_export_becomes_one_unit_per_message(tmp_path: Path) -> None:
    path = tmp_path / "original.txt"
    path.write_text(
        "12/01/2026, 14:33 - Ravi: Screen broke\n12/01/2026, 14:40 - Seller: No refund\n",
        encoding="utf-8",
    )
    doc = load_text(str(path))
    assert doc.kind == "chat"
    assert [u.text for u in doc.units] == [
        "[2026-01-12 14:33:00] Ravi: Screen broke",
        "[2026-01-12 14:40:00] Seller: No refund",
    ]


def test_chat_with_only_system_lines_falls_back_to_text(tmp_path: Path) -> None:
    path = tmp_path / "original.txt"
    path.write_text(
        "12/01/2026, 14:33 - Ravi: <Media omitted>\n12/01/2026, 14:40 - Seller: <Media omitted>\n",
        encoding="utf-8",
    )
    doc = load_text(str(path))
    assert doc.kind == "prose"
    assert len(doc.warnings) == 1


def test_whitespace_only_file_fails(tmp_path: Path) -> None:
    path = tmp_path / "original.txt"
    path.write_text(" \n\t\n", encoding="utf-8")
    with pytest.raises(LoaderError) as exc:
        load_text(str(path))
    assert exc.value.code == "TEXT_EMPTY"
