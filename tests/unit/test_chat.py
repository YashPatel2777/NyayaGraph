"""Chat export parsing (SC-1, SC-2, IN-02, IN-09; 07 T3.2)."""

from __future__ import annotations

import json
from pathlib import Path

from app.ingestion.loaders.chat import (
    infer_day_first,
    is_chat_export,
    normalise_timestamp,
    parse_chat,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "case"


def lines(content: str) -> list[str]:
    return [m.line() for m in parse_chat(content)]


def test_doc07_android_fixture_gives_exactly_two_messages() -> None:
    content = (FIXTURES / "chat_whatsapp_android.txt").read_text(encoding="utf-8")
    assert is_chat_export(content)
    assert lines(content) == [
        "[2026-01-12 14:33:00] Ravi: Sir my phone screen broke in 2 days\nand seller refuses refund",
        "[2026-01-12 14:40:00] Seller: No refund policy",
    ]


def test_ios_bracket_layout_with_narrow_no_break_space_and_marks() -> None:
    content = (
        "[12/01/2026, 2:33:05 PM] Ravi: Screen is broken\n"
        "[12/01/2026, 2:34:10 PM] Seller: ‎image omitted\n"
        "[12/01/2026, 2:35:00 PM] Family Group: ‎Ravi left\n"
        "[13/01/2026, 9:00:00 AM] Seller: Only 2 days left\n"
    )
    assert lines(content) == [
        "[2026-01-12 14:33:05] Ravi: Screen is broken",
        "[2026-01-13 09:00:00] Seller: Only 2 days left",
    ]


def test_telegram_bracket_layout() -> None:
    content = (
        "[15.01.2026 14:32:05] Ravi: Hello\n"
        "[15.01.2026 14:33] Seller Bot: Your ticket is 4471\n"
    )
    assert lines(content) == [
        "[2026-01-15 14:32:05] Ravi: Hello",
        "[2026-01-15 14:33:00] Seller Bot: Your ticket is 4471",
    ]


def test_telegram_two_line_layout_with_continuation() -> None:
    content = (
        "Ravi Mehta, [15.01.2026 14:32]\n"
        "The phone has green lines\n"
        "since 6 January\n"
        "Seller, [15.01.2026 14:40]\n"
        "Please visit the service centre\n"
    )
    assert lines(content) == [
        "[2026-01-15 14:32:00] Ravi Mehta: The phone has green lines\nsince 6 January",
        "[2026-01-15 14:40:00] Seller: Please visit the service centre",
    ]


def test_telegram_json_export() -> None:
    content = json.dumps(
        {
            "name": "Seller",
            "messages": [
                {"type": "service", "actor": "Ravi", "action": "create_group"},
                {"type": "message", "date": "2026-01-15T14:32:05", "from": "Ravi",
                 "text": ["Order ", {"type": "bold", "text": "QK-55120"}, " is defective"]},
                {"type": "message", "date": "2026-01-15T14:33:00", "from": None, "text": ""},
                {"type": "message", "date": "2026-01-15T14:34:00", "from": "Seller", "text": "Noted"},
            ],
        }
    )
    assert is_chat_export(content)
    assert lines(content) == [
        "[2026-01-15 14:32:05] Ravi: Order QK-55120 is defective",
        "[2026-01-15 14:34:00] Seller: Noted",
    ]


def test_month_first_file_is_inferred_from_the_whole_file() -> None:
    # IN-09: 1/15/26 can only be month-first, so 2/3/26 in the same file is 3 February.
    content = (
        "2/3/26, 9:15 AM - Ravi: First message\n"
        "1/15/26, 2:32 PM - Seller: Second message\n"
    )
    assert lines(content) == [
        "[2026-02-03 09:15:00] Ravi: First message",
        "[2026-01-15 14:32:00] Seller: Second message",
    ]


def test_ambiguous_dates_default_to_day_first() -> None:
    assert infer_day_first(["03/04/2026", "05/06/2026"]) is True
    assert infer_day_first(["13/04/2026"]) is True
    assert infer_day_first(["04/13/2026"]) is False
    assert infer_day_first(["2026-01-15"]) is True


def test_normalise_timestamp() -> None:
    assert normalise_timestamp("12/01/2026", "14:33", True) == "2026-01-12 14:33:00"
    assert normalise_timestamp("12/01/26", "2:32 pm", True) == "2026-01-12 14:32:00"
    assert normalise_timestamp("12/01/26", "12:05 a.m.", True) == "2026-01-12 00:05:00"
    assert normalise_timestamp("2026-01-12", "09:00:07", True) == "2026-01-12 09:00:07"
    assert normalise_timestamp("31/02/2026", "25:00", True) is None


def test_unreadable_timestamp_keeps_the_raw_text() -> None:
    content = "12/01/2026, 25:61 - Ravi: hi\n12/01/2026, 10:00 - Ravi: there\n"
    assert lines(content)[0] == "[12/01/2026 25:61] Ravi: hi"


def test_ordinary_sentences_that_look_like_events_are_kept() -> None:
    content = (
        "12/01/2026, 14:33 - Ravi: the charge was added to my bill\n"
        "12/01/2026, 14:34 - Ravi: I added the item to my cart\n"
        "12/01/2026, 14:35 - Seller: it says the payment was removed\n"
        "12/01/2026, 14:36 - Ravi Mehta joined using this group's invite link\n"
        "12/01/2026, 14:37 - Ravi: This message was deleted\n"
    )
    assert [m.body for m in parse_chat(content)] == [
        "the charge was added to my bill",
        "I added the item to my cart",
        "it says the payment was removed",
    ]


def test_edited_tag_is_removed() -> None:
    content = (
        "12/01/2026, 14:33 - Ravi: refund please <This message was edited>\n"
        "12/01/2026, 14:34 - Seller: ok\n"
    )
    assert lines(content)[0] == "[2026-01-12 14:33:00] Ravi: refund please"


def test_plain_note_is_not_a_chat() -> None:
    note = (
        "Notes for the complaint\n\n"
        "On 12/01/2026 I called the seller. They said: no refund.\n"
        "Time of call: 14:33\n"
    )
    assert not is_chat_export(note)
    assert not is_chat_export('{"not": "telegram"}')
    assert not is_chat_export("{broken json")


def test_fixture_seller_chat() -> None:
    content = (FIXTURES / "clean" / "chat_seller.txt").read_text(encoding="utf-8")
    messages = parse_chat(content)
    assert len(messages) == 9  # 11 lines with headers; encryption notice and media dropped
    assert messages[0].sender == "QuickKart Seller Support"
    assert messages[6].body == "This is the photo of the screen.\nI want a refund or a replacement."
    assert messages[-1].timestamp == "2026-01-15 18:20:00"
