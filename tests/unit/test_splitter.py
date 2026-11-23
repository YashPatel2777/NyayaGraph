"""Case chunking (B4, IN-07, SC-2, A-2, A-21; 07 T3.6)."""

from __future__ import annotations

import random

from app.ingestion.chunking.splitter import MAX, OVERLAP, TARGET, Chunk, split_units
from app.ingestion.loaders import TextUnit
from app.schemas.graph_schema import chunk_id

DOC_UUID = "9f1c2d3e-4b5a-4c6d-8e7f-0a1b2c3d4e5f"
WORDS = "seller refund screen defect warranty invoice delivery complaint service centre".split()


def sentences(n_chars: int, seed: int = 1) -> str:
    rng = random.Random(seed)
    out: list[str] = []
    total = 0
    while total < n_chars:
        s = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14))).capitalize() + "."
        out.append(s)
        total += len(s) + 1
    return " ".join(out)


def overlap_len(prev: str, nxt: str) -> int:
    """Length of the longest prefix of nxt that is a suffix of prev."""
    for k in range(min(len(prev), len(nxt), OVERLAP + 50), 0, -1):
        if prev.endswith(nxt[:k]):
            return k
    return 0


def assert_sizes(chunks: list[Chunk]) -> None:
    assert chunks
    assert all(len(c.text) <= MAX for c in chunks)
    assert [c.order for c in chunks] == list(range(len(chunks)))


def test_6000_char_paragraph_gives_bounded_chunks_with_150_overlap() -> None:
    text = sentences(6000)
    assert "\n\n" not in text and len(text) >= 6000
    chunks = split_units([TextUnit(text, page=1)], "prose")
    assert_sizes(chunks)
    assert len(chunks) >= 5
    for prev, nxt in zip(chunks, chunks[1:]):
        k = overlap_len(prev.text, nxt.text)
        assert OVERLAP - 30 <= k <= OVERLAP, k
    # Every chunk except the last is filled close to the target.
    assert all(len(c.text) >= TARGET - 200 for c in chunks[:-1])


def test_paragraph_without_sentence_ends_or_spaces_is_still_bounded() -> None:
    rng = random.Random(2)
    no_stops = " ".join(rng.choice(WORDS) for _ in range(900))
    # Random letters, so the overlap between chunks is unambiguous.
    one_token = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6000))
    for text in (no_stops, one_token):
        chunks = split_units([TextUnit(text, page=3)], "prose")
        assert_sizes(chunks)
        assert all(c.page == 3 for c in chunks)
    # Nothing is lost from the unbroken token: chunks minus overlaps rebuild it.
    chunks = split_units([TextUnit(one_token)], "prose")
    rebuilt = chunks[0].text + "".join(
        c.text[overlap_len(p.text, c.text):] for p, c in zip(chunks, chunks[1:])
    )
    assert rebuilt == one_token


def test_heading_is_attached_to_the_text_after_it() -> None:
    # IN-07: without the rule 'Steps already taken' would end the first chunk.
    para_a = sentences(1100, seed=3)
    heading = "Steps already taken"
    para_b = sentences(800, seed=4)
    chunks = split_units([TextUnit(f"{para_a}\n\n{heading}\n\n{para_b}", page=1)], "prose")
    assert_sizes(chunks)
    assert not any(c.text.rstrip().endswith(heading) for c in chunks)
    holder = next(c for c in chunks if heading in c.text)
    assert f"{heading}\n{para_b[:40]}" in holder.text


def test_heading_rule_ignores_sentences_and_trailing_headings() -> None:
    text = "Short sentence.\n\nAnother short one.\n\nClosing heading"
    chunks = split_units([TextUnit(text)], "prose")
    assert len(chunks) == 1
    assert chunks[0].text == "Short sentence.\n\nAnother short one.\n\nClosing heading"


def test_no_chunk_crosses_a_page() -> None:
    # A-2: pages are packed separately and overlap never carries across a page.
    units = [
        TextUnit("PAGEONE " + sentences(2000, seed=5), page=1),
        TextUnit("short page two PAGETWO", page=2),
        TextUnit("PAGETHREE " + sentences(1300, seed=6), page=3),
    ]
    chunks = split_units(units, "prose")
    assert_sizes(chunks)
    for c in chunks:
        markers = [m for m in ("PAGEONE", "PAGETWO", "PAGETHREE") if m in c.text]
        assert len(markers) <= 1
    by_page = {c.page for c in chunks}
    assert by_page == {1, 2, 3}
    page_two = [c for c in chunks if c.page == 2]
    assert [c.text for c in page_two] == ["short page two PAGETWO"]  # short page, short chunk
    first_of_three = next(c for c in chunks if c.page == 3)
    assert first_of_three.text.startswith("PAGETHREE")


def test_ids_and_orders_are_stable_across_runs() -> None:
    units = [TextUnit(sentences(4000, seed=7), page=1), TextUnit(sentences(500, seed=8), page=2)]
    first = split_units(units, "prose")
    second = split_units(units, "prose")
    assert first == second
    ids = [chunk_id(DOC_UUID, c.order) for c in first]
    assert ids[0] == f"chk:{DOC_UUID}:0000"
    assert len(set(ids)) == len(ids)


def test_chat_packs_whole_messages() -> None:
    rng = random.Random(9)
    messages = [
        f"[2026-01-12 14:{i:02d}:00] {'Ravi' if i % 2 else 'Seller'}: {sentences(rng.randint(40, 400), seed=i)}"
        for i in range(40)
    ]
    chunks = split_units([TextUnit(m) for m in messages], "chat")
    assert_sizes(chunks)
    for c in chunks:
        parts = c.text.split("\n\n")
        assert all(p in messages for p in parts), "a message was split across chunks"
    for prev, nxt in zip(chunks, chunks[1:]):
        first = nxt.text.split("\n\n")[0]
        last = prev.text.split("\n\n")[-1]
        # A-21: overlap is the previous chunk's last message when it fits in 150 characters.
        if len(last) <= OVERLAP:
            assert first == last
        else:
            assert first != last
    covered = {p for c in chunks for p in c.text.split("\n\n")}
    assert covered == set(messages)


def test_long_chat_message_is_split_at_sentence_ends() -> None:
    long_msg = "[2026-01-12 14:33:00] Ravi: " + sentences(3200, seed=10)
    chunks = split_units([TextUnit("[2026-01-12 14:30:00] Seller: Hi"), TextUnit(long_msg)], "chat")
    assert_sizes(chunks)
    assert len(chunks) >= 3


def test_audio_chunks_carry_start_and_end_seconds() -> None:
    units = [
        TextUnit(sentences(500, seed=20 + i), start_sec=i * 30.0, end_sec=i * 30.0 + 28.0)
        for i in range(8)
    ]
    chunks = split_units(units, "audio")
    assert_sizes(chunks)
    assert chunks[0].start_sec == 0.0
    assert chunks[-1].end_sec == 7 * 30.0 + 28.0
    for c in chunks:
        assert c.page is None
        assert c.start_sec is not None and c.end_sec is not None and c.start_sec < c.end_sec
    for prev, nxt in zip(chunks, chunks[1:]):
        assert overlap_len(prev.text, nxt.text) > 0
        # The overlap comes from the previous chunk, so the next chunk starts inside it.
        assert prev.start_sec <= nxt.start_sec <= prev.end_sec


def test_short_text_is_one_chunk_and_empty_input_gives_none() -> None:
    assert split_units([TextUnit("Invoice total Rs. 18,499", page=1)], "prose") == [
        Chunk(order=0, text="Invoice total Rs. 18,499", page=1)
    ]
    assert split_units([], "audio") == []
