"""Case chunking (B4, 4.2, IN-07, SC-2, A-2, A-21).

Target 1,200 characters, hard maximum 1,500, overlap 150.

- prose (pdf, image, text): split on blank lines, then on sentence ends for a long
  paragraph, then at whitespace for a long sentence. A heading line (short, one line, no
  closing full stop) is attached to the paragraph after it, so it never ends a chunk
  (IN-07). A chunk never crosses a page and no overlap is carried across one (A-2).
- chat: whole messages are packed; one message is split only when it is too long (SC-2).
  The overlap is the previous chunk's last message when it fits in 150 characters, never
  part of a message (A-21).
- audio: timestamped units are packed. The chunk's start_sec is the start of the unit its
  first character comes from (including the overlap) and end_sec the end of its last unit.

A piece is at most MAX - OVERLAP - 2 characters, so overlap plus one piece plus its joiner
always fits in MAX. Paragraphs longer than that, not only those over 1,500, are therefore
sentence-split.

Chunk ids are chk:{doc_uuid}:{order, 4 digits} (2.1); orders are stable across runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable

from app.ingestion.loaders import TextUnit, UnitKind

TARGET = 1200
MAX = 1500
OVERLAP = 150
PIECE_MAX = MAX - OVERLAP - 2
HEADING_MAX = 80
# How far into the overlap window we look for a word boundary before cutting mid-word.
OVERLAP_WORD_SEARCH = 30

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+(?=\S)")
_HEADING_END = re.compile(r"[.!?।,;]$")


@dataclass(frozen=True)
class Chunk:
    order: int
    text: str
    page: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None


@dataclass(frozen=True)
class _Piece:
    text: str
    joiner: str  # placed before this piece when it is not first in a chunk
    page: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    whole: bool = True  # False for a fragment of a split paragraph or message


def is_heading(paragraph: str) -> bool:
    return (
        "\n" not in paragraph
        and len(paragraph) <= HEADING_MAX
        and any(ch.isalpha() for ch in paragraph)
        and not _HEADING_END.search(paragraph)
    )


def hard_split(text: str, limit: int) -> list[tuple[str, str]]:
    """Split at the last whitespace before limit; cut at limit if there is none nearby.

    Returns (separator before the part, part); a mid-word cut has an empty separator so
    joining the parts restores the text.
    """
    out: list[tuple[str, str]] = []
    rest, sep = text, ""
    while len(rest) > limit:
        cut = max(rest.rfind(" ", limit // 2, limit + 1), rest.rfind("\n", limit // 2, limit + 1))
        if cut == -1:
            out.append((sep, rest[:limit]))
            rest, sep = rest[limit:], ""
        else:
            out.append((sep, rest[:cut]))
            rest, sep = rest[cut + 1 :], rest[cut]
    if rest:
        out.append((sep, rest))
    return out


def split_long(text: str) -> list[tuple[str, str]]:
    """(separator, part) pairs: text longer than PIECE_MAX is sentence-split, and any
    sentence still too long is hard-split."""
    if len(text) <= PIECE_MAX:
        return [("", text)]
    parts: list[tuple[str, str]] = []
    for i, sentence in enumerate(_SENTENCE_END.split(text)):
        sep = " " if i else ""
        if len(sentence) > PIECE_MAX:
            pieces = hard_split(sentence, PIECE_MAX)
            parts.append((sep, pieces[0][1]))
            parts.extend(pieces[1:])
        else:
            parts.append((sep, sentence))
    return parts


def _pieces_from(text: str, joiner: str, unit: TextUnit) -> list[_Piece]:
    parts = split_long(text)
    whole = len(parts) == 1
    return [
        _Piece(part, joiner if i == 0 else sep, unit.page, unit.start_sec, unit.end_sec, whole)
        for i, (sep, part) in enumerate(parts)
    ]


def _prose_pieces(unit: TextUnit) -> list[_Piece]:
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(unit.text)]
    paragraphs = [p for p in paragraphs if p]
    merged: list[str] = []
    pending = ""
    for index, para in enumerate(paragraphs):
        text = f"{pending}\n{para}" if pending else para
        pending = ""
        if is_heading(para) and index < len(paragraphs) - 1:
            pending = text  # IN-07: the heading travels with the paragraph after it
            continue
        merged.append(text)
    if pending:
        merged.append(pending)
    pieces: list[_Piece] = []
    for para in merged:
        pieces.extend(_pieces_from(para, "\n\n", unit))
    return pieces


def _render(pieces: list[_Piece]) -> str:
    return pieces[0].text + "".join(p.joiner + p.text for p in pieces[1:])


def _text_overlap(pieces: list[_Piece]) -> list[_Piece]:
    """The last OVERLAP characters, moved forward to a word start when one is near."""
    full = _render(pieces)
    if len(full) <= OVERLAP:
        tail_start = 0
    else:
        tail_start = len(full) - OVERLAP
        if not full[tail_start - 1].isspace():
            window = full[tail_start : tail_start + OVERLAP_WORD_SEARCH]
            space = next((i for i, ch in enumerate(window) if ch.isspace()), -1)
            if space != -1:
                tail_start += space + 1
    while tail_start < len(full) and full[tail_start].isspace():
        tail_start += 1
    tail = full[tail_start:]
    if not tail:
        return []
    # The unit the tail starts in gives the overlap's start_sec (audio).
    offset, source = 0, pieces[0]
    for i, piece in enumerate(pieces):
        offset += len(piece.text) + (len(piece.joiner) if i else 0)
        if offset > tail_start:
            source = piece
            break
    return [_Piece(tail, "", source.page, source.start_sec, pieces[-1].end_sec, whole=False)]


def _message_overlap(pieces: list[_Piece]) -> list[_Piece]:
    last = pieces[-1]
    if last.whole and len(last.text) <= OVERLAP:
        return [replace(last, joiner="")]
    return []


def _pack(pieces: list[_Piece], overlap: Callable[[list[_Piece]], list[_Piece]]) -> list[list[_Piece]]:
    groups: list[list[_Piece]] = []
    current: list[_Piece] = []
    fresh = 0  # pieces in current that are not overlap
    for piece in pieces:
        if fresh and len(_render([*current, piece])) > TARGET:
            groups.append(current)
            current = overlap(current)
            fresh = 0
        current.append(piece)
        fresh += 1
    if fresh:
        groups.append(current)
    return groups


def _to_chunk(order: int, group: list[_Piece]) -> Chunk:
    text = _render(group)
    if len(text) > MAX:  # guaranteed by PIECE_MAX; fail loudly rather than store it
        raise AssertionError("chunk exceeds the B4 maximum")
    ends = [p.end_sec for p in group if p.end_sec is not None]
    return Chunk(
        order=order,
        text=text,
        page=group[0].page,
        start_sec=group[0].start_sec,
        end_sec=max(ends) if ends else None,
    )


def split_units(units: list[TextUnit], kind: UnitKind) -> list[Chunk]:
    """Chunks for one document's loader output, in reading order."""
    if kind == "prose":
        # A-2: each page (or the single text unit) is packed on its own.
        segments = [(_prose_pieces(u), _text_overlap) for u in units]
    elif kind == "chat":
        pieces = [p for u in units for p in _pieces_from(u.text, "\n\n", u)]
        segments = [(pieces, _message_overlap)]
    else:
        pieces = [p for u in units for p in _pieces_from(u.text, "\n", u)]
        segments = [(pieces, _text_overlap)]

    chunks: list[Chunk] = []
    for pieces, overlap in segments:
        for group in _pack(pieces, overlap):
            chunks.append(_to_chunk(len(chunks), group))
    return chunks
