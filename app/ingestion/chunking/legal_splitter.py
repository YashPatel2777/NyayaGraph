"""Retrieval-unit rule and embedded text for the Legal KB (2.3, 1.4 A6).

- A Section is a unit when its text is at most 2,000 characters; otherwise its subsection
  Clauses are the units. Each definition is always its own unit, and the Definitions section
  is never a unit (P-08).
- A unit over 2,000 characters is split at proviso or sentence boundaries into parts with
  ids suffixed -p1, -p2 (A-11: parts are LegalUnit nodes under the same Section; the
  whole Clause keeps its full text but is not embedded).
- Clauses of a Section that is itself a unit are not units (A-11), so text is not embedded
  twice.
- search_text (also the embedded text; EmbeddingService adds the prefix, R-07):
  {short_title} | Chapter {roman}: {chapter title} | Section {number}. {title} | ({label})
  {text}
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.legal.parser import ParsedClause, ParsedSection, ParseResult
from app.schemas.graph_schema import clause_id, section_id

UNIT_MAX_CHARS = 2000  # 2.3

PROVISO_SPLIT_RE = re.compile(r"\n(?=Provided)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:—])\s+(?=[(A-Z“])")


@dataclass
class Unit:
    id: str
    kind: str  # 'section', 'clause' or 'part'
    section_number: str
    label: str | None
    text: str
    search_text: str


def split_text(text: str, limit: int = UNIT_MAX_CHARS) -> list[str]:
    """Split at proviso boundaries, then sentence boundaries, packing pieces up to limit.
    A single sentence longer than limit is cut at the last space before limit."""
    pieces: list[tuple[bool, str]] = []  # (starts a proviso block, piece)
    for block in PROVISO_SPLIT_RE.split(text):
        sentences = [p for p in SENTENCE_SPLIT_RE.split(block) if p.strip()]
        pieces.extend((i == 0 and block.startswith("Provided"), p) for i, p in enumerate(sentences))
    parts: list[str] = []
    current = ""
    for proviso_start, piece in pieces:
        if proviso_start and current:  # a proviso boundary starts a new part
            parts.append(current)
            current = ""
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                parts.append(current)
                current = ""
            parts.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        joined = f"{current} {piece}".strip() if current else piece
        if len(joined) <= limit:
            current = joined
        else:
            parts.append(current)
            current = piece
    if current:
        parts.append(current)
    return parts


class LegalSplitter:
    def __init__(self, parsed: ParseResult, act_slug: str, short_title: str) -> None:
        self.parsed = parsed
        self.act_slug = act_slug
        self.short_title = short_title
        self.chapters = {c.number: c.title for c in parsed.chapters}
        self.word = "Rule" if parsed.kind != "act" else "Section"

    def header(self, sec: ParsedSection, label: str | None) -> str:
        parts = [self.short_title]
        if sec.chapter_number:
            parts.append(f"Chapter {sec.chapter_number}: {self.chapters.get(sec.chapter_number, '')}")
        parts.append(f"{self.word} {sec.number}. {sec.title}")
        if label:
            parts.append(f"({label})")
        return " | ".join(parts)

    @staticmethod
    def is_definitions(sec: ParsedSection) -> bool:
        return any(c.kind == "definition" for c in sec.clauses)

    def section_is_unit(self, sec: ParsedSection) -> bool:
        return not self.is_definitions(sec) and len(sec.text) <= UNIT_MAX_CHARS

    def clause_is_unit(self, sec: ParsedSection, clause: ParsedClause) -> bool:
        if clause.kind == "definition":
            return len(clause.text) <= UNIT_MAX_CHARS
        return not self.section_is_unit(sec) and len(clause.text) <= UNIT_MAX_CHARS

    def units(self) -> list[Unit]:
        out: list[Unit] = []
        for sec in self.parsed.sections:
            sid = section_id(self.act_slug, sec.number)
            if self.section_is_unit(sec):
                out.append(Unit(sid, "section", sec.number, None, sec.text,
                                f"{self.header(sec, None)}\n{sec.text}"))
                continue
            embedded_clause = False
            for clause in sec.clauses:
                if clause.kind == "subsection" and self.section_is_unit(sec):
                    continue
                cid = clause_id(self.act_slug, sec.number, clause.label)
                if len(clause.text) <= UNIT_MAX_CHARS:
                    out.append(Unit(cid, "clause", sec.number, clause.label, clause.text,
                                    f"{self.header(sec, clause.label)}\n{clause.text}"))
                else:
                    for n, part in enumerate(split_text(clause.text), start=1):
                        out.append(Unit(f"{cid}-p{n}", "part", sec.number, clause.label, part,
                                        f"{self.header(sec, clause.label)}\n{part}"))
                embedded_clause = True
            if not embedded_clause and not self.is_definitions(sec):
                # A long Section with no subsection Clauses: split the Section itself (A-11).
                for n, part in enumerate(split_text(sec.text), start=1):
                    out.append(Unit(section_id(self.act_slug, sec.number, part=n), "part",
                                    sec.number, None, part, f"{self.header(sec, None)}\n{part}"))
        return out
