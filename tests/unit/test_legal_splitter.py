"""Retrieval-unit rule and search_text (2.3, A-11)."""

from __future__ import annotations

from app.ingestion.chunking.legal_splitter import UNIT_MAX_CHARS, LegalSplitter, split_text
from app.ingestion.legal.parser import (
    ParsedChapter,
    ParsedClause,
    ParsedSection,
    ParseResult,
)


def _result(sections, kind="act", chapters=None) -> ParseResult:
    return ParseResult(kind, chapters or [ParsedChapter("I", "PRELIMINARY")], sections,
                       None, None, [], [])


def _sentences(n: int, word: str = "Sentence") -> str:
    return " ".join(f"{word} number {i} says something about consumers." for i in range(n))


def test_split_text_respects_limit_and_prefers_provisos() -> None:
    text = _sentences(30) + "\nProvided that " + _sentences(30, "Proviso")
    parts = split_text(text)
    assert len(parts) >= 2
    assert all(len(p) <= UNIT_MAX_CHARS for p in parts)
    assert any(p.startswith("Provided that") for p in parts)
    assert " ".join(parts).split() == text.split()  # nothing lost or reworded


def test_split_text_cuts_an_overlong_sentence() -> None:
    parts = split_text("word " * 900)
    assert all(len(p) <= UNIT_MAX_CHARS for p in parts) and len(parts) == 3


def test_short_section_is_the_unit_and_its_clauses_are_not() -> None:
    sec = ParsedSection("5", "Short", "I", "(1) A. (2) B.",
                        [ParsedClause("1", "subsection", "A."), ParsedClause("2", "subsection", "B.")])
    units = LegalSplitter(_result([sec]), "test-2020", "Test Act, 2020").units()
    assert [(u.id, u.kind) for u in units] == [("sec:test-2020:5", "section")]
    assert units[0].search_text == (
        "Test Act, 2020 | Chapter I: PRELIMINARY | Section 5. Short\n(1) A. (2) B."
    )


def test_long_section_uses_clauses_and_long_clause_splits_into_parts() -> None:
    long_clause = _sentences(60)
    sec = ParsedSection("6", "Long", "I", "x" * 2500,
                        [ParsedClause("1", "subsection", "Short clause."),
                         ParsedClause("2", "subsection", long_clause)])
    units = LegalSplitter(_result([sec]), "test-2020", "Test Act, 2020").units()
    ids = [u.id for u in units]
    assert ids[0] == "cl:test-2020:6:1"
    assert ids[1:] == [f"cl:test-2020:6:2-p{i}" for i in range(1, len(ids))]
    assert units[0].search_text.startswith(
        "Test Act, 2020 | Chapter I: PRELIMINARY | Section 6. Long | (1)\n")
    assert all(u.kind == "part" for u in units[1:])


def test_long_section_without_clauses_splits_itself() -> None:
    sec = ParsedSection("7", "No clauses", "I", _sentences(60), [])
    units = LegalSplitter(_result([sec]), "test-2020", "T").units()
    assert units[0].id == "sec:test-2020:7-p1" and all(u.kind == "part" for u in units)


def test_definitions_are_units_and_the_section_is_not() -> None:
    sec = ParsedSection("2", "Definitions", "I", "short",
                        [ParsedClause("1", "definition", "“a” means a;", "a")])
    splitter = LegalSplitter(_result([sec]), "test-2020", "T")
    assert not splitter.section_is_unit(sec)
    assert [u.id for u in splitter.units()] == ["cl:test-2020:2:1"]


def test_rules_use_rule_wording_and_no_chapter() -> None:
    sec = ParsedSection("3", "Fee", None, "Ten rupees.", [])
    units = LegalSplitter(_result([sec], kind="rule", chapters=[]), "test-rules", "Test Rules").units()
    assert units[0].search_text == "Test Rules | Rule 3. Fee\nTen rupees."
