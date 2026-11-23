"""Cross-references (P-10, P-11, P-17)."""

from __future__ import annotations

from app.ingestion.legal.reference_extractor import ReferenceExtractor, expand

ACT = "test-2020"
KNOWN = {
    *(f"sec:{ACT}:{n}" for n in range(1, 31)),
    f"cl:{ACT}:2:5", f"cl:{ACT}:4:1", f"cl:{ACT}:4:2", f"ch:{ACT}:II",
}
SRC = f"sec:{ACT}:10"


def _run(text: str, source: str = SRC, own: str = SRC):
    ex = ReferenceExtractor(ACT, ["The Test Act, 2020", "Test Act, 2020"], KNOWN)
    ex.extract(source, own, text)
    return ex.result


def targets(result) -> set[str]:
    return {e.target for e in result.refers_to}


def test_expand() -> None:
    assert expand("20 and 21") == ["20", "21"]
    assert expand("35, 36, 37 and 39") == ["35", "36", "37", "39"]
    assert expand("3 to 6") == ["3", "4", "5", "6"]
    assert expand("55A") == ["55A"]


def test_lists_ranges_and_line_breaks() -> None:
    r = _run("under sections 20 and\n21, and sections 3 to 5")
    assert targets(r) == {f"sec:{ACT}:{n}" for n in (20, 21, 3, 4, 5)}


def test_other_act_creates_refers_to_act_only() -> None:
    # P-10 / G4: 'section 193 and 228 of the Indian Penal Code' never links to this Act.
    r = _run("within the meaning of section 19 and 28 of the Indian Penal Code (45 of 1860)")
    assert targets(r) == set()
    (edge,) = r.refers_to_act
    assert edge.target == "act:indian-penal-code" and edge.section_ref == "19 and 28"
    assert r.stub_acts == {"act:indian-penal-code": "Indian Penal Code"}


def test_stub_ids_keep_the_year() -> None:
    # P-11, with the A-14 extension for 'Code of ...' names.
    r = _run("section 5 of the Code of Criminal Procedure, 1973 (2 of 1974) and "
             "section 6 of the General Clauses Act, 1897")
    assert {e.target for e in r.refers_to_act} == {
        "act:code-of-criminal-procedure-1973", "act:general-clauses-act-1897",
    }
    assert targets(r) == set()


def test_own_title_is_not_another_act() -> None:
    r = _run("section 5 of the Test Act, 2020")
    assert targets(r) == {f"sec:{ACT}:5"} and not r.refers_to_act


def test_qualifiers() -> None:
    r = _run("sub-section (1) of section 4, subsection (2) of section 4, "
             "sub-section (3) of section 4, clause (5) of section 2")
    assert targets(r) == {
        f"cl:{ACT}:4:1", f"cl:{ACT}:4:2",  # clause exists
        f"sec:{ACT}:4",  # (3) does not exist: fall back to the Section
        f"cl:{ACT}:2:5",
    }
    quals = {e.target: e.qualifier for e in r.refers_to}
    assert quals[f"cl:{ACT}:4:1"] == "sub-section (1)"


def test_qualified_reference_to_other_act() -> None:
    r = _run("clause (j) of sub-section (1) of section 3 of the Food Safety and Standards Act, 2006")
    assert targets(r) == set()
    assert {e.target for e in r.refers_to_act} == {"act:food-safety-and-standards-act-2006"}


def test_chapter_this_act_self_and_subsection_alone() -> None:
    r = _run("Chapter II of this Act and section 10 and sub-section 2 of this section")
    assert targets(r) == {f"ch:{ACT}:II"}  # self-reference and 'sub-section 2' ignored


def test_unknown_target_is_a_warning() -> None:
    r = _run("section 99")
    assert targets(r) == set()
    assert r.warnings == [f"reference from {SRC} to unknown sec:{ACT}:99"]


def test_clause_source_ignores_its_own_section() -> None:
    r = _run("as provided in section 10 and section 11", source=f"cl:{ACT}:10:1", own=SRC)
    assert targets(r) == {f"sec:{ACT}:11"}


def test_rules_refer_to_parent_act_and_own_rules() -> None:
    # P-17
    rules = "test-rules-2021"
    known = KNOWN | {f"sec:{rules}:{n}" for n in (1, 2, 3)} | {f"cl:{rules}:2:1"}
    ex = ReferenceExtractor(rules, ["Test Rules, 2021"], known, parent_slug=ACT,
                            parent_titles=["Test Act, 2020"])
    src = f"sec:{rules}:3"
    ex.extract(src, src, "under section 4 of the Act, section 5 of the Test Act, 2020, "
                         "sub-rule (1) of rule 2, rule 1 and section 79 of the Information Technology Act, 2000")
    assert {e.target for e in ex.result.refers_to} == {
        f"sec:{ACT}:4", f"sec:{ACT}:5", f"cl:{rules}:2:1", f"sec:{rules}:1",
    }
    assert {e.target for e in ex.result.refers_to_act} == {"act:information-technology-act-2000"}
