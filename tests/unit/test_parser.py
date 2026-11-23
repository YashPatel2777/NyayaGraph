"""Statute parser on synthetic pages (P-01 to P-09, P-12, P-13 to P-16, P-18)."""

from __future__ import annotations

from datetime import date

import pytest

from app.ingestion.legal.parser import ParseError, parse_statute

ACT_PAGES = [
    # Page 1: title page and arrangement, dropped by P-02.
    "The Test Act, 2020\nARRANGEMENT OF SECTIONS\n1. Short title.\n2. Definitions.\nIndiaCode1",
    # Page 2
    "2\nTHE TEST ACT, 2020\nAn Act to test parsing.\nBE it enacted as follows:—\n"
    "CHAPTER I \nPRELIMINARY\n"
    "1. Short title.—(1) This Act may be called the Test Act, 2020.\n"
    "(2) It extends to India.\n"
    "2. Definitions.—In this Act,—\n"
    "(1) “buyer” means a person who buys;\n"
    "(2) “seller” means a person who sells and includes a person referred to in sub-section\n"
    "(1) of section 3;\n"
    "1. 5th March, 2021.—s. 1 to 3 (both inclusive) [Except sections 2], vide Notifn. No. S.O. 1(E).\n"
    "IndiaCode",
    # Page 3
    "3\nCHAPTER II\nRIGHTS AND\nREMEDIES\n"
    "3. Right to refund.—(1) A buyer may claim a refund.\n"
    "1[(1A) Inserted text.]\n"
    "(2) More text.\n"
    "4. Penalty for a long title that\nwraps onto the next line.—Any person who contravenes this Act shall be punished.\n"
    "1. Ins. by Act 12 of 2021, s. 2 (w.e.f. 01-02-2021).\n"
    "IndiaCode",
    # Page 4
    "4\nSTATEMENT OF OBJECTS AND REASONS\nWhy this Act exists.",
]


@pytest.fixture(scope="module")
def act():
    return parse_statute(ACT_PAGES, "act", 4)


def test_chapters_and_sections(act) -> None:
    assert [(c.number, c.title) for c in act.chapters] == [
        ("I", "PRELIMINARY"), ("II", "RIGHTS AND REMEDIES"),  # P-04, trailing space tolerated
    ]
    assert [s.number for s in act.sections] == ["1", "2", "3", "4"]
    assert act.sections[0].chapter_number == "I" and act.sections[2].chapter_number == "II"


def test_heading_wrap_joins_lines(act) -> None:
    # P-06: a heading that wraps is joined until '.—' appears.
    assert act.sections[3].title == "Penalty for a long title that wraps onto the next line"
    assert act.sections[3].text.startswith("Any person who contravenes")


def test_definitions_and_wrapped_line_is_not_a_subsection(act) -> None:
    defs = act.sections[1].clauses
    assert [(c.label, c.kind, c.term) for c in defs] == [
        ("1", "definition", "buyer"), ("2", "definition", "seller"),
    ]
    # '(1) of section 3;' starts a printed line but is not a new clause.
    assert defs[1].text.endswith("sub-section (1) of section 3;")
    assert defs[0].text == "“buyer” means a person who buys;"  # label stripped


def test_subsections_and_marker(act) -> None:
    sec3 = act.sections[2]
    assert [c.label for c in sec3.clauses] == ["1", "1A", "2"]  # P-07 after the 1[ marker
    assert "1[" not in sec3.text and "[(1A)" in sec3.text
    assert sec3.clauses[1].text == "Inserted text.]"


def test_amendment_footnote(act) -> None:
    # P-09: note on the section with the marker; Act number and w.e.f. date.
    assert len(act.amendments) == 1
    a = act.amendments[0]
    assert (a.section_number, a.act_number, a.effective_on) == ("3", "12 of 2021", date(2021, 2, 1))
    assert act.sections[2].amendment_notes == ["Ins. by Act 12 of 2021, s. 2 (w.e.f. 01-02-2021)."]
    assert "Ins. by" not in act.sections[3].text  # P-05: footnote removed from the body


def test_commencement_dates(act) -> None:
    # P-09: 's. 1 to 3 [Except sections 2]'; unlisted sections get null.
    got = {s.number: s.in_force_from for s in act.sections}
    assert got == {"1": date(2021, 3, 5), "2": None, "3": date(2021, 3, 5), "4": None}


def test_cleaning_and_body_bounds(act) -> None:
    all_text = " ".join(s.text for s in act.sections)
    assert "IndiaCode" not in all_text  # P-01
    assert "ARRANGEMENT" not in all_text and "BE it enacted" not in all_text  # P-02
    assert act.objects_text == "STATEMENT OF OBJECTS AND REASONS Why this Act exists."  # P-03
    assert "STATEMENT" not in all_text


def test_body_start_missing() -> None:
    with pytest.raises(ParseError) as info:
        parse_statute(["1. Short title.—Text."], "act", None)
    assert info.value.code == "PARSE_BODY_START_NOT_FOUND"


def test_expected_count_mismatch_fails() -> None:
    with pytest.raises(ParseError) as info:
        parse_statute(ACT_PAGES, "act", 5)
    assert info.value.code == "PARSE_VALIDATION_FAILED"
    assert any("expected 5 sections, parsed 4" in p for p in info.value.problems)


def test_gap_in_numbering_fails() -> None:
    pages = ["An Act to test.\n1. One.—Text.\n2. Two.—Text.\n4. Four.—Text."]
    with pytest.raises(ParseError) as info:
        parse_statute(pages, "act", None)
    assert info.value.code == "PARSE_VALIDATION_FAILED"
    assert any("['4']" in p for p in info.value.problems)


def test_letter_suffix_sections_accepted() -> None:
    pages = ["An Act to test.\n1. One.—Text.\n1A. One A.—Text.\n2. Two.—Text."]
    assert [s.number for s in parse_statute(pages, "act", 3).sections] == ["1", "1A", "2"]


RULE_PAGES = [
    "2 THE GAZETTE OF INDIA : EXTRAORDINARY\n"
    "3. जिला आयोग की अधिकारिता – कुछ पाठ\n"
    "NOTIFICATION\n"
    "G.S.R. 1(E).—In exercise of the powers conferred by section 5 of the Test Act, 2020, the\n"
    "Central Government hereby makes the following rules, namely:—\n"
    "1.  Short title.—(1) These rules may be called the Test Rules, 2021.\n"
    "(2) They come into force on publication.\n"
    "2.  Definitions.—(1) In these rules,—\n"
    " (a)   “Act” means the Test Act, 2020;\n"
    "(b) “form” means a form annexed;\n"
    "(2) Other words have the meaning in the Act.\n"
    "3.  Fee.—The fee is ten  rupees.\n"
    "[F. No. 1/2/2021]\nSOME OFFICER, Jt. Secy.\n"
    "Uploaded by the Government Press."
]


@pytest.fixture(scope="module")
def rules():
    return parse_statute(RULE_PAGES, "rule", 3)


def test_rules_body_start_and_enabling_text(rules) -> None:
    # P-13: everything before rule 1 is dropped, including the Hindi line starting '3.'.
    assert [s.number for s in rules.sections] == ["1", "2", "3"]
    assert rules.enabling_text.startswith("In exercise of the powers conferred by section 5")  # P-14
    assert rules.objects_text is None
    assert rules.chapters == [] and rules.sections[0].chapter_number is None  # P-15


def test_rules_definitions_and_subsections(rules) -> None:
    clauses = [(c.label, c.kind, c.term) for c in rules.sections[1].clauses]
    assert clauses == [
        ("1", "subsection", None), ("a", "definition", "Act"),
        ("b", "definition", "form"), ("2", "subsection", None),
    ]  # P-16


def test_rules_body_ends_before_signature(rules) -> None:
    assert rules.sections[2].text == "The fee is ten rupees."  # spacing collapsed, wording kept
    joined = " ".join(s.text for s in rules.sections)
    assert "F. No." not in joined and "Uploaded" not in joined
    assert "जिला" not in joined


def test_rules_without_heading_one_fail() -> None:
    with pytest.raises(ParseError) as info:
        parse_statute(["NOTIFICATION\nSome text only."], "rule", None)
    assert info.value.code == "PARSE_BODY_START_NOT_FOUND"


def test_notification_amendment_gives_note_only() -> None:
    # P-18: no Act number, so no AMENDS data beyond the note.
    pages = [
        "1. Short title.—These rules may be called the Test Rules.\n"
        "2. Duties.—2[(1) Every entity shall comply.]\n"
        "2. Subs. by G.S.R. 328(E), dated 17-5-2021 (w.e.f. 17-5-2021). Prior to substitution it read as:\n"
        "\"(1) Old text.\"\n"
    ]
    r = parse_statute(pages, "rule", 2)
    assert r.amendments[0].act_number is None
    assert r.sections[1].amendment_notes[0].startswith("Subs. by G.S.R. 328(E)")
    assert "Old text" not in r.sections[1].text


def test_gazette_rules_tolerances() -> None:
    # A-17: Gazette separators, running heads inside the body, mis-encoded quotes, '[ F. No.'.
    pages = [
        "G.S.R. 9(E).—In exercise of the powers conferred by section 101 of the Test Act, 2020.\n"
        "1. Short title and commencement. — (1) These rules may be called the Test Rules.\n"
        "(2) They come into force on publication.\n"
        "2. Scope. – These rules apply to e-commerce.\n",
        "8  THE GAZETTE OF INDIA : EXTRAORDINARY  [PART II—SEC. 3(i)]\n"
        "3.  Definitions. — (1) In these rules,—\n"
        "(a) \"Act\" means the Test Act, 2020;\n"
        "(b) ―PAN” means Permanent Account Number;\n"
        "(c) “inventory entity‖ means an entity which owns inventory;\n"
        "4. Duties of entities. ---\n"
        "(1) Every entity shall display:\n",
        "[भाग II—खण् ड 3(i)] भारत का राजपत्र : ऄसाधारण  9\n"
        "(क) its legal name;\n"
        "5. Inventory entities: -\n"
        "(1) Every inventory entity shall comply.\n"
        "[ F. No. J-10/3/2018-CPU]\nSOME OFFICER, Jt. Secy.\n",
    ]
    r = parse_statute(pages, "rule", 5)
    assert [s.title for s in r.sections] == [
        "Short title and commencement", "Scope", "Definitions", "Duties of entities", "Inventory entities",
    ]
    defs = [(c.label, c.term) for c in r.sections[2].clauses if c.kind == "definition"]
    assert defs == [("a", "Act"), ("b", "PAN"), ("c", "inventory entity")]
    joined = " ".join(s.text for s in r.sections)
    assert "GAZETTE" not in joined and "राजपत्र" not in joined and "F. No." not in joined
    assert "\n(क) its legal name;" in r.sections[3].text  # printed label kept, as its own item


def test_act_headings_stay_strict() -> None:
    # A-17 applies to rules only: in an Act, '. —' with spaces is not a heading (P-06 as written).
    pages = ["An Act to test.\n1. One.—Text.\n2. Two. — Text.\n"]
    r = parse_statute(pages, "act", 1)
    assert [s.number for s in r.sections] == ["1"]


# SC-9: amendment notifications (P-19)

from app.ingestion.legal.parser import apply_substitution, extract_substitutions, parse_amending  # noqa: E402

AMEND_PAGES = [
    "1. (1) संक्षिप्त नाम\n"  # Hindi part before the English enabling text
    "NOTIFICATION\nG.S.R. 9(E).—In exercise of the powers conferred by section 101 of the Test Act, 2020,\n"
    "the Central Government makes the following rules to amend the Test Rules, 2020, namely: -\n"
    "1. (1) These rules may be called the Test (Amendment) Rules, 2021.\n"
    "(2) They shall come into force on publication.\n"
    "2. In the Test Rules, 2020, in rule 4, for sub-rule (1), the following sub-rule\n"
    "shall be substituted, namely:--\n"
    "“(1) Every entity shall appoint a nodal officer.”\n"
    "[F. No. 1/2]\nOFFICER\nNote: The principal rules were published vide G.S.R. 1(E)."
]


def test_parse_amending_and_extract_substitution() -> None:
    a = parse_amending(AMEND_PAGES, 2)
    assert [s.number for s in a.sections] == ["1", "2"]
    assert a.enabling_text.startswith("In exercise of the powers conferred by section 101")
    (sub,) = extract_substitutions(a)
    assert (sub.principal_title, sub.rule, sub.sub) == ("Test Rules, 2020", "4", "1")
    assert sub.text == "(1) Every entity shall appoint a nodal officer."


@pytest.mark.parametrize(
    "instruction",
    [
        "2. In the Test Rules, 2020, in rule 4, after sub-rule (1), the following sub-rule shall be inserted, namely:-- “(1A) New.”",
        "2. In the Test Rules, 2020, in rule 4, sub-rule (1) shall be omitted.",
        "2. In the Test Rules, 2020, in rule 4, for sub-rule (1), the following sub-rule shall be substituted, namely:-- “(2) Wrong label.”",
        "2. In the Test Rules, 2020, (a) in rule 4, for sub-rule (1), the following sub-rule shall be substituted, namely: “(1) A.” (b) in rule 5, for sub-rule (2), the following sub-rule shall be substituted, namely: “(2) B.”",
    ],
    ids=["insertion", "omission", "label-mismatch", "two-instructions"],
)
def test_unsupported_amendments_fail_safely(instruction) -> None:
    pages = ["G.S.R. 9(E).—In exercise of the powers conferred by section 101, namely:\n"
             "1. (1) These rules may be called the Test (Amendment) Rules.\n" + instruction + "\n[F. No. 1]"]
    with pytest.raises(ParseError) as info:
        extract_substitutions(parse_amending(pages, 2))
    assert info.value.code == "PARSE_VALIDATION_FAILED"


def _principal():
    return parse_statute([
        "1. Short title.—(1) These rules may be called the Test Rules.\n"
        "2. Duties.—(1) An entity shall:\n(a) be a company; and\n(b) appoint a person.\n(2) Second duty.\n"
    ], "rule", 2)


def test_apply_subrule_substitution() -> None:
    p = _principal()
    previous = apply_substitution(p, "2", "1", "(1) Every entity shall appoint a nodal officer.")
    assert previous == "An entity shall:\n(a) be a company; and\n(b) appoint a person."
    sec = p.sections[1]
    assert sec.clauses[0].text == "Every entity shall appoint a nodal officer."
    assert sec.text == "(1) Every entity shall appoint a nodal officer.\n(2) Second duty."


def test_apply_whole_rule_substitution() -> None:
    p = _principal()
    apply_substitution(p, "2", None, "2. New duties.—(1) Only one duty now.")
    sec = p.sections[1]
    assert (sec.title, sec.text, [c.label for c in sec.clauses]) == ("New duties", "(1) Only one duty now.", ["1"])


@pytest.mark.parametrize("rule, sub", [("9", "1"), ("2", "7")])
def test_apply_substitution_missing_target_fails(rule, sub) -> None:
    with pytest.raises(ParseError) as info:
        apply_substitution(_principal(), rule, sub, f"({sub}) X.")
    assert info.value.code == "PARSE_VALIDATION_FAILED"
