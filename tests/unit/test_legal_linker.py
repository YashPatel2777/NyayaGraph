"""B9 regex citations, Act resolution and R-26 (A-26; 07 T3.9)."""

from __future__ import annotations

import pytest

from app.ingestion.case.legal_linker import (
    ActIndex,
    citation_targets,
    find_citations,
    provision_of,
)

ACT_ROWS = [
    {"id": "act:cpa-2019", "kind": "act", "short_title": "Consumer Protection Act, 2019", "title": "x", "external": False},
    {"id": "act:cp-e-commerce-rules-2020", "kind": "rule", "short_title": "Consumer Protection (E-Commerce) Rules, 2020", "title": "x", "external": False},
    {"id": "act:indian-penal-code", "kind": "act", "short_title": "Indian Penal Code", "title": "x", "external": True},
]
ACTS = ActIndex.from_rows(ACT_ROWS)


def refs(text: str) -> list[tuple[str, tuple[str, ...], str | None]]:
    out = []
    for c in find_citations(text):
        slug = ACTS.resolve(c)
        out.append((c.section, c.labels, slug))
    return out


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("as per Section 2(1)(d) the buyer is a consumer", [("2", ("1", "d"), "cpa-2019")]),
        ("filed under section 35(1)", [("35", ("1",), "cpa-2019")]),
        ("see Sec. 38 and S. 39", [("38", (), "cpa-2019"), ("39", (), "cpa-2019")]),
        ("u/s 35 of the Act", [("35", (), "cpa-2019")]),
        ("sections 35 and 36", [("35", (), "cpa-2019"), ("36", (), "cpa-2019")]),
        ("Sections 34, 35 and 36 of the Consumer Protection Act, 2019",
         [("34", (), "cpa-2019"), ("35", (), "cpa-2019"), ("36", (), "cpa-2019")]),
        ("Section 2 (7) of this Act", [("2", ("7",), "cpa-2019")]),
        ("section 69 CPA", [("69", (), "cpa-2019")]),
    ],
)
def test_citation_forms(text: str, expected) -> None:
    assert refs(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "cheating under Section 420 IPC",
        "Section 420 of the Indian Penal Code",
        "section 195 of the Code of Criminal Procedure, 1973",
        "Section 13 Contract Act",
        "section 12 of the Consumer Protection Act, 1986",
        "Section 2(1)(d) of the CPA, 1986",
        "section 103 of the Bharatiya Nyaya Sanhita",
        "Section 318 BNS",
    ],
)
def test_other_acts_and_other_years_never_link_r26(text: str) -> None:
    assert [slug for *_, slug in refs(text)] == [None]


def test_not_citations() -> None:
    assert find_citations("Paid Rs. 38 and Rs. 18,499 on 2 Jan; the sec of state") == []
    # A list is read only after the plural keyword.
    assert refs("under section 35 and 2 days later") == [("35", (), "cpa-2019")]


def test_text_span_keeps_the_act_words() -> None:
    [c] = find_citations("He relied on section 195 of the Code of Criminal Procedure, 1973 today")
    assert c.text_span == "section 195 of the Code of Criminal Procedure, 1973"
    assert c.act_name == "code of criminal procedure" and c.act_year == 1973


def test_principal_act_needs_exactly_one_internal_act() -> None:
    rows = ACT_ROWS + [{"id": "act:other-2020", "kind": "act", "short_title": "Other Act, 2020",
                        "title": "x", "external": False}]
    two = ActIndex.from_rows(rows)
    [c] = find_citations("section 35")
    assert two.resolve(c) is None
    [named] = find_citations("section 35 of the Other Act")
    assert two.resolve(named) == "other-2020"


def test_targets() -> None:
    [one] = find_citations("section 35(1)")
    assert citation_targets(one, "cpa-2019") == ["cl:cpa-2019:35:1", "sec:cpa-2019:35"]
    [nested] = find_citations("Section 2(1)(d)")
    assert citation_targets(nested, "cpa-2019") == ["sec:cpa-2019:2"]  # A-26
    [plain] = find_citations("Section 107A")
    assert citation_targets(plain, "cpa-2019") == ["sec:cpa-2019:107A"]


def test_part_maps_to_its_provision() -> None:
    assert provision_of("cl:cp-e-commerce-rules-2020:5:3-p2") == "cl:cp-e-commerce-rules-2020:5:3"
    assert provision_of("sec:cpa-2019:38-p1") == "sec:cpa-2019:38"
    assert provision_of("sec:cpa-2019:69") == "sec:cpa-2019:69"
