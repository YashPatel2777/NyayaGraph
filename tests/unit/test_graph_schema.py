"""Id builders and slug (2.1)."""

from __future__ import annotations

import hashlib
import uuid

import pytest

from app.schemas import graph_schema as gs

U = "9f1c2a7e-5b34-4d0a-8a6e-0c1d2e3f4a5b"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Consumer", "consumer"),
        ("cpa-2019", "cpa-2019"),
        ("Indian Penal Code", "indian-penal-code"),
        ("Code of Criminal Procedure, 1973", "code-of-criminal-procedure-1973"),  # P-11
        ("  unfair   trade -- practice!! ", "unfair-trade-practice"),
        ("33 of 2021", "33-of-2021"),
        ("product seller", "product-seller"),
    ],
)
def test_slug(raw, expected) -> None:
    assert gs.slug(raw) == expected


def test_slug_rejects_empty() -> None:
    with pytest.raises(ValueError):
        gs.slug(" -- ")


def test_legal_ids_match_spec_examples() -> None:
    assert gs.act_id("cpa-2019") == "act:cpa-2019"
    assert gs.act_id("Indian Penal Code") == "act:indian-penal-code"
    assert gs.chapter_id("cpa-2019", "IV") == "ch:cpa-2019:IV"
    assert gs.section_id("cpa-2019", "38") == "sec:cpa-2019:38"
    assert gs.section_id("cpa-2019", "55A") == "sec:cpa-2019:55A"
    assert gs.clause_id("cpa-2019", "2", "7") == "cl:cpa-2019:2:7"
    assert gs.clause_id("cpa-2019", "38", "2") == "cl:cpa-2019:38:2"
    assert gs.clause_id("cpa-2019", "2", "1A") == "cl:cpa-2019:2:1A"
    assert gs.clause_id("cpa-2019", "38", "2-p1") == "cl:cpa-2019:38:2-p1"
    assert gs.concept_id("consumer") == "con:consumer"
    assert gs.judgment_id("XYZ v. ABC 2020") == "jdg:xyz-v-abc-2020"
    assert gs.judgment_chunk_id("xyz-v-abc-2020", 3) == "jch:xyz-v-abc-2020:3"


@pytest.mark.parametrize("roman", ["iv", "4", "IV ", ""])
def test_chapter_requires_roman(roman) -> None:
    with pytest.raises(ValueError):
        gs.chapter_id("cpa-2019", roman)


@pytest.mark.parametrize("number", ["1234", "55a", "A5", ""])
def test_section_number_validated(number) -> None:
    with pytest.raises(ValueError):
        gs.section_id("cpa-2019", number)


def test_case_ids() -> None:
    assert gs.case_node_id(U) == f"case:{U}"
    assert gs.document_node_id(U) == f"doc:{U}"
    assert gs.job_node_id(U) == f"job:{U}"
    assert gs.chunk_id(U, 7) == f"chk:{U}:0007"
    assert gs.chunk_id(U, 9999) == f"chk:{U}:9999"
    with pytest.raises(ValueError):
        gs.chunk_id(U, 10000)
    with pytest.raises(ValueError):
        gs.case_node_id(str(uuid.uuid1()))


def test_entity_id_is_deterministic_and_normalised() -> None:
    a = gs.entity_id("pty", U, "Acme  Electronics")
    b = gs.entity_id("pty", U, "  acme electronics ")
    assert a == b
    digest = hashlib.sha1(b"acme electronics").hexdigest()[:12]
    assert a == f"pty:{U}:{digest}"
    assert gs.entity_id("clm", U, "x").startswith(f"clm:{U}:")
    with pytest.raises(ValueError):
        gs.entity_id("foo", U, "x")
    with pytest.raises(ValueError):
        gs.entity_id("evt", U, "   ")


def test_bare_uuid_round_trip() -> None:
    assert gs.bare_uuid(gs.case_node_id(U), "case") == U
    with pytest.raises(ValueError):
        gs.bare_uuid(gs.case_node_id(U), "doc")


def test_new_uuid4_is_v4() -> None:
    assert uuid.UUID(gs.new_uuid4()).version == 4


def test_required_indexes() -> None:
    assert set(gs.REQUIRED_SEARCH_INDEXES) == {
        "legal_unit_embedding", "case_chunk_embedding", "legal_text_ft", "case_text_ft",
    }


def test_case_vector_index_must_declare_scope_properties() -> None:
    # SC-15 (2.2)
    assert gs.index_missing_properties("case_chunk_embedding", ["embedding"]) == ["case_id", "user_id"]
    assert gs.index_missing_properties("case_chunk_embedding", ["embedding", "case_id", "user_id"]) == []
    assert gs.index_missing_properties("case_chunk_embedding", None) == ["case_id", "embedding", "user_id"]
    assert gs.index_missing_properties("legal_unit_embedding", ["embedding"]) == []
