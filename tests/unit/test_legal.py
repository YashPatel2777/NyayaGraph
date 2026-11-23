"""GET /v1/legal/provisions/{provision_id} (3.4, A-45): the lookup and the route."""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.agents.nodes.graph_expander import ConceptIndex
from app.deps import get_concepts, get_neo4j
from app.errors import AppError, ErrorCode
from app.ingestion.legal.legal_processor import get_provision
from app.main import create_app
from app.services.retrieval import graph_queries

CPA = "Consumer Protection Act, 2019"


class Date:
    def __init__(self, d: date) -> None:
        self.d = d

    def to_native(self) -> date:
        return self.d


def _hyd(pid, number, label=None, act=CPA, kind="act"):
    return {"id": pid, "is_section": label is None, "label": label, "section_number": number,
            "act_short_title": act, "act_kind": kind}


ROWS = {
    "cl:cpa-2019:35:1": {
        "id": "cl:cpa-2019:35:1", "is_section": False, "label": "1",
        "text": "A complaint may be filed by the consumer to whom goods are sold, or a complainant.",
        "section_number": "35", "status": "in_force", "in_force_from": Date(date(2020, 7, 20)),
        "amendment_notes": ["Subs. by Act 33 of 2021"], "parent_id": "sec:cpa-2019:35",
        "act_short_title": CPA, "act_kind": "act",
        "refers_to": ["sec:cpa-2019:69"], "referenced_by": ["cl:cpa-2019:49:1"],
    },
    "sec:cp-jurisdiction-rules-2021:3": {
        "id": "sec:cp-jurisdiction-rules-2021:3", "is_section": True, "label": None,
        "text": "the District Commission shall have jurisdiction where the value does not exceed fifty lakh rupees.",
        "section_number": "3", "status": None, "in_force_from": None, "amendment_notes": None,
        "parent_id": None, "act_short_title": "Jurisdiction Rules, 2021", "act_kind": "rule",
        "refers_to": [], "referenced_by": [],
    },
}
HYDRATE = {
    "sec:cpa-2019:35": _hyd("sec:cpa-2019:35", "35"),
    "sec:cpa-2019:69": _hyd("sec:cpa-2019:69", "69"),
    "cl:cpa-2019:49:1": _hyd("cl:cpa-2019:49:1", "49", "1"),
}
CONCEPTS = ConceptIndex([
    {"id": "con:consumer", "name": "consumer", "clause_id": "cl:cpa-2019:2:7"},
    {"id": "con:complainant", "name": "complainant", "clause_id": "cl:cpa-2019:2:5"},
    {"id": "con:District Commission", "name": "District Commission", "clause_id": "cl:cpa-2019:2:15"},
])


class FakeNeo4j:
    async def read(self, query, params=None):
        if query == graph_queries.PROVISION_GET:
            return [ROWS[params["id"]]] if params["id"] in ROWS else []
        if query == graph_queries.LEGAL_PROVISIONS_HYDRATE:
            return [HYDRATE[i] for i in params["ids"] if i in HYDRATE]
        raise AssertionError("unexpected query")


async def test_clause_with_parent_references_and_definitions() -> None:
    p = await get_provision(FakeNeo4j(), "cl:cpa-2019:35:1", CONCEPTS.terms_in)  # type: ignore[arg-type]
    assert p["citation"] == f"Section 35(1), {CPA}"
    assert p["parent"] == {"id": "sec:cpa-2019:35", "citation": f"Section 35, {CPA}"}
    assert p["refers_to"] == [{"id": "sec:cpa-2019:69", "citation": f"Section 69, {CPA}"}]
    assert p["referenced_by"] == [{"id": "cl:cpa-2019:49:1", "citation": f"Section 49(1), {CPA}"}]
    assert p["in_force_from"] == date(2020, 7, 20) and p["amendment_notes"] == ["Subs. by Act 33 of 2021"]
    # Whole words only, in order of appearance, with the definition Clause id (A-45).
    assert p["definitions"] == [{"term": "consumer", "id": "cl:cpa-2019:2:7"},
                                {"term": "complainant", "id": "cl:cpa-2019:2:5"}]


async def test_rule_section_has_no_parent_and_defaults() -> None:
    p = await get_provision(FakeNeo4j(), "sec:cp-jurisdiction-rules-2021:3", CONCEPTS.terms_in)  # type: ignore[arg-type]
    assert p["citation"] == "Rule 3, Jurisdiction Rules, 2021"
    assert p["parent"] is None and p["status"] == "in_force" and p["amendment_notes"] == []
    assert p["definitions"] == [{"term": "District Commission", "id": "cl:cpa-2019:2:15"}]


async def test_own_definition_is_not_listed() -> None:
    assert CONCEPTS.terms_in("a consumer means", {"cl:cpa-2019:2:7"}) == []


@pytest.mark.parametrize("pid", ["sec:none:1", "cl:cpa-2019:2:47-p1"])
async def test_unknown_or_split_part_is_404(pid: str) -> None:
    with pytest.raises(AppError) as exc:
        await get_provision(FakeNeo4j(), pid, CONCEPTS.terms_in)  # type: ignore[arg-type]
    assert exc.value.code == ErrorCode.PROVISION_NOT_FOUND


def test_route_returns_the_3_4_fields() -> None:
    app = create_app()
    app.dependency_overrides[get_neo4j] = lambda: FakeNeo4j()
    app.dependency_overrides[get_concepts] = lambda: CONCEPTS
    resp = TestClient(app).get("/v1/legal/provisions/cl:cpa-2019:35:1")
    assert resp.status_code == 200
    assert set(resp.json()) == {"provision_id", "citation", "text", "status", "in_force_from", "amendment_notes",
                                "parent", "refers_to", "referenced_by", "definitions"}
    missing = TestClient(app).get("/v1/legal/provisions/sec:none:1")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "PROVISION_NOT_FOUND"
