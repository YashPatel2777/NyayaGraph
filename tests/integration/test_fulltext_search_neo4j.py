"""Full-text search against a real Neo4j: escaped input parses (R-03), scope holds with
overlapping content across users (R-01, Phase 4 isolation), and R-09 finds a sparse case."""

from __future__ import annotations

import uuid

import pytest

from app.services.retrieval.fulltext_search import case_fulltext_search, legal_fulltext_search
from app.services.retrieval.neo4j_service import Neo4jService
from app.services.retrieval.scope import CaseScope, ScopedGraph
from scripts.init_neo4j import init_schema

pytestmark = pytest.mark.neo4j

CASE_A = str(uuid.uuid4())  # user local, 3 chunks
CASE_B = str(uuid.uuid4())  # user other, same texts
CASE_SPARSE = str(uuid.uuid4())  # user local, 2 chunks among 600 other-user chunks
TEXTS = [
    "The seller refused a refund for the defective refrigerator.",
    "Invoice 2025/03 shows 42,000 rupees paid (by UPI) on 2 March.",
    "The compressor failed on 28 March; the service centre did not respond.",
]


@pytest.fixture(scope="module")
def db(neo4j_test_db: Neo4jService) -> Neo4jService:
    init_schema(neo4j_test_db)
    rows = []
    for i, text in enumerate(TEXTS):
        rows.append({"id": f"chk:a:{i}", "case_id": CASE_A, "user_id": "local", "text": text})
        rows.append({"id": f"chk:b:{i}", "case_id": CASE_B, "user_id": "other", "text": text})
    for i in range(600):
        rows.append({"id": f"chk:n:{i}", "case_id": str(uuid.uuid4()), "user_id": "other",
                     "text": f"The seller refused a refund, complaint number {i}."})
    rows.append({"id": "chk:s:0", "case_id": CASE_SPARSE, "user_id": "local",
                 "text": "Seller refused refund."})
    rows.append({"id": "chk:s:1", "case_id": CASE_SPARSE, "user_id": "local",
                 "text": "A refund was refused by the seller again."})
    neo4j_test_db.write_sync(
        "UNWIND $rows AS r CREATE (c:CaseChunk {id: r.id, case_id: r.case_id, user_id: r.user_id, "
        "text: r.text, doc_id: 'd'})",
        {"rows": rows},
    )
    neo4j_test_db.write_sync(
        "CREATE (:LegalUnit {id: 'sec:t:69', search_text: 'complaint shall not be admitted unless filed within two years'})"
    )
    neo4j_test_db.write_sync("CALL db.awaitIndexes(60)")
    return neo4j_test_db


def _graph(db: Neo4jService, case_id: str, user_id: str) -> ScopedGraph:
    return ScopedGraph(db, CaseScope(case_id, user_id))


async def test_user_b_never_sees_user_a_chunks(db: Neo4jService) -> None:
    rows = await case_fulltext_search(_graph(db, CASE_B, "other"), "seller refund compressor invoice", k=10)
    assert {r["id"] for r in rows} == {"chk:b:0", "chk:b:1", "chk:b:2"}


async def test_right_case_wrong_user_returns_nothing(db: Neo4jService) -> None:
    assert await case_fulltext_search(_graph(db, CASE_A, "other"), "seller refund", k=5) == []


@pytest.mark.parametrize(
    "text",
    ['x" OR *:* OR "y', "2025/03 (by UPI)", "a && b || !c", "refund~2 ^3 [1 TO 9] {a} field:value \\ ?*", '"', ":"],
)
async def test_hostile_text_parses_and_stays_in_scope(db: Neo4jService, text: str) -> None:
    rows = await case_fulltext_search(_graph(db, CASE_A, "local"), text, k=5)
    assert all(r["id"].startswith("chk:a:") for r in rows)


async def test_escaped_terms_still_match(db: Neo4jService) -> None:
    rows = await case_fulltext_search(_graph(db, CASE_A, "local"), "(by UPI)", k=5)
    assert [r["id"] for r in rows] == ["chk:a:1"]


async def test_overfetch_reaches_a_sparse_case(db: Neo4jService) -> None:
    # 600 other-user chunks match 'seller refused refund' too. How many R-09 rounds this
    # needs depends on BM25 ranking; the loop itself is tested in tests/unit.
    rows = await case_fulltext_search(_graph(db, CASE_SPARSE, "local"), "seller refused refund", k=2)
    assert {r["id"] for r in rows} == {"chk:s:0", "chk:s:1"}


async def test_legal_fulltext(db: Neo4jService) -> None:
    rows = await legal_fulltext_search(db, "complaint within two years?")
    assert rows[0]["id"] == "sec:t:69"
