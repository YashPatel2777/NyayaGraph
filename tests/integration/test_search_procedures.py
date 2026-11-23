"""O-1 guard: the 4.2 search queries still work on this Neo4j server.

Neo4j 2026.10.2 reports db.index.vector.queryNodes and db.index.fulltext.queryNodes as
deprecated ('replaced by SEARCH'). Legal vector search and both full-text searches keep them
(O-1 in docs/README.md) and the app driver silences the notices. Case vector search uses the
SEARCH clause with an in-index filter (SC-15). If a Neo4j upgrade breaks any of them, this
test fails with that message instead of search breaking quietly.
"""

from __future__ import annotations

import pytest
from neo4j.exceptions import ClientError

from app.services.retrieval import graph_queries
from app.services.retrieval.neo4j_service import Neo4jService
from scripts.init_neo4j import init_schema

pytestmark = pytest.mark.neo4j

CASE = "3f2a1b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b"
VEC = [0.01] * 767 + [0.02]
REMOVED = ("A 4.2 search procedure failed on this Neo4j version; it may have been removed "
           "(O-1). Do not upgrade Neo4j until graph_queries.py moves to SEARCH.")


@pytest.fixture(scope="module")
def db(neo4j_test_db: Neo4jService) -> Neo4jService:
    init_schema(neo4j_test_db)
    neo4j_test_db.write_sync(
        "CREATE (:CaseChunk {id: 'chk:t', case_id: $c, user_id: 'local', doc_id: 'd', text: 'seller refused refund', "
        "embedding: $v}), (:LegalUnit {id: 'sec:t:1', search_text: 'refund', embedding: $v})",
        {"c": CASE, "v": VEC},
    )
    neo4j_test_db.write_sync("CALL db.awaitIndexes(60)")
    return neo4j_test_db


@pytest.mark.parametrize(
    ("query", "params"),
    [
        (graph_queries.LEGAL_VECTOR_SEARCH, {"k": 1, "query_vec": VEC}),
        (graph_queries.CASE_VECTOR_SEARCH,
         {"k": 1, "query_vec": VEC, "case_id": CASE, "user_id": "local"}),
        (graph_queries.CASE_FULLTEXT_SEARCH,
         {"lucene": '"refund"', "fetch_k": 50, "k": 1, "case_id": CASE, "user_id": "local"}),
    ],
    ids=["legal-vector", "case-vector", "case-fulltext"],
)
def test_4_2_search_procedures_still_work(db: Neo4jService, query: str, params: dict) -> None:
    try:
        rows = db.read_sync(query, params)
    except ClientError as exc:
        pytest.fail(f"{REMOVED} Server said: {exc.code}")
    assert len(rows) == 1


def test_app_driver_silences_deprecation_notices(monkeypatch: pytest.MonkeyPatch) -> None:
    # The notices log the full query text on every search (R-04), so they stay off.
    from neo4j import NotificationDisabledClassification

    from app.services.retrieval import neo4j_service

    seen: dict = {}

    def fake_driver(uri, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(neo4j_service.GraphDatabase, "driver", fake_driver)
    Neo4jService("bolt://x", "u", "p", "neo4j")
    assert NotificationDisabledClassification.DEPRECATION in seen["notifications_disabled_classifications"]
