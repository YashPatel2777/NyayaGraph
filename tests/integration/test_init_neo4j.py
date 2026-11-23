"""init_neo4j runs twice with no error and no change (Phase 1 exit, 2.2).

Runs against a throwaway database (tests/conftest.py) and skips if Neo4j is not reachable.
"""

from __future__ import annotations

import pytest

from app.schemas.graph_schema import REQUIRED_SEARCH_INDEXES
from app.services.retrieval import graph_queries
from scripts.init_neo4j import init_schema

pytestmark = pytest.mark.neo4j

EXPECTED_CONSTRAINTS = {
    "act_id", "chapter_id", "section_id", "clause_id", "legalunit_id", "concept_id",
    "judgment_id", "case_id", "casedoc_id", "casechunk_id", "party_id", "event_id",
    "evidence_id", "claim_id", "job_id", "meta_key",
}
EXPECTED_INDEXES = {
    "legal_unit_embedding", "case_chunk_embedding", "legal_text_ft", "case_text_ft",
    "casechunk_scope", "case_owner", "job_owner",
}


def test_init_twice_no_error_no_change(neo4j_test_db) -> None:
    first = init_schema(neo4j_test_db)
    assert first.counters["constraints_added"] == 16
    assert first.counters["indexes_added"] == 7
    assert first.counters["nodes_created"] == 1

    second = init_schema(neo4j_test_db)
    assert second.changes == 0, second.counters
    assert second.before == second.after
    assert second.after == first.after
    assert second.unchanged


def test_schema_matches_spec(neo4j_test_db) -> None:
    init_schema(neo4j_test_db)
    constraints = {r["name"] for r in neo4j_test_db.read_sync(graph_queries.CONSTRAINT_INVENTORY)}
    assert EXPECTED_CONSTRAINTS <= constraints

    indexes = {r["name"]: r for r in neo4j_test_db.read_sync(graph_queries.SCHEMA_INVENTORY)}
    assert EXPECTED_INDEXES <= indexes.keys()
    for name in ("legal_unit_embedding", "case_chunk_embedding"):
        config = indexes[name]["options"]["indexConfig"]
        assert config["vector.dimensions"] == 768
        assert config["vector.similarity_function"].lower() == "cosine"

    states = neo4j_test_db.read_sync(
        graph_queries.INDEX_STATES, {"names": list(REQUIRED_SEARCH_INDEXES)}
    )
    assert {r["name"]: r["state"] for r in states} == {n: "ONLINE" for n in REQUIRED_SEARCH_INDEXES}

    meta = neo4j_test_db.read_sync(graph_queries.GET_META, {"key": "embedding"})
    assert meta == [{
        "model": "nomic-embed-text", "dim": 768, "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ", "schema_version": 1,
    }]


def test_pre_sc15_case_vector_index_is_rebuilt(neo4j_test_db) -> None:
    # SC-15 (2.2): an index created by the old DDL lacks the scope filter properties.
    neo4j_test_db.write_sync(
        "CREATE (:CaseChunk {id: 'chk:old', case_id: 'c', user_id: 'local', embedding: $v})",
        {"v": [0.01] * 768},
    )
    neo4j_test_db.run_autocommit_sync(graph_queries.DROP_CASE_CHUNK_EMBEDDING_INDEX)  # module-scoped DB
    neo4j_test_db.run_autocommit_sync(
        "CREATE VECTOR INDEX case_chunk_embedding IF NOT EXISTS FOR (n:CaseChunk) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}"
    )
    report = init_schema(neo4j_test_db)
    assert report.counters["indexes_removed"] == 1
    rows = neo4j_test_db.read_sync(graph_queries.INDEX_STATES, {"names": ["case_chunk_embedding"]})
    assert rows[0]["state"] == "ONLINE"
    assert set(rows[0]["properties"]) == {"embedding", "case_id", "user_id"}
    # The vector stayed on the node and the next run changes nothing.
    assert neo4j_test_db.read_sync("MATCH (c:CaseChunk {id: 'chk:old'}) RETURN size(c.embedding) AS n") == [{"n": 768}]
    assert init_schema(neo4j_test_db).unchanged
