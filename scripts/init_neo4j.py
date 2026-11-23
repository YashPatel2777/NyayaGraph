"""Create the 2.2 schema: constraints, vector, full-text and range indexes, and the Meta node.

Idempotent: every statement is IF NOT EXISTS or MERGE ... ON CREATE, so a second run makes
no change. One migration runs first: a case vector index without the SC-15 filter
properties is dropped and created again (2.2). The script prints the change counters and compares the schema inventory before
and after the run to show it.

Run: python -m scripts.init_neo4j [--database NAME]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.schemas.graph_schema import CASE_CHUNK_EMBEDDING_INDEX, index_missing_properties
from app.services.retrieval import graph_queries
from app.services.retrieval.neo4j_service import Neo4jService

AWAIT_INDEXES_TIMEOUT_S = 300


@dataclass
class InitReport:
    counters: dict[str, int] = field(default_factory=dict)
    before: dict[str, list[dict]] = field(default_factory=dict)
    after: dict[str, list[dict]] = field(default_factory=dict)

    @property
    def changes(self) -> int:
        return sum(self.counters.values())

    @property
    def unchanged(self) -> bool:
        return self.changes == 0 and self.before == self.after


def _inventory(neo4j: Neo4jService) -> dict[str, list[dict]]:
    return {
        "constraints": neo4j.read_sync(graph_queries.CONSTRAINT_INVENTORY),
        "indexes": neo4j.read_sync(graph_queries.SCHEMA_INVENTORY),
        "meta": neo4j.read_sync(graph_queries.META_NODES),
    }


def _drop_outdated_case_vector_index(neo4j: Neo4jService, report: InitReport) -> None:
    """SC-15: a case vector index created before SC-15 lacks the scope filter properties.
    Drop it so the 2.2 DDL below creates it again; the vectors stay on the nodes."""
    rows = neo4j.read_sync(graph_queries.INDEX_STATES, {"names": [CASE_CHUNK_EMBEDDING_INDEX]})
    if rows and index_missing_properties(CASE_CHUNK_EMBEDDING_INDEX, rows[0].get("properties")):
        for key, value in neo4j.run_autocommit_sync(graph_queries.DROP_CASE_CHUNK_EMBEDDING_INDEX).items():
            report.counters[key] = report.counters.get(key, 0) + value


def init_schema(neo4j: Neo4jService) -> InitReport:
    report = InitReport(before=_inventory(neo4j))
    _drop_outdated_case_vector_index(neo4j, report)
    for statement in graph_queries.SCHEMA_DDL:
        for key, value in neo4j.run_autocommit_sync(statement).items():
            report.counters[key] = report.counters.get(key, 0) + value
    neo4j.run_autocommit_sync(graph_queries.AWAIT_INDEXES, {"timeout_s": AWAIT_INDEXES_TIMEOUT_S})
    report.after = _inventory(neo4j)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the Neo4j schema from spec 2.2.")
    parser.add_argument("--database", help="Target database (default: NEO4J_DATABASE)")
    args = parser.parse_args(argv)

    settings = get_settings()
    database = args.database or settings.NEO4J_DATABASE
    neo4j = Neo4jService(
        settings.NEO4J_URI,
        settings.NEO4J_USER,
        settings.NEO4J_PASSWORD.get_secret_value(),
        database,
    )
    start = time.perf_counter()
    try:
        neo4j.verify_connectivity_sync()
        report = init_schema(neo4j)
    finally:
        neo4j.close()

    ms = (time.perf_counter() - start) * 1000
    print(f"database={database} statements={len(graph_queries.SCHEMA_DDL)} ms={ms:.0f}")
    for key in sorted(report.counters):
        print(f"  {key:<20} {report.counters[key]}")
    print(
        f"constraints={len(report.after['constraints'])} "
        f"indexes={len(report.after['indexes'])} meta_nodes={len(report.after['meta'])}"
    )
    print("result: no change" if report.unchanged else f"result: {report.changes} change(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
