"""Shared fixtures, including the throwaway Neo4j database (06 section 1)."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from dotenv import load_dotenv

# Real values come from .env when present. Placeholders let unit tests import the app on a
# machine without secrets; tests that need a live service skip when it is unreachable.
load_dotenv(".env", override=False)
os.environ.setdefault("NEO4J_PASSWORD", "unit-test-placeholder")
os.environ.setdefault("GROQ_API_KEY", "unit-test-placeholder")

from app.config import get_settings  # noqa: E402
from app.services.retrieval import graph_queries  # noqa: E402
from app.services.retrieval.neo4j_service import Neo4jService  # noqa: E402


def _service(database: str) -> Neo4jService:
    s = get_settings()
    return Neo4jService(s.NEO4J_URI, s.NEO4J_USER, s.NEO4J_PASSWORD.get_secret_value(), database)


@pytest.fixture(scope="module")
def neo4j_test_db() -> Iterator[Neo4jService]:
    """A freshly created database per test module, dropped afterwards. Needs Neo4j
    Enterprise or Desktop (Community has a single user database); skips otherwise."""
    from neo4j.exceptions import ClientError

    system = _service("system")
    try:
        system.verify_connectivity_sync()
    except Exception as exc:
        system.close()
        pytest.skip(f"Neo4j is not reachable ({type(exc).__name__})")

    name = f"test-{uuid.uuid4().hex[:12]}"
    try:
        system.run_autocommit_sync(graph_queries.CREATE_DATABASE, {"name": name}, database="system")
    except ClientError as exc:
        system.close()
        pytest.skip(f"cannot create a throwaway database ({exc.code})")

    db = _service(name)
    try:
        yield db
    finally:
        db.close()
        system.run_autocommit_sync(graph_queries.DROP_DATABASE, {"name": name}, database="system")
        system.close()
