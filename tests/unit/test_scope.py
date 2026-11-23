"""scope.py always passes case_id and user_id as parameters (Phase 1 exit, R-01, R-02)."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.services.retrieval.scope import (
    CaseScope,
    ScopedGraph,
    require_scope_filter,
    scoped_params,
)

CASE = str(uuid.uuid4())
OTHER_CASE = str(uuid.uuid4())
USER = "local"

SCOPED_QUERY = (
    "MATCH (c:CaseChunk) WHERE c.case_id = $case_id AND c.user_id = $user_id "
    "AND c.order >= $min_order RETURN c.id AS id"
)


class RecordingNeo4j:
    """Stands in for Neo4jService and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def read(self, query, params=None):
        self.calls.append(("read", query, dict(params)))
        return []

    async def write(self, query, params=None):
        self.calls.append(("write", query, dict(params)))
        return []

    async def write_batches(self, query, rows, params=None, batch_size=200):
        self.calls.append(("write_batches", query, dict(params)))
        return len(rows)

    def read_sync(self, query, params=None):
        self.calls.append(("read_sync", query, dict(params)))
        return []

    def write_sync(self, query, params=None):
        self.calls.append(("write_sync", query, dict(params)))
        return []

    def write_batches_sync(self, query, rows, params=None, batch_size=200):
        self.calls.append(("write_batches_sync", query, dict(params)))
        return len(rows)


@pytest.fixture
def graph() -> tuple[ScopedGraph, RecordingNeo4j]:
    rec = RecordingNeo4j()
    return ScopedGraph(rec, CaseScope(CASE, USER)), rec  # type: ignore[arg-type]


async def test_every_method_passes_case_id_and_user_id(graph) -> None:
    scoped, rec = graph
    await scoped.read(SCOPED_QUERY, {"min_order": 0})
    await scoped.write(SCOPED_QUERY, {"min_order": 0})
    await scoped.write_batches(SCOPED_QUERY, [{"x": 1}], {"min_order": 0})
    scoped.read_sync(SCOPED_QUERY, {"min_order": 0})
    scoped.write_sync(SCOPED_QUERY, {"min_order": 0})
    scoped.write_batches_sync(SCOPED_QUERY, [{"x": 1}], {"min_order": 0})

    assert len(rec.calls) == 6
    for method, query, params in rec.calls:
        assert params["case_id"] == CASE, method
        assert params["user_id"] == USER, method
        assert params["min_order"] == 0, method
        # Values travel only as parameters; the query text is passed through untouched.
        assert query == SCOPED_QUERY
        assert CASE not in query and USER not in query


async def test_scope_is_added_when_caller_passes_no_params(graph) -> None:
    scoped, rec = graph
    await scoped.read(SCOPED_QUERY)
    assert rec.calls[0][2] == {"case_id": CASE, "user_id": USER}


@pytest.mark.parametrize("key", ["case_id", "user_id"])
async def test_caller_cannot_override_scope(graph, key) -> None:
    scoped, rec = graph
    with pytest.raises(ValueError, match="may not be supplied"):
        await scoped.read(SCOPED_QUERY, {key: OTHER_CASE if key == "case_id" else "attacker"})
    assert rec.calls == []


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (c:CaseChunk) RETURN c",
        "MATCH (c:CaseChunk) WHERE c.case_id = $case_id RETURN c",
        "MATCH (c:CaseChunk) WHERE c.user_id = $user_id RETURN c",
        "MATCH (c:CaseChunk) WHERE c.case_id = $case_identifier AND c.user_id = $user_id RETURN c",
    ],
)
async def test_query_without_both_scope_filters_is_refused(graph, query) -> None:
    scoped, rec = graph
    with pytest.raises(ValueError, match="must reference both"):
        await scoped.read(query)
    assert rec.calls == []


def test_scoped_params_returns_new_dict() -> None:
    given = {"k": 5}
    out = scoped_params(CaseScope(CASE, USER), given)
    assert out == {"k": 5, "case_id": CASE, "user_id": USER}
    assert given == {"k": 5}


def test_require_scope_filter_accepts_both() -> None:
    require_scope_filter(SCOPED_QUERY)


def test_case_scope_normalises_and_validates_case_id() -> None:
    assert CaseScope(CASE.upper(), USER).case_id == CASE
    assert CaseScope(uuid.UUID(CASE), USER).case_id == CASE  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CaseScope("not-a-uuid", USER)
    with pytest.raises(ValueError):
        CaseScope(str(uuid.uuid1()), USER)  # not v4
    with pytest.raises(ValueError):
        CaseScope("case:" + CASE, USER)  # node ids are not accepted; the property is bare
    with pytest.raises(ValueError):
        CaseScope(CASE, "")


def test_case_scope_is_immutable() -> None:
    scope = CaseScope(CASE, USER)
    with pytest.raises(AttributeError):
        scope.user_id = "someone-else"  # type: ignore[misc]


def test_user_id_comes_from_deps(monkeypatch) -> None:
    from app.config import get_settings
    from app.deps import get_current_user

    monkeypatch.setenv("DEFAULT_USER_ID", "operator-7")
    get_settings.cache_clear()
    try:
        assert get_current_user() == "operator-7"
        assert CaseScope(CASE, get_current_user()).user_id == "operator-7"
    finally:
        get_settings.cache_clear()
