"""vector_search.py: case search is scoped through scope.py and uses the SC-15 SEARCH query."""

from __future__ import annotations

import uuid

from app.services.retrieval import graph_queries
from app.services.retrieval.scope import CaseScope, ScopedGraph, require_scope_filter
from app.services.retrieval.vector_search import case_vector_search, legal_vector_search

VEC = [0.1] * 768


class RecordingNeo4j:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def read(self, query, params=None):
        self.calls.append((query, dict(params)))
        return []


def test_case_query_filters_inside_search_with_parameters() -> None:
    query = graph_queries.CASE_VECTOR_SEARCH
    require_scope_filter(query)
    search_block = query[query.index("SEARCH node IN"):query.index("SCORE AS score")]
    assert "node.case_id = $case_id AND node.user_id = $user_id" in search_block
    assert "LIMIT $k" in search_block and "queryNodes" not in query


async def test_case_search_passes_scope_and_k_only() -> None:
    neo = RecordingNeo4j()
    graph = ScopedGraph(neo, CaseScope(str(uuid.uuid4()), "local"))  # type: ignore[arg-type]
    await case_vector_search(graph, VEC, k=15)
    query, params = neo.calls[0]
    assert query == graph_queries.CASE_VECTOR_SEARCH
    assert params == {"query_vec": VEC, "k": 15, "case_id": graph.scope.case_id, "user_id": "local"}


async def test_legal_search_fetches_30() -> None:
    neo = RecordingNeo4j()
    await legal_vector_search(neo, VEC)  # type: ignore[arg-type]
    assert neo.calls == [(graph_queries.LEGAL_VECTOR_SEARCH, {"k": 30, "query_vec": VEC})]
