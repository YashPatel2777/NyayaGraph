"""Lucene escaping (R-03) and the R-09 over-fetch loop in fulltext_search.py."""

from __future__ import annotations

import uuid

import pytest

from app.services.retrieval import graph_queries
from app.services.retrieval.fulltext_search import (
    FETCH_K_MAX,
    build_lucene_query,
    case_fulltext_search,
    escape_lucene,
    legal_fulltext_search,
    overfetch,
)
from app.services.retrieval.scope import CaseScope, ScopedGraph

BS = "\\"


@pytest.mark.parametrize("ch", list('+-&|!(){}[]^"~*?:/') + [BS])
def test_every_r03_character_is_escaped(ch: str) -> None:
    assert escape_lucene(f"a{ch}b") == f"a{BS}{ch}b"


def test_two_character_operators_are_escaped() -> None:
    assert escape_lucene("a && b || c") == r"a \&\& b \|\| c"


def test_plain_text_is_unchanged() -> None:
    assert escape_lucene("refund of 42,000 rupees") == "refund of 42,000 rupees"


def test_each_term_is_quoted_and_escaped() -> None:
    assert build_lucene_query('section 2(1) "consumer"') == r'"section" "2\(1\)" "\"consumer\""'


def test_injection_attempt_stays_inside_quotes() -> None:
    query = build_lucene_query('x" OR *:* OR "y')
    assert query == r'"x\"" "OR" "\*\:\*" "\"y"'  # the repeated OR is kept once
    # No unescaped quote other than the ones that open and close each term.
    for term in query.split(" "):
        assert term.startswith('"') and term.endswith('"')
        assert '"' not in term[1:-1].replace(BS + '"', "")


def test_repeated_terms_are_kept_once_case_insensitively() -> None:
    assert build_lucene_query("Refund refund REFUND seller") == '"Refund" "seller"'


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_nothing_searchable_returns_none(text: str) -> None:
    assert build_lucene_query(text) is None


async def test_overfetch_stops_when_k_rows_survive() -> None:
    asked: list[int] = []

    async def fetch(fetch_k: int) -> list[dict]:
        asked.append(fetch_k)
        return [{"id": str(i)} for i in range(fetch_k // 25)]  # 2, 4, 8 ... survive

    rows, rounds = await overfetch(fetch, k=5)
    assert asked == [50, 100, 200]
    assert rounds == 3 and len(rows) == 5


async def test_overfetch_doubles_to_800_in_at_most_5_rounds() -> None:
    asked: list[int] = []

    async def fetch(fetch_k: int) -> list[dict]:
        asked.append(fetch_k)
        return []

    rows, rounds = await overfetch(fetch, k=15)
    assert asked == [50, 100, 200, 400, 800]
    assert asked[-1] == FETCH_K_MAX and rounds == 5 and rows == []


class RecordingNeo4j:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.rows = rows or []

    async def read(self, query, params=None):
        self.calls.append((query, dict(params)))
        return self.rows


async def test_case_search_is_scoped_and_parameterised() -> None:
    neo = RecordingNeo4j([{"id": "chk:a:0000", "score": 1.0}])
    graph = ScopedGraph(neo, CaseScope(str(uuid.uuid4()), "local"))  # type: ignore[arg-type]
    rows = await case_fulltext_search(graph, "refund (urgent)", k=1)
    assert rows == [{"id": "chk:a:0000", "score": 1.0}]
    query, params = neo.calls[0]
    assert query == graph_queries.CASE_FULLTEXT_SEARCH
    assert params["lucene"] == r'"refund" "\(urgent\)"'
    assert params["case_id"] == graph.scope.case_id and params["user_id"] == "local"
    assert params["fetch_k"] == 50 and params["k"] == 1


async def test_case_search_with_no_terms_skips_the_query() -> None:
    neo = RecordingNeo4j()
    graph = ScopedGraph(neo, CaseScope(str(uuid.uuid4()), "local"))  # type: ignore[arg-type]
    assert await case_fulltext_search(graph, "  ", k=5) == []
    assert neo.calls == []


async def test_legal_search_fetches_30_once() -> None:
    neo = RecordingNeo4j()
    await legal_fulltext_search(neo, "limitation period")  # type: ignore[arg-type]
    assert neo.calls == [(graph_queries.LEGAL_FULLTEXT_SEARCH, {"lucene": '"limitation" "period"', "k": 30})]
