"""Full-text search over case chunks and legal units (4.2).

- R-03: user text is Lucene-escaped and every term is quoted before it reaches an index.
  The resulting string is still passed as the $lucene parameter, never spliced into Cypher.
- R-09: case search over-fetches then filters by scope: fetch_k starts at 50 and doubles
  while fewer than k rows survive and fetch_k is below 800, at most 5 rounds.
- Legal search is unscoped (shared KB), so it fetches 30 once (4.2).
- R-04: logs carry counts and timings only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Final

from app.services.retrieval import graph_queries
from app.services.retrieval.neo4j_service import Neo4jService
from app.services.retrieval.scope import ScopedGraph

logger = logging.getLogger(__name__)

# R-09
FETCH_K_START: Final = 50
FETCH_K_MAX: Final = 800
MAX_ROUNDS: Final = 5
# 4.2: legal vector and full-text fetch 30 each.
LEGAL_FETCH_K: Final = 30

# R-03: + - && || ! ( ) { } [ ] ^ " ~ * ? : / and backslash. '&' and '|' are escaped
# singly, which also covers the two-character operators.
_LUCENE_SPECIAL: Final = frozenset('+-&|!(){}[]^"~*?:/\\')


def escape_lucene(text: str) -> str:
    return "".join("\\" + ch if ch in _LUCENE_SPECIAL else ch for ch in text)


def build_lucene_query(text: str) -> str | None:
    """Escape each whitespace-separated term and quote it (R-03). Repeated terms are kept
    once. Returns None when nothing searchable is left, so the caller skips the search."""
    seen: set[str] = set()
    terms: list[str] = []
    for raw in text.split():
        term = raw.strip()
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        terms.append(f'"{escape_lucene(term)}"')
    return " ".join(terms) or None


async def overfetch(
    fetch: Callable[[int], Awaitable[list[dict]]], k: int
) -> tuple[list[dict], int]:
    """R-09 loop around one scoped search. `fetch(fetch_k)` returns the rows that survived
    the scope filter. Returns (at most k rows, rounds used)."""
    fetch_k = FETCH_K_START
    rounds = 0
    while True:
        rounds += 1
        rows = await fetch(fetch_k)
        if len(rows) >= k or fetch_k >= FETCH_K_MAX or rounds >= MAX_ROUNDS:
            return rows[:k], rounds
        fetch_k *= 2


async def case_fulltext_search(graph: ScopedGraph, text: str, k: int) -> list[dict]:
    """Scoped full-text search over CaseChunk; scope.py supplies case_id and user_id (R-01).
    Rows: id, text, doc_id, page, start_sec, end_sec, score."""
    lucene = build_lucene_query(text)
    if lucene is None:
        return []
    started = time.perf_counter()

    async def fetch(fetch_k: int) -> list[dict]:
        return await graph.read(
            graph_queries.CASE_FULLTEXT_SEARCH, {"lucene": lucene, "fetch_k": fetch_k, "k": k}
        )

    rows, rounds = await overfetch(fetch, k)
    logger.info(
        "case fulltext search: rows=%d k=%d rounds=%d ms=%d",
        len(rows), k, rounds, int((time.perf_counter() - started) * 1000),
    )
    return rows


async def legal_fulltext_search(
    neo4j: Neo4jService, text: str, k: int = LEGAL_FETCH_K
) -> list[dict]:
    """Unscoped full-text search over LegalUnit. Rows: id, score."""
    lucene = build_lucene_query(text)
    if lucene is None:
        return []
    return await neo4j.read(graph_queries.LEGAL_FULLTEXT_SEARCH, {"lucene": lucene, "k": k})
