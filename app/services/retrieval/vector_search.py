"""Vector search over legal units and case chunks (4.2).

Query vectors come only from EmbeddingService.embed_query, which adds 'search_query: '
(R-07). Legal search is unscoped (shared KB) and fetches 30 (4.2). Case search filters by
scope inside the index (SC-15), so it needs no R-09 over-fetch loop; scope.py supplies
case_id and user_id (R-01). R-04: logs carry counts and timings only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from app.services.retrieval import graph_queries
from app.services.retrieval.fulltext_search import LEGAL_FETCH_K
from app.services.retrieval.neo4j_service import Neo4jService
from app.services.retrieval.scope import ScopedGraph

logger = logging.getLogger(__name__)


async def legal_vector_search(
    neo4j: Neo4jService, query_vec: Sequence[float], k: int = LEGAL_FETCH_K
) -> list[dict]:
    """Unscoped vector search over LegalUnit. Rows: id, score."""
    return await neo4j.read(graph_queries.LEGAL_VECTOR_SEARCH, {"k": k, "query_vec": list(query_vec)})


async def case_vector_search(graph: ScopedGraph, query_vec: Sequence[float], k: int) -> list[dict]:
    """Scoped vector search over CaseChunk (SC-15). Rows: id, text, doc_id, page, start_sec,
    end_sec, score; at most k, best first."""
    started = time.perf_counter()
    rows = await graph.read(graph_queries.CASE_VECTOR_SEARCH, {"query_vec": list(query_vec), "k": k})
    logger.info(
        "case vector search: rows=%d k=%d ms=%d",
        len(rows), k, int((time.perf_counter() - started) * 1000),
    )
    return rows
