"""C4 case_retriever: scoped vector and full-text search, RRF, FlashRank, top chunks.

- Vector search filters by scope inside the index (SC-15); full-text over-fetches (R-09).
- The two lists are fused with RRF (R-10); the best 15 go to the reranker (4.2) and at most
  options.max_case_chunks (default 5) are kept (R-11).
- E-05: if nothing matches, case_chunks is empty, the intent becomes LEGAL_ONLY with the
  issue statement equal to the original query (SC-22), trace.downgraded is set and a gaps
  note added.
"""

from __future__ import annotations

import asyncio
import copy

from langchain_core.runnables import RunnableConfig

from app.agents.state import AgentState, QueryDeps, configurable, timed
from app.services.retrieval.fulltext_search import case_fulltext_search
from app.services.retrieval.hybrid import rrf_fuse
from app.services.retrieval.scope import CaseScope, ScopedGraph
from app.services.retrieval.vector_search import case_vector_search

CASE_CANDIDATES = 15  # 4.2: case candidates before rerank
CASE_CHUNKS_KEPT = 5  # 4.2, R-11; options.max_case_chunks overrides (3.4)
NO_CASE_MATERIAL_NOTE = "No case material matched the question."  # E-05

CHUNK_FIELDS = ("id", "text", "doc_id", "page", "start_sec", "end_sec")


async def case_retriever(state: AgentState, config: RunnableConfig, *, deps: QueryDeps) -> dict:
    trace = copy.deepcopy(state["trace"])
    keep = int(configurable(config).get("max_case_chunks", CASE_CHUNKS_KEPT))
    query = state["search_query"]
    graph = ScopedGraph(deps.neo4j, CaseScope(state["case_id"], state["user_id"]))

    with timed(trace, "case"):
        query_vec = await deps.embeddings.embed_query(query)
        vector_rows, text_rows = await asyncio.gather(
            case_vector_search(graph, query_vec, CASE_CANDIDATES),
            case_fulltext_search(graph, query, CASE_CANDIDATES),
        )
        fused = rrf_fuse([vector_rows, text_rows])[:CASE_CANDIDATES]
        trace["counts"]["case_candidates"] = len(fused)
        reranked = await deps.ranker.rerank(query, fused, top_n=keep) if fused else []

    chunks = [
        {**{f: row.get(f) for f in CHUNK_FIELDS}, "score": row["rerank_score"]} for row in reranked
    ]
    trace["counts"]["case_final"] = len(chunks)
    update: dict = {"case_chunks": chunks, "trace": trace}
    if not chunks:  # E-05
        trace["downgraded"] = True
        trace["notes"].append(NO_CASE_MATERIAL_NOTE)
        update.update(intent="LEGAL_ONLY", issue_statement=state["original_query"])  # SC-22
    return update
