"""C6 legal_retriever: hybrid search over LegalUnit for the issue statement and, separately,
the original query; RRF; plus seed units reached through case-graph links.

- Each text runs vector and full-text search, 30 each (4.2), fused with RRF (R-10); the two
  fused lists are fused again with RRF and the best 20 form the pool (4.2).
- Seeds: legal ids from case_entities (claims' ALLEGES_VIOLATION_OF, then chunks' INVOKES),
  at most 10, not already in the pool (4.2).
- SC-22: the issue statement equals the original query for every intent, so the search runs
  on the question alone (one fused list) and the second search is skipped as a duplicate.

hydrate() is shared by the later legal nodes and the verifier: it reads provisions from Neo4j
by id, so citation, status, act_as_on and text always come from the store (R-21).
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Iterable

from langchain_core.runnables import RunnableConfig

from app.agents.state import AgentState, QueryDeps, timed
from app.schemas import graph_schema as gs
from app.services.retrieval import graph_queries
from app.services.retrieval.fulltext_search import legal_fulltext_search
from app.services.retrieval.hybrid import rrf_fuse
from app.services.retrieval.vector_search import legal_vector_search

LEGAL_POOL = 20  # 4.2: legal pool before expansion
MAX_SEEDS = 10  # 4.2: plus up to 10 seed units from case-graph links


def _native(value: object) -> object:
    to_native = getattr(value, "to_native", None)
    return to_native() if callable(to_native) else value


def _item(row: dict) -> dict:
    is_section = bool(row["is_section"])
    label = None if is_section else row.get("label")
    kind = "section" if is_section else (row.get("kind") or "subsection")
    act_title = row.get("act_short_title") or ""
    return {
        "id": row["id"],
        "base_id": gs.provision_base_id(row["id"]),
        "kind": kind,
        "is_unit": bool(row["is_unit"]),
        "text": row.get("text") or "",
        "search_text": row.get("search_text"),
        "act_id": row.get("act_node_id"),
        "act_short_title": act_title,
        "act_kind": row.get("act_kind"),
        "act_as_on": _native(row.get("act_as_on")),
        "number": row.get("section_number"),
        "label": label,
        "section_id": row.get("section_id"),
        "status": row.get("status") or "in_force",
        "citation": gs.format_citation(row.get("section_number") or "", label, act_title, row.get("act_kind")),
    }


async def hydrate(deps: QueryDeps, ids: Iterable[str]) -> dict[str, dict]:
    """Provision items by id, read from Neo4j. Unknown ids are absent from the result."""
    wanted = list(dict.fromkeys(ids))
    if not wanted:
        return {}
    rows = await deps.neo4j.read(graph_queries.LEGAL_PROVISIONS_HYDRATE, {"ids": wanted})
    return {row["id"]: _item(row) for row in rows}


async def _hybrid(deps: QueryDeps, text: str) -> list[dict]:
    query_vec = await deps.embeddings.embed_query(text)
    vector_rows, text_rows = await asyncio.gather(
        legal_vector_search(deps.neo4j, query_vec), legal_fulltext_search(deps.neo4j, text)
    )
    return rrf_fuse([vector_rows, text_rows])


def seed_ids(case_entities: list[dict]) -> list[str]:
    """Claims' linked units first, then chunks' regex citations, in order, without repeats."""
    ordered = [e for e in case_entities if e["kind"] == gs.CLAIM] + [
        e for e in case_entities if e["kind"] == gs.CASE_CHUNK
    ]
    return list(dict.fromkeys(i for e in ordered for i in e.get("legal_ids", [])))


async def legal_retriever(state: AgentState, config: RunnableConfig, *, deps: QueryDeps) -> dict:
    trace = copy.deepcopy(state["trace"])
    issue = state.get("issue_statement") or state["original_query"]  # SC-22
    texts = list(dict.fromkeys([issue, state["original_query"]]))

    with timed(trace, "legal"):
        fused_lists = await asyncio.gather(*(_hybrid(deps, t) for t in texts))
        pool = rrf_fuse(fused_lists)[:LEGAL_POOL]
        pool_ids = [r["id"] for r in pool]
        seeds = [i for i in seed_ids(state.get("case_entities", [])) if i not in pool_ids][:MAX_SEEDS]
        items = await hydrate(deps, pool_ids + seeds)

    candidates = []
    for rank, row in enumerate(pool, start=1):
        if row["id"] in items:
            candidates.append({**items[row["id"]], "score": row["rrf_score"], "rank": rank, "via": "search"})
    for sid in seeds:
        if sid in items:
            candidates.append({**items[sid], "score": 0.0, "rank": None, "via": "case_link"})
    trace["counts"]["legal_candidates"] = len(candidates)
    return {"legal_candidates": candidates, "issue_statement": issue, "trace": trace}
