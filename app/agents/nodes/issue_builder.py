"""C5 issue_builder: case entities around the selected chunks; the issue statement equals the
original query (SC-22, option b; no LLM call).

case_entities holds the entities mentioned in the chunks, their one-hop neighbours (A-37:
at most 40), the legal units their claims are linked to (ALLEGES_VIOLATION_OF), and one
entry per chunk with regex citations (INVOKES). legal_retriever reads the legal ids from it
as seed units (C6).

SC-22: an LLM-written issue statement describes the whole case, and using it in C6 and C8
pulled retrieval away from narrow questions (O-8: recall 0.33 against 0.87 on 25 labelled
questions). The question itself is the issue statement for every intent, as for LEGAL_ONLY.
"""

from __future__ import annotations

import copy

from langchain_core.runnables import RunnableConfig

from app.agents.state import AgentState, QueryDeps, timed
from app.schemas import graph_schema as gs
from app.services.retrieval import graph_queries
from app.services.retrieval.scope import CaseScope, ScopedGraph

MAX_ENTITIES = 40  # A-37


def _entity_kind(labels: list[str]) -> str:
    for label in (gs.PARTY, gs.EVENT, gs.CLAIM, gs.EVIDENCE):
        if label in labels:
            return label
    return labels[0] if labels else "Entity"


def _entity_text(row: dict, kind: str) -> str:
    if kind == gs.PARTY:
        return f"{row['name']} ({row.get('kind') or 'party'}, {row.get('role') or 'other'})"
    if kind == gs.EVENT:
        return f"{row.get('date') or 'undated'}: {row.get('description') or ''}"
    if kind == gs.CLAIM:
        return f"{row.get('claim_type') or 'claim'}: {row.get('text') or ''}"
    return f"{row.get('kind') or 'evidence'}: {row.get('description') or ''}"


async def issue_builder(state: AgentState, config: RunnableConfig, *, deps: QueryDeps) -> dict:
    trace = copy.deepcopy(state["trace"])
    graph = ScopedGraph(deps.neo4j, CaseScope(state["case_id"], state["user_id"]))
    chunk_ids = [c["id"] for c in state.get("case_chunks", [])]

    with timed(trace, "issue"):
        entity_rows = await graph.read(
            graph_queries.CASE_ENTITIES_FOR_CHUNKS, {"chunk_ids": chunk_ids, "limit": MAX_ENTITIES}
        )
        invoke_rows = await graph.read(graph_queries.CASE_CHUNK_INVOKES, {"chunk_ids": chunk_ids})

    entities = []
    for row in entity_rows:
        kind = _entity_kind(row["labels"])
        entities.append({
            "id": row["id"], "kind": kind, "text": _entity_text(row, kind),
            "mentioned": bool(row["is_mentioned"]), "legal_ids": list(row["legal_ids"]),
        })
    for row in invoke_rows:
        entities.append({
            "id": row["chunk_id"], "kind": gs.CASE_CHUNK, "text": "", "mentioned": True,
            "legal_ids": list(row["legal_ids"]),
        })
    return {
        "case_entities": entities,
        "issue_statement": state["original_query"],  # SC-22
        "trace": trace,
    }
