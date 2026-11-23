"""C7 graph_expander: the 4.2 expansion query around the legal candidates (R-12), plus
definitions (R-13).

R-12: REFERS_TO one hop both ways (outgoing up to 10, incoming up to 5 per unit), then one
more hop on outgoing edges only; at most 40 expanded nodes. Expanded Sections that are not
retrieval units carry their text truncated to 1,200 characters. Rules linked by
PRESCRIBES_FOR are added first, because R-27 makes their text take precedence.
R-13: each LegalConcept whose lowercase name appears as a whole word in a candidate or
expanded text adds its definition Clause, at most 5 per query. The concept list is cached at
startup (ConceptIndex).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig

from app.agents.nodes.legal_retriever import hydrate
from app.agents.state import AgentState, QueryDeps, timed
from app.services.retrieval import graph_queries
from app.services.retrieval.neo4j_service import Neo4jService

MAX_EXPANDED = 40  # R-12
SECTION_TEXT_CAP = 1200  # R-12
MAX_DEFINITIONS = 5  # R-13


@dataclass(frozen=True)
class Concept:
    id: str
    name: str  # lowercase, for matching
    clause_id: str
    pattern: re.Pattern[str]
    term: str = ""  # as defined, for display (A-45)


class ConceptIndex:
    """The R-13 concept list, loaded once at startup."""

    def __init__(self, rows: list[dict]) -> None:
        self.concepts = [
            Concept(r["id"], r["name"].lower(), r["clause_id"],
                    re.compile(r"(?<![a-z0-9])" + re.escape(r["name"].lower()) + r"(?![a-z0-9])"),
                    r["name"])
            for r in rows if r.get("name")
        ]

    @classmethod
    async def load(cls, neo4j: Neo4jService) -> ConceptIndex:
        return cls(await neo4j.read(graph_queries.LEGAL_CONCEPTS))

    def terms_in(self, text: str, exclude: set[str] | None = None) -> list[tuple[str, str]]:
        """(term, definition Clause id) for every concept named in text, in order of first
        appearance; no R-13 cap (GET /v1/legal/provisions, A-45)."""
        lower = text.lower()
        hits = sorted(
            (m.start(), c.term, c.clause_id)
            for c in self.concepts
            if c.clause_id not in (exclude or set()) and (m := c.pattern.search(lower))
        )
        return [(term, clause_id) for _, term, clause_id in hits]

    def definitions_for(self, texts: list[str], exclude: set[str], limit: int = MAX_DEFINITIONS) -> list[str]:
        """Definition Clause ids for concepts named in texts, in order of first appearance
        (texts in rank order, then position in the text)."""
        found: list[str] = []
        for text in texts:
            lower = text.lower()
            hits = sorted(
                (m.start(), c.clause_id)
                for c in self.concepts
                if c.clause_id not in exclude and (m := c.pattern.search(lower))
            )
            for _, clause_id in hits:
                if clause_id not in found:
                    found.append(clause_id)
                    if len(found) == limit:
                        return found
        return found


def truncate_words(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return cut[:space] if space > limit // 2 else cut


async def graph_expander(state: AgentState, config: RunnableConfig, *, deps: QueryDeps) -> dict:
    trace = copy.deepcopy(state["trace"])
    candidates = [dict(c) for c in state.get("legal_candidates", [])]  # annotated below
    seen = {c["id"] for c in candidates}

    with timed(trace, "expand"):
        rows = await deps.neo4j.read(graph_queries.LEGAL_EXPANSION, {"unit_ids": [c["id"] for c in candidates]})
        by_unit = {r["unit_id"]: r for r in rows}
        prescribed: dict[str, list[str]] = {}  # section id -> rule ids (R-27)
        expanded: list[tuple[str, str]] = []  # (id, via), in priority order

        def add(node_id: str, via: str) -> None:
            if node_id not in seen and len(expanded) < MAX_EXPANDED:
                seen.add(node_id)
                expanded.append((node_id, via))

        for c in candidates:  # rank order
            row = by_unit.get(c["id"])
            if row and row["prescribed_by"]:
                prescribed[row["section_id"]] = list(row["prescribed_by"])
                for rule_id in row["prescribed_by"]:
                    add(rule_id, "prescribes_for")
        hop1_out: list[str] = []
        for c in candidates:
            row = by_unit.get(c["id"])
            if not row:
                continue
            for out_id in row["refers_to"]:
                hop1_out.append(out_id)
                add(out_id, "refers_to")
            for inc_id in row["referenced_by"]:
                add(inc_id, "referenced_by")
        if hop1_out and len(expanded) < MAX_EXPANDED:
            hop2 = await deps.neo4j.read(
                graph_queries.LEGAL_EXPANSION_OUTGOING, {"ids": list(dict.fromkeys(hop1_out))}
            )
            for row in hop2:
                for out_id in row["refers_to"]:
                    add(out_id, "refers_to_hop2")

        items = await hydrate(deps, [i for i, _ in expanded])
        context = []
        for node_id, via in expanded:
            item = items.get(node_id)
            if item is None:
                continue
            if item["kind"] == "section" and not item["is_unit"]:
                item = {**item, "text": truncate_words(item["text"], SECTION_TEXT_CAP)}  # R-12
            context.append({**item, "score": 0.0, "rank": None, "via": via})

        texts = [x["text"] for x in candidates + context]
        definition_ids = deps.concepts.definitions_for(texts, exclude=seen)
        definitions = await hydrate(deps, definition_ids)
        context += [
            {**definitions[d], "score": 0.0, "rank": None, "via": "definition"}
            for d in definition_ids if d in definitions
        ]

    for item in candidates + context:
        rules = prescribed.get(item.get("section_id") or "")
        if rules:
            item["prescribed_by"] = rules
    trace["counts"]["expanded_nodes"] = len(context)
    return {"legal_candidates": candidates, "graph_context": context, "trace": trace}
