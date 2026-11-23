"""Agent state (2.5) and what every query node shares.

AgentState is exactly the 2.5 TypedDict. Per-request settings that are not state (mode,
options) travel in the LangGraph config under "configurable" as plain values; services are
bound into the nodes when the graph is built (QueryDeps), so nothing unserialisable reaches
the MemorySaver checkpointer.

MemorySaver keeps the whole state per thread, so every per-turn field is reset at the start
of a turn (turn_input); only `messages` accumulates across turns.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional, TypedDict

from app.services.llm.groq_client import GroqClient
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.neo4j_service import Neo4jService
from app.services.retrieval.ranking_service import RankingService


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    user_id: str
    case_id: Optional[str]
    original_query: str
    search_query: str
    intent: Literal['CONVERSATIONAL', 'CASE_ONLY', 'LEGAL_ONLY', 'CASE_AND_LEGAL']
    case_chunks: list[dict]        # id, text, doc_id, page, start_sec, end_sec, score
    case_entities: list[dict]
    issue_statement: str
    legal_candidates: list[dict]   # id, kind, text, act_id, number, score, via
    graph_context: list[dict]
    legal_final: list[dict]
    analysis: dict
    trace: dict


@dataclass(frozen=True)
class QueryDeps:
    neo4j: Neo4jService
    embeddings: EmbeddingService
    llm: GroqClient
    ranker: RankingService
    concepts: Any  # nodes.graph_expander.ConceptIndex (R-13), loaded at startup
    rerank_high: float  # RERANK_HIGH (1.3, R-24)


def new_trace() -> dict:
    """Code-side accounting for one turn. `notes` are gaps_or_conflicts lines written by code
    (E-05, E-06); the rest maps onto Trace (3.5)."""
    return {
        "downgraded": False,
        "notes": [],
        "counts": {
            "case_candidates": 0, "case_final": 0, "legal_candidates": 0, "legal_final": 0,
            "expanded_nodes": 0, "dropped_citations": 0,
        },
        "timings_ms": {},
        "llm": {"prompt_tokens": 0, "completion_tokens": 0},
    }


def turn_input(user_id: str, case_id: str | None, question: str) -> dict:
    """The graph input for one turn: every per-turn field reset (see module docstring)."""
    return {
        "user_id": user_id,
        "case_id": case_id,
        "original_query": question,
        "search_query": question,
        "case_chunks": [],
        "case_entities": [],
        "issue_statement": "",
        "legal_candidates": [],
        "graph_context": [],
        "legal_final": [],
        "analysis": {},
        "trace": new_trace(),
    }


def add_llm_usage(trace: dict, prompt_tokens: int, completion_tokens: int) -> None:
    trace["llm"]["prompt_tokens"] += prompt_tokens
    trace["llm"]["completion_tokens"] += completion_tokens


@contextmanager
def timed(trace: dict, name: str) -> Iterator[None]:
    """Add the block's wall time to trace.timings_ms[name] (3.5 Trace)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = int((time.perf_counter() - start) * 1000)
        trace["timings_ms"][name] = trace["timings_ms"].get(name, 0) + ms


def configurable(config: Any) -> dict:
    return dict((config or {}).get("configurable", {}))


# R-06: block markers used by the query prompts. Case text, earlier turns and statutory text
# never contain them once neutralised, so no block can be closed early.
DELIMITERS = (
    "<<<CASE MATERIAL>>>", "<<<END CASE MATERIAL>>>", "<<<PROVISIONS>>>", "<<<END PROVISIONS>>>",
    "<<<CONVERSATION>>>", "<<<END CONVERSATION>>>",
)
NO_HISTORY = "(none)"
HISTORY_TURNS = 3  # A-34
HISTORY_CHARS = 500  # A-34: per message


def neutralise(text: str) -> str:
    for marker in DELIMITERS:
        text = text.replace(marker, "[marker removed]")
    return text


def history_block(messages: list[dict]) -> str:
    """The last HISTORY_TURNS question-and-answer pairs of the thread, each message cut to
    HISTORY_CHARS (A-34)."""
    recent = messages[-2 * HISTORY_TURNS:]
    if not recent:
        return NO_HISTORY
    lines = []
    for m in recent:
        who = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{who}: {neutralise(str(m.get('content', '')))[:HISTORY_CHARS]}")
    return "\n".join(lines)
