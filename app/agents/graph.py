"""LangGraph wiring and routing for the query agent (1.1, 1.4 C).

Node order (1.1): planner > case_retriever > issue_builder > legal_retriever >
graph_expander > legal_reranker > responder > verifier.

Routing (1.4 C):
- CONVERSATIONAL goes straight to responder.
- LEGAL_ONLY skips case_retriever and issue_builder (the issue statement equals the query).
- CASE_ONLY ends after case_retriever and goes to responder.
- E-05: case_retriever can downgrade to LEGAL_ONLY, which then continues at legal_retriever.

The graph is compiled once with a MemorySaver checkpointer (section 0); the thread key is
{user_id}:{case_id or none}:{thread_id} (3.4). run_query applies C1 (the case must exist and
be active) and E-12 (45 s budget, 504 QUERY_TIMEOUT).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.nodes.case_retriever import case_retriever
from app.agents.nodes.graph_expander import graph_expander
from app.agents.nodes.issue_builder import issue_builder
from app.agents.nodes.legal_reranker import legal_reranker
from app.agents.nodes.legal_retriever import legal_retriever
from app.agents.nodes.planner import planner
from app.agents.nodes.responder import responder
from app.agents.nodes.verifier import verifier
from app.agents.state import AgentState, QueryDeps, turn_input
from app.errors import AppError, ErrorCode
from app.guardrails.hook import guard
from app.ingestion.case.case_processor import require_active_case
from app.schemas.output import DISCLAIMER, AnalysisResponse, Confidence, Trace
from app.services.retrieval.scope import CaseScope, ScopedGraph

logger = logging.getLogger(__name__)

QUERY_TIMEOUT_S = 45.0  # 4.2, E-12

NODES = {
    "planner": planner,
    "case_retriever": case_retriever,
    "issue_builder": issue_builder,
    "legal_retriever": legal_retriever,
    "graph_expander": graph_expander,
    "legal_reranker": legal_reranker,
    "responder": responder,
    "verifier": verifier,
}


def route_after_planner(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "CONVERSATIONAL":
        return "responder"
    if intent == "LEGAL_ONLY":
        return "legal_retriever"
    return "case_retriever"


def route_after_case(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "CASE_ONLY":
        return "responder"
    if intent == "LEGAL_ONLY":  # E-05 downgrade
        return "legal_retriever"
    return "issue_builder"


def build_graph(deps: QueryDeps, checkpointer: Any | None = None):
    builder = StateGraph(AgentState)
    for name, node in NODES.items():
        builder.add_node(name, partial(node, deps=deps))
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_after_planner,
                                  ["responder", "legal_retriever", "case_retriever"])
    builder.add_conditional_edges("case_retriever", route_after_case,
                                  ["responder", "legal_retriever", "issue_builder"])
    builder.add_edge("issue_builder", "legal_retriever")
    builder.add_edge("legal_retriever", "graph_expander")
    builder.add_edge("graph_expander", "legal_reranker")
    builder.add_edge("legal_reranker", "responder")
    builder.add_edge("responder", "verifier")
    builder.add_edge("verifier", END)
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def thread_key(user_id: str, case_id: str | None, thread_id: str) -> str:
    return f"{user_id}:{case_id or 'none'}:{thread_id}"  # 3.4


async def run_query(
    graph: Any,
    deps: QueryDeps,
    *,
    user_id: str,
    case_id: str | None,
    question: str,
    thread_id: str,
    mode: str = "auto",
    max_case_chunks: int = 5,
    max_provisions: int = 8,
    timeout_s: float = QUERY_TIMEOUT_S,
) -> dict:
    """One query turn. Returns the final state. C1: 404 CASE_NOT_FOUND or 409 CASE_DELETING
    before any model call; E-12: 504 QUERY_TIMEOUT when the run passes the budget."""
    if case_id is not None:
        await require_active_case(ScopedGraph(deps.neo4j, CaseScope(case_id, user_id)))
    config = {"configurable": {
        "thread_id": thread_key(user_id, case_id, thread_id),
        "mode": mode, "max_case_chunks": max_case_chunks, "max_provisions": max_provisions,
    }}
    try:
        return await asyncio.wait_for(
            graph.ainvoke(turn_input(user_id, case_id, question), config), timeout=timeout_s
        )
    except TimeoutError as exc:
        raise AppError(ErrorCode.QUERY_TIMEOUT) from exc


def to_response(
    state: dict, *, request_id: str, case_id: str | None, thread_id: str, include_trace: bool, total_ms: int
) -> AnalysisResponse:
    """The verified analysis as the 3.5 response. Trace only when options.include_trace."""
    trace = None
    if include_trace:
        t = state["trace"]
        trace = Trace(
            search_query=state.get("search_query", ""),
            issue_statement=state.get("issue_statement", ""),
            downgraded=t["downgraded"],
            counts=t["counts"],
            timings_ms={**t["timings_ms"], "total": total_ms},
            llm=t["llm"],
        )
    return AnalysisResponse(
        request_id=request_id, case_id=case_id, thread_id=thread_id, intent=state["intent"],
        trace=trace, **state["analysis"],
    )


def blocked_response(reply: str | None, *, request_id: str, case_id: str | None, thread_id: str) -> AnalysisResponse:
    """C2: the guard's block reply, shaped as an AnalysisResponse (A-41)."""
    return AnalysisResponse(
        request_id=request_id, case_id=case_id, thread_id=thread_id, intent="CONVERSATIONAL",
        summary=(reply or "This request cannot be answered.")[:800], relevant_facts=[],
        applicable_provisions=[], precedents=[], analysis="", gaps_or_conflicts=[],
        confidence=Confidence(level="low", reasons=["request blocked by guardrails"]),
        disclaimer=DISCLAIMER,
    )


@dataclass(frozen=True)
class QueryAgent:
    """The compiled graph and its services, created once at startup (1.2)."""

    graph: Any
    deps: QueryDeps
    enable_guardrails: bool = False

    async def answer(
        self,
        *,
        request_id: str,
        user_id: str,
        case_id: str | None,
        question: str,
        thread_id: str,
        mode: str,
        max_case_chunks: int,
        max_provisions: int,
        include_trace: bool,
    ) -> AnalysisResponse:
        start = time.perf_counter()
        if self.enable_guardrails:  # C2
            verdict = guard(question, user_id)
            if verdict.blocked:
                return blocked_response(verdict.reply, request_id=request_id, case_id=case_id, thread_id=thread_id)
        state = await run_query(
            self.graph, self.deps, user_id=user_id, case_id=case_id, question=question,
            thread_id=thread_id, mode=mode, max_case_chunks=max_case_chunks, max_provisions=max_provisions,
        )
        total_ms = int((time.perf_counter() - start) * 1000)
        counts = state["trace"]["counts"]
        # R-04: ids, counts and timings only.
        logger.info(
            "query_done request_id=%s intent=%s case_final=%d legal_final=%d dropped=%d "
            "prompt_tokens=%d completion_tokens=%d ms=%d",
            request_id, state["intent"], counts["case_final"], counts["legal_final"],
            counts["dropped_citations"], state["trace"]["llm"]["prompt_tokens"],
            state["trace"]["llm"]["completion_tokens"], total_ms,
        )
        return to_response(state, request_id=request_id, case_id=case_id, thread_id=thread_id,
                           include_trace=include_trace, total_ms=total_ms)
