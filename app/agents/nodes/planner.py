"""C3 planner: intent and search_query (Groq, temperature 0, 200 tokens, R-19).

Intent rules applied by code after the model (A-33):
- mode case_and_legal, legal_only or case_only fixes the intent; the model still writes
  search_query.
- with no case_id, CASE_ONLY and CASE_AND_LEGAL become LEGAL_ONLY (3.4: null case_id with mode
  auto behaves as legal_only).
- E-06: a case with no ready document gives 409 NO_READY_DOCUMENTS for mode case_only (checked
  before the LLM call); other case intents continue as LEGAL_ONLY with a gaps note.
"""

from __future__ import annotations

import copy

from langchain_core.runnables import RunnableConfig

from app.agents.state import AgentState, QueryDeps, add_llm_usage, configurable, history_block, timed
from app.errors import AppError, ErrorCode
from app.prompts.planner import PLANNER_SYSTEM, PLANNER_USER
from app.schemas.output import PlannerReply
from app.services.retrieval import graph_queries
from app.services.retrieval.scope import CaseScope, ScopedGraph

PLANNER_TEMPERATURE = 0.0  # R-19
PLANNER_MAX_TOKENS = 200  # R-19

MODE_INTENT = {"case_and_legal": "CASE_AND_LEGAL", "legal_only": "LEGAL_ONLY", "case_only": "CASE_ONLY"}
CASE_INTENTS = ("CASE_ONLY", "CASE_AND_LEGAL")
NO_READY_DOCUMENTS_NOTE = "The case has no processed documents yet, so only the statutory text was searched."


async def _ready_documents(deps: QueryDeps, user_id: str, case_id: str) -> int:
    graph = ScopedGraph(deps.neo4j, CaseScope(case_id, user_id))
    rows = await graph.read(graph_queries.CASE_READY_DOC_COUNT)
    return int(rows[0]["n"]) if rows else 0


async def planner(state: AgentState, config: RunnableConfig, *, deps: QueryDeps) -> dict:
    trace = copy.deepcopy(state["trace"])
    mode = configurable(config).get("mode", "auto")
    case_id = state.get("case_id")
    question = state["original_query"]

    with timed(trace, "planner"):
        ready = await _ready_documents(deps, state["user_id"], case_id) if case_id else 0
        if mode == "case_only" and ready == 0:
            raise AppError(ErrorCode.NO_READY_DOCUMENTS)  # E-06

        result = await deps.llm.complete_json(
            [
                ("system", PLANNER_SYSTEM),
                ("user", PLANNER_USER.format(
                    has_case="yes" if case_id else "no",
                    history=history_block(state.get("messages", [])),
                    question=question,
                )),
            ],
            PlannerReply,
            temperature=PLANNER_TEMPERATURE,
            max_tokens=PLANNER_MAX_TOKENS,
        )
    add_llm_usage(trace, result.prompt_tokens, result.completion_tokens)

    intent = MODE_INTENT.get(mode, result.value.intent)
    if intent in CASE_INTENTS and not case_id:
        intent = "LEGAL_ONLY"
    if intent in CASE_INTENTS and ready == 0:
        intent = "LEGAL_ONLY"  # E-06
        trace["notes"].append(NO_READY_DOCUMENTS_NOTE)
    search_query = result.value.search_query.strip() or question
    return {"intent": intent, "search_query": search_query, "trace": trace}
