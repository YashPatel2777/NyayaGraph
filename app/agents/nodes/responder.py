"""C9 responder: Groq with the LegalAnalysis reply schema (temperature 0.1, 1,800 tokens,
R-19).

R-11: the case and legal text sent is at most 18,000 characters; the lowest-ranked item is
dropped first (by rerank score across both lists) and no item is cut in the middle.
R-20, A-36: items are shown as labels C1.. and P1..; the label maps go into
analysis['labels'] for the verifier, which accepts nothing else.
CONVERSATIONAL questions get no material.
"""

from __future__ import annotations

import copy

from langchain_core.runnables import RunnableConfig

from app.agents.state import AgentState, QueryDeps, add_llm_usage, history_block, neutralise, timed
from app.prompts.responder import (
    CHUNK_BLOCK,
    NO_MATERIAL,
    PROVISION_BLOCK,
    RESPONDER_SYSTEM,
    RESPONDER_USER,
)
from app.schemas.output import LegalAnalysisReply

RESPONDER_TEMPERATURE = 0.1  # R-19
RESPONDER_MAX_TOKENS = 1800  # R-19
CONTEXT_CHARS = 18000  # R-11


def fit_context(chunks: list[dict], provisions: list[dict], limit: int = CONTEXT_CHARS) -> tuple[list[dict], list[dict]]:
    """R-11: drop the lowest-scored item (either list) until the texts fit in limit."""
    items = [("c", i, c) for i, c in enumerate(chunks)] + [("p", i, p) for i, p in enumerate(provisions)]
    total = sum(len(x[2]["text"]) for x in items)
    for entry in sorted(items, key=lambda x: (x[2].get("score") or 0.0, x[0] == "c", -x[1])):
        if total <= limit:
            break
        items.remove(entry)
        total -= len(entry[2]["text"])
    return [x[2] for x in items if x[0] == "c"], [x[2] for x in items if x[0] == "p"]


def _chunk_source(chunk: dict, doc_numbers: dict[str, int]) -> str:
    where = f"document {doc_numbers[chunk['doc_id']]}"
    if chunk.get("page") is not None:
        return f"{where}, page {chunk['page']}"
    if chunk.get("start_sec") is not None:
        return f"{where}, {chunk['start_sec']:.0f} to {chunk.get('end_sec') or chunk['start_sec']:.0f} s"
    return where


async def responder(state: AgentState, config: RunnableConfig, *, deps: QueryDeps) -> dict:
    trace = copy.deepcopy(state["trace"])
    intent = state["intent"]
    if intent == "CONVERSATIONAL":
        chunks, provisions = [], []
    else:
        chunks, provisions = fit_context(state.get("case_chunks", []), state.get("legal_final", []))

    chunk_labels = {f"C{i + 1}": c["id"] for i, c in enumerate(chunks)}
    provision_labels = {f"P{i + 1}": p["id"] for i, p in enumerate(provisions)}
    doc_numbers: dict[str, int] = {}
    for c in chunks:
        doc_numbers.setdefault(c["doc_id"], len(doc_numbers) + 1)
    case_material = "\n\n".join(
        CHUNK_BLOCK.format(label=label, source=_chunk_source(c, doc_numbers), text=neutralise(c["text"]))
        for label, c in zip(chunk_labels, chunks)
    ) or NO_MATERIAL
    provision_text = "\n\n".join(
        PROVISION_BLOCK.format(label=label, citation=p["citation"], text=neutralise(p["text"]))
        for label, p in zip(provision_labels, provisions)
    ) or NO_MATERIAL

    with timed(trace, "generate"):
        result = await deps.llm.complete_json(
            [
                ("system", RESPONDER_SYSTEM),
                ("user", RESPONDER_USER.format(
                    intent=intent,
                    history=history_block(state.get("messages", [])),
                    question=neutralise(state["original_query"]),
                    case_material=case_material,
                    provisions=provision_text,
                )),
            ],
            LegalAnalysisReply,
            temperature=RESPONDER_TEMPERATURE,
            max_tokens=RESPONDER_MAX_TOKENS,
        )
    add_llm_usage(trace, result.prompt_tokens, result.completion_tokens)
    trace["context_chars"] = sum(len(c["text"]) for c in chunks) + sum(len(p["text"]) for p in provisions)
    return {
        "analysis": {
            "reply": result.value.model_dump(),
            "labels": {"chunks": chunk_labels, "provisions": provision_labels},
        },
        "trace": trace,
    }
