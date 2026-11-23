"""C8 legal_reranker: FlashRank over candidates and expanded nodes against the issue
statement plus the query; keep the top 8 (R-11; options.max_provisions overrides, 3.4).

- Split parts of one Clause or Section are merged into one provision under the base id, in
  part order, scored by their best part (A-35), so two parts never take two slots.
- R-27: when a kept Section has a rule linked by PRESCRIBES_FOR in the pool, the rule is
  kept too, replacing the lowest-ranked provision that is not itself pinned.
"""

from __future__ import annotations

import copy
import re

from langchain_core.runnables import RunnableConfig

from app.agents.state import AgentState, QueryDeps, configurable, timed

LEGAL_UNITS_KEPT = 8  # 4.2, R-11

_PART_NO = re.compile(r"-p([0-9]+)$")


def _rerank_text(item: dict) -> str:
    return item.get("search_text") or f"{item['citation']}\n{item['text']}"


def _part_no(unit_id: str) -> int:
    m = _PART_NO.search(unit_id)
    return int(m.group(1)) if m else 0


def merge_parts(ranked: list[dict]) -> list[dict]:
    """Group items by base_id in rank order; a group's text joins its parts in part order."""
    groups: dict[str, list[dict]] = {}
    for item in ranked:
        groups.setdefault(item["base_id"], []).append(item)
    merged = []
    for base_id, items in groups.items():
        best = items[0]
        if len(items) == 1 and best["id"] == base_id:
            merged.append({**best, "unit_ids": [best["id"]]})
            continue
        parts = sorted(items, key=lambda x: _part_no(x["id"]))
        merged.append({
            **best, "id": base_id, "unit_ids": [p["id"] for p in parts],
            "text": "\n".join(p["text"] for p in parts),
        })
    return merged


def pin_prescribed_rules(selected: list[dict], pool: list[dict], keep: int) -> list[dict]:
    """R-27: add rules that prescribe for a selected Section, dropping the lowest-ranked
    unpinned items so the list stays within keep."""
    by_id = {p["id"]: p for p in pool}
    result = list(selected)
    pinned: set[str] = set()
    for item in selected:
        for rule_id in item.get("prescribed_by", []):
            rule = by_id.get(rule_id)
            if rule is None:
                continue
            pinned.update({item["id"], rule_id})
            if all(r["id"] != rule_id for r in result):
                result.append(rule)
    while len(result) > keep:
        drop = next((r for r in reversed(result) if r["id"] not in pinned), None)
        if drop is None:
            break
        result.remove(drop)
    return result


async def legal_reranker(state: AgentState, config: RunnableConfig, *, deps: QueryDeps) -> dict:
    trace = copy.deepcopy(state["trace"])
    keep = int(configurable(config).get("max_provisions", LEGAL_UNITS_KEPT))
    issue, question = state.get("issue_statement", ""), state["original_query"]
    query = issue if issue == question else f"{issue} {question}".strip()

    pool = list({x["id"]: x for x in state.get("legal_candidates", []) + state.get("graph_context", [])}.values())
    with timed(trace, "rerank"):
        reranked = await deps.ranker.rerank(
            query, [{**x, "rerank_input": _rerank_text(x)} for x in pool], text_key="rerank_input"
        )
    merged = merge_parts([{k: v for k, v in x.items() if k != "rerank_input"} for x in reranked])
    for item in merged:
        item["score"] = item.pop("rerank_score")
    final = pin_prescribed_rules(merged[:keep], merged, keep)
    trace["counts"]["legal_final"] = len(final)
    return {"legal_final": final, "trace": trace}
