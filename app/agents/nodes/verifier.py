"""C10 verifier: authoritative citation check, confidence and disclaimer.

- R-21: a fact whose chunk label, or a provision whose label, is not among the items the
  responder was given is removed and counted in trace.counts.dropped_citations. Citation,
  status, act_as_on, excerpt and source are filled from Neo4j data, never from the model.
- SC-16 (R-21): a fact is also removed unless its cited chunk supports it: at least 60 % of
  its content words and every number in it appear in the chunk text (A-40).
- SC-17 (R-21): C and P labels left in summary, analysis or gaps are replaced by the file
  name and page or the citation of an item that was sent; unknown labels are removed.
- R-23 as changed by SC-18: with no surviving provision, applicable_provisions is empty,
  confidence is low, and for LEGAL_ONLY and CASE_AND_LEGAL gaps_or_conflicts includes 'No
  supporting provision was retrieved.'
- R-24: confidence is computed here. R-25: the disclaimer is the constant DISCLAIMER.
- C11: the turn's question and summary are appended to the thread's messages.
"""

from __future__ import annotations

import copy
import re

from langchain_core.runnables import RunnableConfig

from app.agents.state import AgentState, QueryDeps, timed
from app.schemas.output import DISCLAIMER, Confidence, Fact, Provision, SourceRef
from app.services.retrieval import graph_queries
from app.services.retrieval.scope import CaseScope, ScopedGraph

NO_PROVISION_NOTE = "No supporting provision was retrieved."  # R-23
NO_PROVISION_NOTE_INTENTS = ("LEGAL_ONLY", "CASE_AND_LEGAL")  # SC-18
EXCERPT_CHARS = 600  # 3.5
RELEVANCE_MAX_CHARS = 500  # 3.5: Provision.relevance max_length
FACT_SUPPORT_MIN = 0.6  # SC-16: share of a fact's content words found in its chunk
STEM_CHARS = 5  # A-40: 'refused' and 'refuses' match on their first 5 letters

_DIGIT_COMMA = re.compile(r"(?<=[0-9]),(?=[0-9])")
_TOKEN = re.compile(r"[a-z0-9]+")
_LABEL = re.compile(r"\b[CP][0-9]{1,3}\b")
_EMPTY_PARENS = re.compile(r"[(\[]\s*[)\]]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:)])")
_SPACES = re.compile(r"[ \t]{2,}")
# A-40: function words that carry no fact.
STOPWORDS = frozenset(
    "a an and are as at be been but by did do does for from had has have he her his i in is it its "
    "my me of on or our she so that the their them they this to was we were which who will with "
    "you your not no".split()
)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(_DIGIT_COMMA.sub("", text.lower()))


def _number(token: str) -> str:
    return token.lstrip("0") or "0"


def fact_supported(fact: str, chunk_text: str) -> bool:
    """SC-16: every number in the fact, and at least FACT_SUPPORT_MIN of its content words
    (exactly or by their first STEM_CHARS letters), appear in the chunk."""
    chunk = _tokens(chunk_text)
    chunk_words = set(chunk)
    chunk_stems = {w[:STEM_CHARS] for w in chunk}
    chunk_numbers = {_number(w) for w in chunk if w.isdigit()}
    tokens = _tokens(fact)
    if any(_number(t) not in chunk_numbers for t in tokens if t.isdigit()):
        return False
    words = [t for t in tokens if not t.isdigit() and t not in STOPWORDS]
    if not words:
        return True
    found = sum(1 for w in words if w in chunk_words or w[:STEM_CHARS] in chunk_stems)
    return found / len(words) >= FACT_SUPPORT_MIN


def replace_labels(text: str, names: dict[str, str]) -> str:
    """SC-17: each C or P label becomes its name from `names`; an unknown label is removed."""
    out = _LABEL.sub(lambda m: names.get(m.group(0), ""), text)
    out = _EMPTY_PARENS.sub("", out)
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    return _SPACES.sub(" ", out).strip()


def chunk_name(src: dict) -> str:
    if src.get("page") is not None:
        return f"{src['filename']}, page {src['page']}"
    if src.get("start_sec") is not None:
        return f"{src['filename']}, {src['start_sec']:.0f} to {src.get('end_sec') or src['start_sec']:.0f} s"
    return src["filename"]


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """Stored text cut at a word boundary to fit Provision.excerpt (3.5)."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return cut[:space] if space > limit // 2 else cut


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def confidence(provisions: list[Provision], facts: list[Fact], best_score: float | None, rerank_high: float) -> Confidence:
    """R-24."""
    direct = sum(1 for p in provisions if p.relation == "direct")
    reasons = []
    if direct:
        reasons.append(_plural(direct, "direct provision"))
    elif provisions:
        reasons.append(_plural(len(provisions), "supporting provision") + ", none direct")
    else:
        reasons.append("no supporting provision")
    reasons.append(_plural(len(facts), "supporting case fact"))
    strong = best_score is not None and best_score >= rerank_high
    if direct and facts and strong:
        return Confidence(level="high", reasons=reasons + ["best legal rerank score at or above RERANK_HIGH"])
    if provisions:
        if direct and facts and not strong:
            reasons.append("best legal rerank score below RERANK_HIGH")
        return Confidence(level="medium", reasons=reasons)
    return Confidence(level="low", reasons=reasons)


async def _sources(deps: QueryDeps, state: AgentState, chunk_ids: list[str]) -> dict[str, dict]:
    if not chunk_ids or not state.get("case_id"):
        return {}
    graph = ScopedGraph(deps.neo4j, CaseScope(state["case_id"], state["user_id"]))
    rows = await graph.read(graph_queries.CASE_CHUNK_SOURCES, {"chunk_ids": chunk_ids})
    return {r["id"]: r for r in rows}


async def verifier(state: AgentState, config: RunnableConfig, *, deps: QueryDeps) -> dict:
    trace = copy.deepcopy(state["trace"])
    reply = state["analysis"]["reply"]
    labels = state["analysis"]["labels"]
    given_chunks = {c["id"]: c for c in state.get("case_chunks", [])}
    given_provisions = {p["id"]: p for p in state.get("legal_final", [])}
    dropped = 0

    with timed(trace, "verify"):
        wanted = [labels["chunks"].get(f["source"]) for f in reply["relevant_facts"]]
        sources = await _sources(deps, state, [c for c in labels["chunks"].values() if c in given_chunks])
        names = {label: chunk_name(sources[cid]) for label, cid in labels["chunks"].items()
                 if cid in sources and sources[cid].get("filename")}
        names.update({label: given_provisions[pid]["citation"] for label, pid in labels["provisions"].items()
                      if pid in given_provisions})
        facts: list[Fact] = []
        for f, chunk_id in zip(reply["relevant_facts"], wanted):
            src = sources.get(chunk_id) if chunk_id in given_chunks else None
            if src is None or src.get("filename") is None:
                dropped += 1
                continue
            # SC-17 (extended 2026-11-25): a label inside a fact is removed, not named,
            # because the fact's source is shown beside it. Checked after removal (SC-16).
            fact_text = replace_labels(f["fact"], {})
            if not fact_text or not fact_supported(fact_text, given_chunks[chunk_id]["text"]):  # SC-16
                dropped += 1
                continue
            facts.append(Fact(
                fact=fact_text, source_chunk_id=chunk_id,
                source=SourceRef(doc_id=src["doc_id"], filename=src["filename"], page=src.get("page"),
                                 start_sec=src.get("start_sec"), end_sec=src.get("end_sec")),
            ))

        provisions: list[Provision] = []
        cited: set[str] = set()
        for p in reply["applicable_provisions"]:
            pid = labels["provisions"].get(p["source"])
            item = given_provisions.get(pid) if pid else None
            if item is None:
                dropped += 1
                continue
            if pid in cited:  # the same provision twice: keep the first
                continue
            cited.add(pid)
            # SC-17 (extended 2026-11-25): labels in relevance are named like the prose
            # fields; if naming would pass the 500-character limit (3.5), they are removed.
            relevance = replace_labels(p["relevance"], names)
            if len(relevance) > RELEVANCE_MAX_CHARS:
                relevance = replace_labels(p["relevance"], {})
            provisions.append(Provision(
                provision_id=pid, act=item["act_short_title"], act_as_on=item["act_as_on"],
                citation=item["citation"], status=item["status"], relation=p["relation"],
                relevance=relevance, excerpt=excerpt(item["text"]),
            ))

    model_gaps = [replace_labels(g, names) for g in reply["gaps_or_conflicts"]]  # SC-17
    gaps = list(dict.fromkeys(trace["notes"] + [g for g in model_gaps if g]))
    if not provisions and state["intent"] in NO_PROVISION_NOTE_INTENTS and NO_PROVISION_NOTE not in gaps:
        gaps.append(NO_PROVISION_NOTE)  # R-23, SC-18
    best = max((p["score"] for p in state.get("legal_final", [])), default=None)  # R-24
    trace["counts"]["dropped_citations"] = dropped

    analysis = {
        "summary": replace_labels(reply["summary"], names),  # SC-17
        "relevant_facts": [f.model_dump(mode="json") for f in facts],
        "applicable_provisions": [p.model_dump(mode="json") for p in provisions],
        "precedents": [],
        "analysis": replace_labels(reply["analysis"], names),  # SC-17
        "gaps_or_conflicts": gaps,
        "confidence": confidence(provisions, facts, best, deps.rerank_high).model_dump(),
        "disclaimer": DISCLAIMER,  # R-25
    }
    return {
        "analysis": analysis,
        "trace": trace,
        "messages": [
            {"role": "user", "content": state["original_query"]},
            {"role": "assistant", "content": replace_labels(reply["summary"], names)},
        ],
    }
