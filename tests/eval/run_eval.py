"""Phase 4 evaluation: provision recall at 8, unverified citations, p95 latency (05 Phase 4).

    python -m tests.eval.run_eval --list [--out FILE]   every provision id, as a labelling aid
    python -m tests.eval.run_eval --check               validate tests/eval/questions.jsonl
    python -m tests.eval.run_eval [--results FILE]      scored run (resumable)

Format and method: 'Evaluation set' in docs/README.md. Questions are written by the operator;
this script never suggests provision ids from the retriever.

Scored run:
- Each question runs through the real query graph (run_query) against the Legal KB and the
  synthetic eval cases in the configured database.
- SC-14 c1: the run waits PACE_S between questions so the Groq per-minute budget refills,
  and the wait is not part of the measured latency.
- O-6: results are appended to the results file one question at a time; a re-run skips the
  ids already there, so a run cut off by the daily token limit resumes the next day.
- Recall at 8: |expected ∩ ids of the first 8 legal_final items| / |expected|, averaged over
  questions with a non-empty expected list. A Section does not count for one of its
  Clauses (or the reverse) unless the labeller listed it in acceptable_provisions.
- Unverified citations (hard gate): a provision in the answer that is not in legal_final, or
  a fact whose chunk is not in case_chunks. The verifier should make this always 0.
- Cited recall (reported, not a gate; O-9): |expected ∩ provisions cited in the answer| /
  |expected|. Retrieval recall can be 1.0 while the answer relies on the wrong provision.
- R-04: the report holds ids, counts and timings; question text stays in the questions file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

from app.agents.graph import build_graph, run_query
from app.agents.nodes.graph_expander import ConceptIndex
from app.agents.state import QueryDeps
from app.config import get_settings
from app.errors import AppError
from app.services.llm.groq_client import GroqClient
from app.services.retrieval import graph_queries
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.neo4j_service import Neo4jService
from app.services.retrieval.ranking_service import RankingService

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.jsonl"
DEFAULT_RESULTS = HERE / "results.jsonl"

# Fixture key -> case title in the database (created through the API, docs/README.md).
CASE_TITLES = {
    "quickkart": "QuickKart refrigerator (synthetic)",
    "velora_car": "eval: velora_car (synthetic)",
    "swiftdrop_courier": "eval: swiftdrop_courier (synthetic)",
    "bazaarkart_earbuds": "eval: bazaarkart_earbuds (synthetic)",
}
CATEGORIES = {
    "case_and_legal": "CASE_AND_LEGAL", "legal_only": "LEGAL_ONLY",
    "conversational": "CONVERSATIONAL", "out_of_scope": None,
}
MIN_COUNTS = {"case_and_legal": 20, "legal_only": 5}  # 05 Phase 4
MIN_OTHER = 5  # conversational + out_of_scope
MODES = {"auto", "case_and_legal", "legal_only", "case_only"}
REQUIRED = ("id", "category", "case", "mode", "question", "expected_intent", "expected_provisions")
INTENTS = {"CONVERSATIONAL", "CASE_ONLY", "LEGAL_ONLY", "CASE_AND_LEGAL"}
RECALL_AT = 8
PACE_S = 60  # SC-14 c1
_PART = re.compile(r"-p[0-9]+$")


def _service() -> Neo4jService:
    s = get_settings()
    return Neo4jService(s.NEO4J_URI, s.NEO4J_USER, s.NEO4J_PASSWORD.get_secret_value(), s.NEO4J_DATABASE)


def _section_key(number: str) -> tuple[int, str]:
    m = re.match(r"([0-9]+)(.*)", number or "")
    return (int(m.group(1)), m.group(2)) if m else (10**6, number or "")


def _label_key(label: str) -> tuple[int, str]:
    return _section_key(label) if label and label[0].isdigit() else (0, label or "")


# ---------------------------------------------------------------------------------------
# --list


def provision_index(neo4j: Neo4jService) -> str:
    rows = neo4j.read_sync(graph_queries.LEGAL_PROVISION_INDEX)
    rows.sort(key=lambda r: (r["act_node_id"], _section_key(r["number"])))
    lines = ["# Provision ids in the Legal KB (labelling aid; regenerate with --list)", ""]
    current = None
    for r in rows:
        if r["act_node_id"] != current:
            current = r["act_node_id"]
            lines += ["", f"## {r['act_short_title']} ({r['act_kind']})", ""]
        unit = "Rule" if r["act_kind"] in ("rule", "regulation") else "Section"
        lines.append(f"- `{r['id']}` {unit} {r['number']}. {r['title'] or ''}")
        for c in sorted(r["clauses"], key=lambda c: _label_key(c["label"])):
            what = f"'{c['term']}'" if c.get("term") else " ".join((c.get("text") or "").split())
            lines.append(f"  - `{c['id']}` ({c['label']}) {what}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------
# --check


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} does not exist yet: write the questions first (format in docs/README.md)")
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"line {n}: not valid JSON ({exc.msg})") from exc
    return items


def check_questions(items: list[dict], known_ids: set[str], known_cases: set[str]) -> list[str]:
    """Problems with the question file; an empty list means it is valid."""
    problems: list[str] = []
    seen: set[str] = set()
    counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    for n, q in enumerate(items, start=1):
        where = f"line {n} ({q.get('id', '?')})"
        missing = [k for k in REQUIRED if k not in q]
        if missing:
            problems.append(f"{where}: missing keys {missing}")
            continue
        if q["id"] in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(q["id"])
        cat = q["category"]
        if cat not in CATEGORIES:
            problems.append(f"{where}: category must be one of {sorted(CATEGORIES)}")
            continue
        counts[cat] += 1
        if q["mode"] not in MODES:
            problems.append(f"{where}: mode must be one of {sorted(MODES)}")
        if q["expected_intent"] not in INTENTS:
            problems.append(f"{where}: expected_intent must be one of {sorted(INTENTS)}")
        if not isinstance(q["question"], str) or not 3 <= len(q["question"]) <= 2000:
            problems.append(f"{where}: question must be 3 to 2,000 characters (3.4)")
        if q["case"] is not None and q["case"] not in CASE_TITLES:
            problems.append(f"{where}: unknown case {q['case']!r}; known: {sorted(CASE_TITLES)}")
        elif q["case"] is not None and q["case"] not in known_cases:
            problems.append(f"{where}: case {q['case']!r} is not in the database (not ingested?)")
        if cat == "case_and_legal" and q["case"] is None:
            problems.append(f"{where}: case_and_legal needs a case")
        expected = q["expected_provisions"]
        acceptable = q.get("acceptable_provisions", [])
        if not isinstance(expected, list) or not isinstance(acceptable, list):
            problems.append(f"{where}: expected_provisions and acceptable_provisions must be lists")
            continue
        if cat in ("conversational", "out_of_scope") and expected:
            problems.append(f"{where}: {cat} questions have no expected provisions")
        if cat in ("case_and_legal", "legal_only") and not expected:
            problems.append(f"{where}: {cat} questions need at least one expected provision")
        for pid in expected + acceptable:
            if _PART.search(pid):
                problems.append(f"{where}: {pid} is a split part; label its Clause or Section (A-35)")
            elif pid not in known_ids:
                problems.append(f"{where}: {pid} is not a Section or Clause in the Legal KB")
        if set(expected) & set(acceptable):
            problems.append(f"{where}: an id is both expected and acceptable")
    for cat, minimum in MIN_COUNTS.items():
        if counts[cat] < minimum:
            problems.append(f"need at least {minimum} {cat} questions, have {counts[cat]}")
    if counts["conversational"] + counts["out_of_scope"] < MIN_OTHER:
        problems.append(f"need at least {MIN_OTHER} conversational or out_of_scope questions, "
                        f"have {counts['conversational'] + counts['out_of_scope']}")
    return problems


def retrieval_warnings(items: list[dict], retrievable: set[str]) -> list[str]:
    """Expected ids that retrieval cannot return directly (not a LegalUnit and not split into
    parts). They can still arrive through graph expansion, so this warns, never fails: usually
    the Section is the unit and the label should be the Section id (A-11)."""
    warnings = []
    for q in items:
        for pid in q.get("expected_provisions", []):
            if isinstance(pid, str) and pid not in retrievable and not _PART.search(pid):
                warnings.append(f"{q.get('id', '?')}: {pid} is not a retrieval unit; it can only "
                                f"arrive through graph expansion (label its Section if that is the unit)")
    return warnings


def case_ids(neo4j: Neo4jService, user_id: str) -> dict[str, str]:
    """Fixture key -> case_id, for the eval cases that exist in the database."""
    rows, cursor = [], None
    while True:
        page = neo4j.read_sync(graph_queries.CASE_LIST, {"user_id": user_id, "limit": 200, "cursor": cursor})
        rows += page
        if len(page) < 200:
            break
        cursor = page[-1]["case_id"]
    by_title = {r["title"]: r["case_id"] for r in rows if r["status"] == "active"}
    return {key: by_title[title] for key, title in CASE_TITLES.items() if title in by_title}


# ---------------------------------------------------------------------------------------
# Scoring


def score(q: dict, state: dict) -> dict:
    final_ids = [p["id"] for p in state.get("legal_final", [])][:RECALL_AT]
    expected = q["expected_provisions"]
    hits = [pid for pid in expected if pid in final_ids]
    given_provisions = {p["id"] for p in state.get("legal_final", [])}
    given_chunks = {c["id"] for c in state.get("case_chunks", [])}
    analysis = state["analysis"]
    unverified = sum(1 for p in analysis["applicable_provisions"] if p["provision_id"] not in given_provisions)
    unverified += sum(1 for f in analysis["relevant_facts"] if f["source_chunk_id"] not in given_chunks)
    cited = [p["provision_id"] for p in analysis["applicable_provisions"]]
    return {
        "recall": len(hits) / len(expected) if expected else None,
        "cited_recall": sum(pid in cited for pid in expected) / len(expected) if expected else None,
        "hits": hits,
        "missed": [pid for pid in expected if pid not in final_ids],
        "acceptable_hits": [pid for pid in q.get("acceptable_provisions", []) if pid in final_ids],
        "legal_final": final_ids,
        "cited": cited,
        "facts": len(analysis["relevant_facts"]),  # case facts the answer used (SC-23 check)
        "unverified": unverified,
        "intent": state["intent"],
        "intent_ok": state["intent"] == q["expected_intent"],
        "confidence": analysis["confidence"]["level"],
        "dropped_citations": state["trace"]["counts"]["dropped_citations"],
        "tokens": state["trace"]["llm"],
    }


def report(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("error") is None]
    recalls = [r["recall"] for r in ok if r["recall"] is not None]
    latencies = sorted(r["latency_ms"] for r in ok)
    p95 = latencies[max(0, -(-95 * len(latencies) // 100) - 1)] if latencies else None
    return {
        "questions": len(results),
        "errors": {r["id"]: r["error"] for r in results if r.get("error")},
        "recall_at_8": round(statistics.mean(recalls), 3) if recalls else None,
        "recall_gate_0_80": bool(recalls) and statistics.mean(recalls) >= 0.80,
        "cited_recall": round(statistics.mean(cited), 3) if (cited := [r["cited_recall"] for r in ok if r.get("cited_recall") is not None]) else None,
        "unverified_citations": sum(r["unverified"] for r in ok),
        "intent_accuracy": round(sum(r["intent_ok"] for r in ok) / len(ok), 3) if ok else None,
        "p95_latency_ms": p95,
        "latency_gate_20s": p95 is not None and p95 <= 20000,
        "tokens": {k: sum(r["tokens"][k] for r in ok) for k in ("prompt_tokens", "completion_tokens")},
    }


async def run(results_path: Path, only: set[str] | None) -> dict:
    s = get_settings()
    neo4j = _service()
    questions = load_questions()
    known_cases = case_ids(neo4j, s.DEFAULT_USER_ID)
    done = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("error") is None:
                    done[r["id"]] = r
    deps = QueryDeps(
        neo4j=neo4j, embeddings=EmbeddingService(s.OLLAMA_URL, s.EMBED_MODEL, s.EMBED_DIM),
        llm=GroqClient(s.GROQ_API_KEY.get_secret_value(), s.GROQ_MODEL),
        ranker=RankingService(s.FLASHRANK_MODEL), concepts=await ConceptIndex.load(neo4j),
        rerank_high=s.RERANK_HIGH,
    )
    graph = build_graph(deps)
    todo = [q for q in questions if q["id"] not in done and (only is None or q["id"] in only)]
    print(f"{len(done)} already scored, {len(todo)} to run")
    for i, q in enumerate(todo):
        if i:
            await asyncio.sleep(PACE_S)
        start = time.perf_counter()
        try:
            state = await run_query(
                graph, deps, user_id=s.DEFAULT_USER_ID,
                case_id=known_cases.get(q["case"]) if q["case"] else None,
                question=q["question"], thread_id=f"eval-{q['id']}", mode=q["mode"],
            )
            result = {"id": q["id"], "category": q["category"], "error": None,
                      "latency_ms": int((time.perf_counter() - start) * 1000), **score(q, state)}
        except AppError as exc:
            result = {"id": q["id"], "category": q["category"], "error": exc.code.value}
        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")
        print(f"{q['id']}: {result.get('error') or 'recall=%s latency=%dms' % (result['recall'], result['latency_ms'])}")
        if result.get("error") is None:
            done[q["id"]] = result
    await deps.embeddings.aclose()
    neo4j.close()
    return report([done[q["id"]] for q in questions if q["id"] in done])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="print every provision id")
    group.add_argument("--check", action="store_true", help="validate questions.jsonl")
    parser.add_argument("--out", type=Path, help="with --list: write to this file")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--only", help="comma-separated question ids to run")
    args = parser.parse_args(argv)

    if args.list:
        neo4j = _service()
        try:
            text = provision_index(neo4j)
        finally:
            neo4j.close()
        if args.out:
            args.out.write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            sys.stdout.write(text)
        return 0

    if args.check:
        neo4j = _service()
        try:
            items = load_questions()
            ids = sorted({p for q in items for p in q.get("expected_provisions", []) + q.get("acceptable_provisions", [])
                          if isinstance(p, str)})
            known = {r["id"] for r in neo4j.read_sync(graph_queries.LEGAL_PROVISIONS_EXIST, {"ids": ids})}
            retrievable = {r["id"] for r in neo4j.read_sync(graph_queries.LEGAL_RETRIEVABLE, {"ids": ids})
                           if r["unit"] or r["split"]}
            cases = set(case_ids(neo4j, get_settings().DEFAULT_USER_ID))
        finally:
            neo4j.close()
        problems = check_questions(items, known, cases)
        counts = {c: sum(1 for q in items if q.get("category") == c) for c in CATEGORIES}
        print(f"{len(items)} questions: {counts}")
        for w in retrieval_warnings(items, retrievable):
            print("WARNING:", w)
        for p in problems:
            print("PROBLEM:", p)
        print("OK" if not problems else f"{len(problems)} problem(s)")
        return 0 if not problems else 1

    only = set(args.only.split(",")) if args.only else None
    print(json.dumps(asyncio.run(run(args.results, only)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
