"""tests/eval/run_eval.py: question-file validation, recall at 8 and the report gates."""

from __future__ import annotations

from tests.eval.run_eval import check_questions, report, score

KNOWN = {"sec:cpa-2019:69", "sec:cpa-2019:35", "cl:cpa-2019:39:1", "sec:cpa-2019:39"}
CASES = {"quickkart", "velora_car"}


def _q(i: int, category: str, **kw) -> dict:
    base = {"id": f"q{i:02d}", "category": category, "case": "quickkart", "mode": "auto",
            "question": "What can I claim?", "expected_intent": "CASE_AND_LEGAL",
            "expected_provisions": ["sec:cpa-2019:35"], "notes": ""}
    if category == "legal_only":
        base.update(case=None, expected_intent="LEGAL_ONLY")
    if category in ("conversational", "out_of_scope"):
        base.update(case=None, expected_intent="CONVERSATIONAL", expected_provisions=[])
    return {**base, **kw}


def _valid_set() -> list[dict]:
    return ([_q(i, "case_and_legal") for i in range(20)] + [_q(20 + i, "legal_only") for i in range(5)]
            + [_q(25 + i, "conversational") for i in range(3)] + [_q(28 + i, "out_of_scope") for i in range(2)])


def test_valid_set_passes() -> None:
    assert check_questions(_valid_set(), KNOWN, CASES) == []


def test_problems_are_reported() -> None:
    items = _valid_set()
    items[0] = _q(0, "case_and_legal", expected_provisions=["cl:cpa-2019:2:47-p1"])  # split part
    items[1] = _q(1, "case_and_legal", expected_provisions=["sec:cpa-2019:999"])  # unknown
    items[2] = _q(2, "case_and_legal", case="swiftdrop_courier")  # known key, not ingested
    items[3] = _q(3, "case_and_legal", case=None)
    items[4] = _q(4, "case_and_legal", acceptable_provisions=["sec:cpa-2019:35"])  # both lists
    items[5] = _q(1, "case_and_legal")  # duplicate id
    items[25] = _q(25, "conversational", expected_provisions=["sec:cpa-2019:69"])
    items.append({"id": "bad"})
    problems = "\n".join(check_questions(items, KNOWN, CASES))
    for fragment in ("split part", "sec:cpa-2019:999 is not a Section", "not in the database",
                     "case_and_legal needs a case", "both expected and acceptable", "duplicate id",
                     "conversational questions have no expected provisions", "missing keys"):
        assert fragment in problems, fragment


def test_minimum_counts() -> None:
    problems = check_questions(_valid_set()[:19], KNOWN, CASES)
    assert any("at least 20 case_and_legal" in p for p in problems)
    assert any("at least 5 legal_only" in p for p in problems)


STATE = {
    "intent": "CASE_AND_LEGAL",
    "legal_final": [{"id": f"sec:x:{i}"} for i in range(7)] + [{"id": "sec:cpa-2019:39"}, {"id": "sec:cpa-2019:35"}],
    "case_chunks": [{"id": "chk:a:0000"}],
    "analysis": {"applicable_provisions": [{"provision_id": "sec:cpa-2019:39"}],
                 "relevant_facts": [{"source_chunk_id": "chk:a:0000"}], "confidence": {"level": "medium"}},
    "trace": {"counts": {"dropped_citations": 1}, "llm": {"prompt_tokens": 10, "completion_tokens": 2}},
}


def test_recall_counts_only_the_first_8_and_exact_ids() -> None:
    q = _q(0, "case_and_legal", expected_provisions=["sec:cpa-2019:39", "sec:cpa-2019:35", "cl:cpa-2019:39:1"])
    s = score(q, STATE)
    # sec 39 is 8th (counted), sec 35 is 9th (not counted), clause 39(1) is not matched by sec 39.
    assert s["hits"] == ["sec:cpa-2019:39"] and s["recall"] == 1 / 3
    assert s["cited_recall"] == 1 / 3  # sec 39 is the only expected id the answer cites
    assert s["unverified"] == 0 and s["intent_ok"] is True


def test_unverified_citations_are_counted() -> None:
    bad = {**STATE, "analysis": {**STATE["analysis"],
                                 "applicable_provisions": [{"provision_id": "sec:cpa-2019:107"}],
                                 "relevant_facts": [{"source_chunk_id": "chk:other:0001"}]}}
    assert score(_q(0, "case_and_legal"), bad)["unverified"] == 2


def test_report_gates_and_p95() -> None:
    results = [{"id": f"q{i}", "error": None, "recall": 1.0 if i < 17 else 0.0, "cited_recall": 0.5, "unverified": 0,
                "intent_ok": True, "latency_ms": 1000 * (i + 1), "tokens": {"prompt_tokens": 1, "completion_tokens": 1}}
               for i in range(20)]
    r = report(results + [{"id": "q99", "error": "QUERY_TIMEOUT"}])
    assert r["recall_at_8"] == 0.85 and r["recall_gate_0_80"] is True and r["cited_recall"] == 0.5
    assert r["p95_latency_ms"] == 19000 and r["latency_gate_20s"] is True
    assert r["errors"] == {"q99": "QUERY_TIMEOUT"} and r["unverified_citations"] == 0
