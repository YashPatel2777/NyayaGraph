"""SC-24: Streamlit UI smoke tests with AppTest and a fake API client (no server)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from ui.api_client import ApiError  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py")
CASE = {"case_id": "11111111-1111-4111-8111-111111111111", "title": "Phone case",
        "status": "active", "created_at": "2026-11-25T10:00:00Z"}
ANSWER = {
    "request_id": "r", "case_id": CASE["case_id"], "thread_id": "t-9", "intent": "CASE_AND_LEGAL",
    "summary": "The phone was defective.",
    "relevant_facts": [{"fact": "Green lines appeared on 6 January.", "source_chunk_id": "chk:x:0001",
                        "source": {"doc_id": "d", "filename": "chat.txt", "page": None,
                                   "start_sec": None, "end_sec": None}}],
    "applicable_provisions": [{"provision_id": "cl:cpa-2019:2:10", "act": "Consumer Protection Act, 2019",
                               "act_as_on": "2026-09-01", "citation": "Section 2(10), Consumer Protection Act, 2019",
                               "status": "in_force", "relation": "definition",
                               "relevance": "Defines defect.", "excerpt": "defect means any fault"}],
    "precedents": [], "analysis": "See ![x](http://evil.example/leak) for details.",
    "gaps_or_conflicts": [], "confidence": {"level": "medium", "reasons": ["1 direct provision"]},
    "disclaimer": "This is an automated analysis. It is not legal advice.", "trace": None,
}


class FakeClient:
    def __init__(self, *, healthy=True, cases=(CASE,), query_error=None):
        self.healthy, self.cases, self.query_error = healthy, list(cases), query_error
        self.queries: list[dict] = []
        self.created: list[str] = []

    def health(self):
        return {"status": "ok" if self.healthy else "fail",
                "checks": {"neo4j": "ok", "indexes": "ok", "embedding_meta": "ok",
                           "ollama": "ok" if self.healthy else "fail"}}

    def list_cases(self):
        return self.cases

    def create_case(self, title):
        self.created.append(title)
        case = dict(CASE, case_id="22222222-2222-4222-8222-222222222222", title=title)
        self.cases.append(case)
        return case

    def list_documents(self, case_id):
        return [{"doc_id": "d", "case_id": case_id, "filename": "chat.txt", "media_type": "text",
                 "mime": "text/plain", "size_bytes": 2048, "sha256": "x", "processing_status": "ready",
                 "page_count": None, "duration_sec": None, "language": None, "error_code": None,
                 "uploaded_at": "2026-11-25T10:01:00Z"}]

    def get_job(self, job_id):
        raise AssertionError("no jobs in these tests")

    def query(self, question, **kw):
        self.queries.append({"question": question, **kw})
        if self.query_error:
            raise self.query_error
        return ANSWER

    def get_provision(self, pid):
        raise AssertionError("not opened in these tests")


def run(fake: FakeClient) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["api_client"] = fake
    return at.run()


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.error] + [w.value for w in at.warning] + [s.value for s in at.success]
    return "\n".join(str(p) for p in parts)


def test_renders_with_case_and_health():
    at = run(FakeClient())
    assert not at.exception
    titles = [t.value for t in at.title]
    assert "NyayaGraph" in titles and "Phone case" in titles
    assert any("Backend: all systems ok" == c.value for c in at.caption)


def test_degraded_health_names_failing_check():
    at = run(FakeClient(healthy=False))
    assert any("Backend degraded: ollama" == c.value for c in at.caption)


def test_ask_renders_verified_answer_and_keeps_thread():
    fake = FakeClient()
    at = run(fake)
    at.chat_input[0].set_value("Is this a defect?").run()
    assert not at.exception
    body = texts(at)
    assert "Section 2\\(10\\), Consumer Protection Act, 2019" in body or "Section 2(10)" in body
    assert "Green lines appeared on 6 January." in body
    assert "It is not legal advice." in body
    assert "Source: chat.txt" in body
    # R-06: model text is escaped, so an image link cannot fetch an external URL.
    assert "![x](http://evil.example/leak)" not in body
    assert fake.queries[0]["case_id"] == CASE["case_id"]
    assert fake.queries[0]["thread_id"] is None
    at.chat_input[0].set_value("And the refund?").run()
    assert fake.queries[1]["thread_id"] == "t-9"  # follow-up uses the returned thread


def test_legal_only_mode_sends_no_case():
    fake = FakeClient()
    at = run(fake)
    at.sidebar.selectbox(key="mode").set_value("legal_only").run()
    at.chat_input[0].set_value("What is a consumer?").run()
    assert fake.queries[0]["case_id"] is None
    assert fake.queries[0]["mode"] == "legal_only"


def test_api_error_is_shown_with_code():
    fake = FakeClient(query_error=ApiError(504, "QUERY_TIMEOUT", "Query timed out.", {}, "r-7"))
    at = run(fake)
    at.chat_input[0].set_value("Is this a defect?").run()
    assert any("QUERY_TIMEOUT" in e.value and "r-7" in e.value for e in at.error)


def test_sliders_set_query_options_and_trace_is_always_requested():
    fake = FakeClient()
    at = run(fake)
    at.sidebar.slider(key="max_case_chunks").set_value(3)
    at.sidebar.slider(key="max_provisions").set_value(4).run()
    at.chat_input[0].set_value("Is this a defect?").run()
    q = fake.queries[0]
    assert (q["max_case_chunks"], q["max_provisions"], q["include_trace"]) == (3, 4, True)


def test_cooldown_note_after_a_large_answer_and_trace_hidden_by_default():
    fake = FakeClient()
    traced = dict(ANSWER, trace={"search_query": "s", "issue_statement": "i", "downgraded": False,
                                 "counts": {}, "timings_ms": {},
                                 "llm": {"prompt_tokens": 5300, "completion_tokens": 500}})
    fake.query = lambda question, **kw: (fake.queries.append(kw), traced)[1]
    at = run(fake)
    at.chat_input[0].set_value("Is this a defect?").run()
    at.run()  # the note's fragment refreshes every second; one more run stands in for that tick
    assert any("next question ready in about" in c.value for c in at.caption)
    assert not any(e.label == "Retrieval trace" for e in at.expander)
    at.sidebar.toggle(key="include_trace").set_value(True).run()
    assert any(e.label == "Retrieval trace" for e in at.expander)


def test_create_case_from_sidebar():
    fake = FakeClient(cases=())
    at = run(fake)
    at.sidebar.text_input[0].set_value("New dispute")
    at.sidebar.button[0].click().run()
    assert fake.created == ["New dispute"]
    assert not at.exception
