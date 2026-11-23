"""Request and response models (3.4, 3.5, 2.5)."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.schemas.api import (
    CaseCreateRequest,
    CaseResponse,
    JobResponse,
    LegalIngestMetadata,
    ListParams,
    QueryRequest,
)
from app.schemas.extraction import ExtractionResult
from app.schemas.output import DISCLAIMER, AnalysisResponse

CASE = str(uuid.uuid4())


def test_query_defaults() -> None:
    q = QueryRequest(question="What can I claim?", case_id=CASE)
    assert q.mode == "auto"
    assert q.options.max_case_chunks == 5
    assert q.options.max_provisions == 8
    assert q.options.include_trace is False
    assert q.thread_id is None


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "hi"},
        {"question": "x" * 2001},
        {"question": "valid question", "thread_id": ""},
        {"question": "valid question", "thread_id": "t" * 65},
        {"question": "valid question", "mode": "other"},
        {"question": "valid question", "case_id": "not-a-uuid"},
        {"question": "valid question", "options": {"max_case_chunks": 11}},
        {"question": "valid question", "options": {"max_provisions": 0}},
        {"question": "valid question", "mode": "case_only"},
        {"question": "valid question", "mode": "case_and_legal"},
    ],
)
def test_query_rejects(payload) -> None:
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(payload)


@pytest.mark.parametrize("mode", ["auto", "legal_only"])
def test_query_without_case_id_allowed_for_auto_and_legal_only(mode) -> None:
    assert QueryRequest(question="valid question", mode=mode).case_id is None


def test_user_id_in_request_body_is_ignored() -> None:
    # 3.2, R-02
    q = QueryRequest.model_validate(
        {"question": "valid question", "user_id": "attacker", "case_id": CASE}
    )
    assert not hasattr(q, "user_id")
    assert "user_id" not in q.model_dump()
    c = CaseCreateRequest.model_validate({"title": "Fridge", "user_id": "attacker"})
    assert "user_id" not in c.model_dump()


def test_list_params() -> None:
    assert ListParams().limit == 25
    with pytest.raises(ValidationError):
        ListParams(limit=0)
    with pytest.raises(ValidationError):
        ListParams(limit=101)


def test_case_title_limits() -> None:
    with pytest.raises(ValidationError):
        CaseCreateRequest(title="")
    with pytest.raises(ValidationError):
        CaseCreateRequest(title="x" * 201)


def test_timestamps_serialise_as_utc_with_z() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    resp = CaseResponse(
        case_id=CASE, title="t", status="active",
        created_at=datetime(2026, 11, 20, 16, 0, tzinfo=ist),
    )
    assert resp.model_dump(mode="json")["created_at"] == "2026-11-20T10:30:00Z"


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        CaseResponse(case_id=CASE, title="t", status="active", created_at=datetime(2026, 1, 1))


def test_job_response_optional_timestamps() -> None:
    job = JobResponse(
        job_id=str(uuid.uuid4()), type="legal_ingest", status="queued", progress=0,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    dumped = job.model_dump(mode="json")
    assert dumped["started_at"] is None and dumped["error"] is None
    assert dumped["created_at"] == "2026-01-01T00:00:00Z"


def _meta(**over):
    base = dict(
        act_id="cpa-2019", title="The Consumer Protection Act, 2019",
        short_title="Consumer Protection Act, 2019", year=2019, number="35 of 2019",
        jurisdiction="IN", as_on_date="2026-09-01", kind="act", expected_sections=107,
    )
    return {**base, **over}


def test_legal_ingest_metadata_valid() -> None:
    m = LegalIngestMetadata.model_validate(_meta())
    assert m.as_on_date == date(2026, 9, 1)
    assert m.replace is False and m.issued_under is None


@pytest.mark.parametrize(
    "over",
    [
        {"act_id": "CPA 2019"},
        {"kind": "statute"},
        {"jurisdiction": "US"},
        {"kind": "rule"},  # issued_under missing
        {"as_on_date": "01-09-2026"},
        {"expected_sections": 0},
    ],
)
def test_legal_ingest_metadata_rejects(over) -> None:
    with pytest.raises(ValidationError):
        LegalIngestMetadata.model_validate(_meta(**over))


def test_rule_with_issued_under() -> None:
    m = LegalIngestMetadata.model_validate(
        _meta(act_id="cp-jurisdiction-rules-2021", kind="rule", expected_sections=None,
              issued_under={"act_id": "cpa-2019", "sections": ["34", "47", "58"]},
              prescribes_for=[{"rule": "3", "section": "34"}])
    )
    assert m.issued_under is not None and m.issued_under.sections == ["34", "47", "58"]
    assert m.prescribes_for[0].rule == "3"


def test_issued_under_needs_a_section_list() -> None:
    # SC-6: sections is a non-empty list.
    for bad in ({"act_id": "cpa-2019", "sections": []}, {"act_id": "cpa-2019", "section": "34"}):
        with pytest.raises(ValidationError):
            LegalIngestMetadata.model_validate(_meta(kind="rule", issued_under=bad))


def test_prescribes_for_needs_issued_under() -> None:
    with pytest.raises(ValidationError):
        LegalIngestMetadata.model_validate(_meta(prescribes_for=[{"rule": "3", "section": "34"}]))


def test_extraction_result_parses_spec_shape() -> None:
    r = ExtractionResult.model_validate(
        {
            "parties": [{"name": "A", "kind": "person", "role": "complainant"}],
            "events": [{"description": "bought", "date": "2025-03", "chunk_ids": ["c1"]}],
            "claims": [{"text": "defect", "claim_type": "defect", "chunk_ids": ["c1"]}],
            "evidence": [{"kind": "receipt", "description": "bill", "supports": [0], "chunk_ids": ["c1"]}],
        }
    )
    assert r.parties[0].aliases == []
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate({"claims": [{"text": "x", "claim_type": "fraud", "chunk_ids": []}]})
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate({"events": [{"description": "x"}]})  # chunk_ids required


def test_analysis_response_accepts_spec_example() -> None:
    example = {
        "request_id": "7c1e0b9a-4d52-4a47-9a52-1f0f4c8e6a10",
        "case_id": "3f0d2c1e-8a5b-4c63-9d11-2b7a6e5f4c90",
        "thread_id": "t-001",
        "intent": "CASE_AND_LEGAL",
        "summary": "s",
        "relevant_facts": [{
            "fact": "f", "source_chunk_id": "chk:9f1c2a7e-5b34-4d0a-8a6e-0c1d2e3f4a5b:0003",
            "source": {"doc_id": "b1a2c3d4-0000-4000-8000-123456789abc", "filename": "invoice.pdf",
                       "page": 1, "start_sec": None, "end_sec": None},
        }],
        "applicable_provisions": [{
            "provision_id": "sec:cpa-2019:39", "act": "Consumer Protection Act, 2019",
            "act_as_on": "2026-09-01", "citation": "Section 39, Consumer Protection Act, 2019",
            "status": "in_force", "relation": "direct", "relevance": "r", "excerpt": "e",
        }],
        "precedents": [],
        "analysis": "a",
        "gaps_or_conflicts": [],
        "confidence": {"level": "medium", "reasons": ["1 direct provision"]},
        "disclaimer": DISCLAIMER,
        "trace": None,
    }
    resp = AnalysisResponse.model_validate(example)
    assert resp.applicable_provisions[0].act_as_on == date(2026, 9, 1)
    assert DISCLAIMER.startswith("This is an automated analysis")


def test_amends_metadata() -> None:
    # SC-9
    base = _meta(act_id="test-amendment-2021", kind="rule", expected_sections=2,
                 issued_under={"act_id": "cpa-2019", "sections": ["101"]})
    m = LegalIngestMetadata.model_validate({**base, "amends": {"act_id": "test-rules-2020"}})
    assert m.amends.act_id == "test-rules-2020" and m.amends.effective_on is None
    for bad in (
        {**base, "kind": "act", "amends": {"act_id": "x"}},
        {**base, "amends": {"act_id": "test-amendment-2021"}},
        {**base, "amends": {"act_id": "x"}, "prescribes_for": [{"rule": "1", "section": "34"}]},
    ):
        with pytest.raises(ValidationError):
            LegalIngestMetadata.model_validate(bad)
