"""Response models for POST /v1/query (3.5).

The LLM returns only ids, relevance text and relation. The verifier hydrates citation,
status, act_as_on, excerpt and source from Neo4j (R-21), and code sets confidence (R-24) and
the disclaimer (R-25).
"""

from __future__ import annotations

from datetime import date
from typing import Final, Literal, Optional

from pydantic import BaseModel, Field

# R-25: a constant appended by code, never written by the LLM. Text as given in 3.5.
DISCLAIMER: Final = (
    "This is an automated analysis of the retrieved case material and statutory text. "
    "It is not legal advice. Verify every provision against the current official text "
    "and consult a qualified advocate before acting."
)

Intent = Literal["CONVERSATIONAL", "CASE_ONLY", "LEGAL_ONLY", "CASE_AND_LEGAL"]


class SourceRef(BaseModel):
    doc_id: str
    filename: str
    page: Optional[int] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None


class Fact(BaseModel):
    fact: str = Field(max_length=500)
    source_chunk_id: str
    source: SourceRef  # hydrated by code


class Provision(BaseModel):
    provision_id: str  # for example sec:cpa-2019:38 or cl:cpa-2019:2:7
    act: str  # short title, hydrated
    act_as_on: date  # hydrated from Act.as_on_date
    citation: str  # for example 'Section 38(2), Consumer Protection Act, 2019'
    status: Literal["in_force", "repealed", "omitted"]
    relation: Literal["direct", "referenced", "definition", "interpretation"]
    relevance: str = Field(max_length=500)  # the only LLM-written field
    excerpt: str = Field(max_length=600)  # copied from stored text by code


class Precedent(BaseModel):
    judgment_id: str
    citation: str
    relevance: str = Field(max_length=500)


class Confidence(BaseModel):
    level: Literal["low", "medium", "high"]
    reasons: list[str]


class Trace(BaseModel):
    search_query: str
    issue_statement: str
    downgraded: bool = False
    # case_candidates, case_final, legal_candidates, legal_final, expanded_nodes,
    # dropped_citations
    counts: dict[str, int]
    # planner, case, issue, legal, expand, rerank, generate, verify, total
    timings_ms: dict[str, int]
    llm: dict[str, int]  # prompt_tokens, completion_tokens


class AnalysisResponse(BaseModel):
    request_id: str
    case_id: Optional[str]
    thread_id: str
    intent: Intent
    summary: str = Field(max_length=800)
    relevant_facts: list[Fact]
    applicable_provisions: list[Provision]
    precedents: list[Precedent]  # empty until judgments are ingested
    analysis: str = Field(max_length=4000)
    gaps_or_conflicts: list[str]
    confidence: Confidence
    disclaimer: str  # constant, appended by code (R-25)
    trace: Optional[Trace] = None  # present only when options.include_trace is true


# ---------------------------------------------------------------------------------------
# LLM reply models for the query agent (1.4 C3, C9). Internal: never returned by the API.
# SC-22 removed the C5 issue-statement call. The responder sees chunks and provisions as
# short labels (C1, P1), which code maps back to ids (A-36); it writes only labels, relation
# and relevance for provisions (3.5).


class PlannerReply(BaseModel):
    intent: Intent
    # Empty is valid: the model leaves it empty for greetings; the planner then uses the question.
    search_query: str = Field(default="", max_length=500)


class FactReply(BaseModel):
    fact: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=20)  # chunk label, for example C2


class ProvisionReply(BaseModel):
    source: str = Field(min_length=1, max_length=20)  # provision label, for example P3
    relation: Literal["direct", "referenced", "definition", "interpretation"]
    relevance: str = Field(min_length=1, max_length=500)


class LegalAnalysisReply(BaseModel):
    summary: str = Field(min_length=1, max_length=800)
    relevant_facts: list[FactReply] = []
    applicable_provisions: list[ProvisionReply] = []
    analysis: str = Field(max_length=4000)
    gaps_or_conflicts: list[str] = []
