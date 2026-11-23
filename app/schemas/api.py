"""Request and response models for every endpoint in 3.4.

Conventions (3.1): API-facing case, document and job ids are bare UUID v4 strings; provision
ids are used exactly as stored. Timestamps are ISO 8601 UTC with a trailing Z. A user_id in
a request body is ignored (3.2): request models drop unknown fields.

Where 3.4 does not list a response's fields, the models below follow the assumptions
recorded in docs/README.md (A-4 to A-7).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Annotated, Generic, Literal, Optional, TypeVar
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
)

from app.errors import ErrorCode
from app.schemas.graph_schema import slug


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# 3.1: ISO 8601 UTC with a trailing Z.
UtcDatetime = Annotated[
    AwareDatetime, AfterValidator(_to_utc), PlainSerializer(_iso_z, return_type=str)
]


class RequestModel(BaseModel):
    # 3.2: a user_id (or any other unknown field) in a request is ignored.
    model_config = ConfigDict(extra="ignore")


# Lists (3.1)

T = TypeVar("T")


class ListParams(RequestModel):
    limit: int = Field(default=25, ge=1, le=100)
    cursor: Optional[str] = Field(default=None, min_length=1, max_length=512)


class Page(BaseModel, Generic[T]):
    items: list[T]
    next_cursor: Optional[str] = None


# Error envelope (3.3)


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    details: dict = {}
    request_id: Optional[str] = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


# GET /health (3.4)

CheckStatus = Literal["ok", "fail"]


class HealthChecks(BaseModel):
    neo4j: CheckStatus
    indexes: CheckStatus
    embedding_meta: CheckStatus
    ollama: CheckStatus


class HealthResponse(BaseModel):
    # 200 with 'ok'; 503 with the same shape (A-4: status is 'fail' on 503).
    status: CheckStatus
    checks: HealthChecks


# Cases (3.4)


class CaseCreateRequest(RequestModel):
    title: str = Field(min_length=1, max_length=200)


class CaseResponse(BaseModel):
    # Returned by POST (201), GET and DELETE (202, status 'deleting'; A-5).
    case_id: UUID
    title: str
    status: Literal["active", "deleting"]
    created_at: UtcDatetime


# Documents (3.4). The upload itself is multipart: field file (required) and language.

ISO_639_1 = re.compile(r"^[a-z]{2}$")


def _iso_639_1(value: Optional[str]) -> Optional[str]:
    if value is not None and not ISO_639_1.fullmatch(value):
        raise ValueError("language must be an ISO 639-1 code, for example 'en' or 'hi'")
    return value


LanguageHint = Annotated[Optional[str], AfterValidator(_iso_639_1)]

ProcessingStatus = Literal["queued", "processing", "ready", "ready_with_warnings", "failed"]
MediaType = Literal["pdf", "image", "audio", "text"]


class DocumentUploadResponse(BaseModel):
    # 202 for a new document; 200 with job_id null and duplicate true for a repeat (E-09).
    doc_id: UUID
    job_id: Optional[UUID]
    processing_status: ProcessingStatus
    duplicate: bool


class DocumentResponse(BaseModel):
    # A-6: the CaseDocument properties from 2.3, without user_id and stored_path.
    doc_id: UUID
    case_id: UUID
    filename: str
    media_type: MediaType
    mime: str
    size_bytes: int
    sha256: str
    processing_status: ProcessingStatus
    page_count: Optional[int] = None
    duration_sec: Optional[float] = None
    language: Optional[str] = None
    error_code: Optional[str] = None
    uploaded_at: UtcDatetime


# Jobs (3.4)

JobType = Literal["case_document", "legal_ingest"]
JobStatus = Literal[
    "queued", "running", "succeeded", "succeeded_with_warnings", "failed", "cancelled"
]
CASE_DOCUMENT_STAGES = (
    "stored",
    "text_extraction",
    "chunking",
    "embedding",
    "entity_extraction",
    "entity_resolution",
    "legal_linking",
    "done",
)
LEGAL_INGEST_STAGES = ("text_extraction", "parsing", "embedding", "writing", "verification", "done")


class JobError(BaseModel):
    code: str
    message: str


class JobResponse(BaseModel):
    job_id: UUID
    type: JobType
    status: JobStatus
    stage: Optional[str] = None
    progress: int = Field(ge=0, le=100)
    warnings: list[str] = []
    error: Optional[JobError] = None
    created_at: UtcDatetime
    started_at: Optional[UtcDatetime] = None
    finished_at: Optional[UtcDatetime] = None


class JobAccepted(BaseModel):
    # 202 from POST /v1/admin/legal/ingest.
    job_id: UUID


# POST /v1/query (3.4). The 200 body is AnalysisResponse (app/schemas/output.py).

QueryMode = Literal["auto", "case_and_legal", "legal_only", "case_only"]


class QueryOptions(RequestModel):
    max_case_chunks: int = Field(default=5, ge=1, le=10)
    max_provisions: int = Field(default=8, ge=1, le=12)
    include_trace: bool = False


class QueryRequest(RequestModel):
    question: str = Field(min_length=3, max_length=2000)
    case_id: Optional[UUID] = None
    thread_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    mode: QueryMode = "auto"
    options: QueryOptions = QueryOptions()

    @model_validator(mode="after")
    def _case_id_required(self) -> QueryRequest:
        # case_id is required unless mode is legal_only; null with auto behaves as
        # legal_only (handled by the agent).
        if self.case_id is None and self.mode in ("case_and_legal", "case_only"):
            raise ValueError(f"case_id is required when mode is {self.mode}")
        return self


# GET /v1/legal/provisions/{provision_id} (3.4)


class ProvisionRef(BaseModel):
    id: str
    citation: str


class DefinitionRef(BaseModel):
    term: str
    id: str


class ProvisionResponse(BaseModel):
    provision_id: str
    citation: str
    text: str
    status: Literal["in_force", "repealed", "omitted"]
    in_force_from: Optional[date] = None
    amendment_notes: list[str] = []
    parent: Optional[ProvisionRef] = None
    refers_to: list[ProvisionRef] = []
    referenced_by: list[ProvisionRef] = []
    definitions: list[DefinitionRef] = []


# Admin legal ingest (3.4). The metadata form field is a JSON string with these keys.


SectionNumber = Annotated[str, Field(pattern=r"^[0-9]{1,3}[A-Z]?$")]


class IssuedUnder(RequestModel):
    # SC-6: a rule may be issued under several sections (e.g. 34, 47 and 58).
    act_id: str = Field(min_length=1)
    sections: list[SectionNumber] = Field(min_length=1)


class PrescribesFor(RequestModel):
    # SC-6: creates PRESCRIBES_FOR from the rule's Section to the parent Act's Section.
    rule: SectionNumber
    section: SectionNumber
    note: Optional[str] = Field(default=None, max_length=300)


class Amends(RequestModel):
    # SC-9: this notification amends the rules with this act_id.
    act_id: str = Field(min_length=1)
    effective_on: Optional[date] = None  # defaults to as_on_date


class LegalIngestMetadata(RequestModel):
    # A-7: required unless 3.4 marks the key "or null"; replace defaults to false.
    act_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=500)
    short_title: str = Field(min_length=1, max_length=300)
    year: int = Field(ge=1800, le=2100)
    number: str = Field(min_length=1, max_length=50)
    jurisdiction: Literal["IN"]
    as_on_date: date
    kind: Literal["act", "rule", "regulation"]
    issued_under: Optional[IssuedUnder] = None
    prescribes_for: Optional[list[PrescribesFor]] = None
    amends: Optional[Amends] = None
    expected_sections: Optional[int] = Field(default=None, ge=1)
    replace: bool = False

    @model_validator(mode="after")
    def _check(self) -> LegalIngestMetadata:
        if self.act_id != slug(self.act_id):
            raise ValueError("act_id must be a slug, for example cpa-2019")
        # 1.4 A1: rules carry issued_under (act_id and sections).
        if self.kind == "rule" and self.issued_under is None:
            raise ValueError("issued_under is required when kind is rule")
        if self.prescribes_for and self.issued_under is None:
            raise ValueError("prescribes_for needs issued_under")
        if self.amends and (self.kind != "rule" or self.prescribes_for):
            raise ValueError("amends is for rule notifications without prescribes_for (SC-9)")
        if self.amends and self.amends.act_id == self.act_id:
            raise ValueError("a notification cannot amend itself")
        return self


class ActResponse(BaseModel):
    # A-7: an item of GET /v1/admin/legal/acts; the Act properties from 2.3.
    act_id: str
    kind: Literal["act", "rule", "regulation"]
    title: str
    short_title: str
    year: Optional[int] = None
    number: Optional[str] = None
    jurisdiction: Optional[str] = None
    as_on_date: Optional[date] = None
    status: Optional[Literal["in_force", "repealed"]] = None
    external: bool = False
    issued_under_act_id: Optional[str] = None
    issued_under_section: Optional[list[str]] = None
