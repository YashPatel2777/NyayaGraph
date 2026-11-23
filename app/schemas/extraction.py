"""Case entity extraction models (2.5). Groq output for one batch of chunks (B6) is validated
against ExtractionResult. AgentState lives in app/agents/state.py (Phase 4)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ExtParty(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["person", "organisation"]
    role: Literal["complainant", "opposite_party", "witness", "other"]
    aliases: list[str] = []


class ExtEvent(BaseModel):
    description: str = Field(max_length=400)
    date: Optional[str] = None  # ISO 8601, partial allowed: YYYY, YYYY-MM, YYYY-MM-DD (R-17)
    location: Optional[str] = None
    participants: list[str] = []  # names exactly as they appear in parties[]
    chunk_ids: list[str]  # must be ids from the input batch (R-15)


class ExtClaim(BaseModel):
    text: str = Field(max_length=500)
    claim_type: Literal[
        "defect",
        "deficiency",
        "unfair_trade_practice",
        "overcharging",
        "product_liability",
        "unfair_contract",
        "other",
    ]
    asserted_by: Optional[str] = None  # a party name from parties[]
    chunk_ids: list[str]


class ExtEvidence(BaseModel):
    kind: Literal["document", "receipt", "message", "photo", "testimony", "other"]
    description: str = Field(max_length=400)
    supports: list[int] = []  # indexes into claims[]
    contradicts: list[int] = []
    chunk_ids: list[str]


class ExtractionResult(BaseModel):
    parties: list[ExtParty] = []
    events: list[ExtEvent] = []
    claims: list[ExtClaim] = []
    evidence: list[ExtEvidence] = []


# Replies of the two yes/no checks in case ingestion. Not API models; ids are re-checked in
# code against what was sent.


class LinkVerdict(BaseModel):
    unit_id: str
    applies: bool
    rationale: str = Field(default="", max_length=300)  # 2.4 ALLEGES_VIOLATION_OF


class LinkVerification(BaseModel):
    # B9: one verdict per candidate provision of one claim (A-27).
    verdicts: list[LinkVerdict] = []


class SameEntityVerdict(BaseModel):
    # R-14: the Groq yes/no for name similarity 80 to 91.
    same: bool
