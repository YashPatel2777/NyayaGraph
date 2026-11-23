"""Entity resolution inside one case (B7, R-14) and BEFORE edges (B8).

- R-14: names are normalised (lowercase; Mr, Mrs, Ms, Dr and M/s dropped; punctuation
  removed; spaces collapsed). Only parties of the same kind merge. rapidfuzz
  token_sort_ratio of 92 or more merges automatically, 80 to 91 asks Groq once (the answer
  is cached per pair), below 80 stays separate. A merged party keeps every alias and every
  MENTIONS edge.
- SC-13: when an organisation scores below 80, its name is compared again without company
  suffixes (Pvt, Ltd, Private Limited, Retail and similar). A score of 80 or more there goes
  to the Groq check; it never merges automatically.
- Parties already in the case (from other documents) take part, so one party has one node
  per case.
- A-24: events, claims and evidence are merged only when their normalised text (and, for
  events, date) is identical; R-14's fuzzy rule is about names.
- Ids are deterministic (2.1, R-16): pty {kind} {name}, evt {date} {description},
  clm {claim_type} {text}, evd {kind} {description}.
- BEFORE is computed in code from dates (B8), never by the LLM. A partial date is the
  interval it covers, and A comes BEFORE B only when A's interval ends before B's starts;
  each event links to the earliest later group, so the edges form a chain, not all pairs.
"""

from __future__ import annotations

import calendar
import re
import unicodedata
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import date

from rapidfuzz import fuzz

from app.schemas.graph_schema import (
    CLAIM_PREFIX,
    EVENT_PREFIX,
    EVIDENCE_PREFIX,
    PARTY_PREFIX,
    entity_id,
)

AUTO_MERGE = 92  # R-14
ADJUDICATE_MIN = 80  # R-14
# SC-13: legal-form and trade words dropped from organisation names for the second
# comparison. Matched as whole words after R-14 normalisation.
COMPANY_SUFFIXES = frozenset({
    "pvt", "private", "ltd", "limited", "llp", "inc", "incorporated", "co", "company",
    "corp", "corporation", "plc", "retail",
})

_HONORIFIC = re.compile(r"(?<![\w/])(?:m/s|mrs|mr|ms|dr)\.?(?![\w/])")
_SPACES = re.compile(r"\s+")
_ISO_PARTIAL = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


def normalise_name(name: str) -> str:
    """R-14 name normalisation."""
    text = _HONORIFIC.sub(" ", name.lower())
    # Punctuation and symbols by Unicode category, so Devanagari vowel signs (combining
    # marks, which are not \w) survive.
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return _SPACES.sub(" ", text).strip()


def core_name(normalised: str) -> str:
    """SC-13: an organisation's normalised name without company suffixes ('quickkart retail
    private limited' -> 'quickkart'). A name made only of suffix words is kept whole."""
    words = [w for w in normalised.split() if w not in COMPANY_SUFFIXES]
    return " ".join(words) if words else normalised


def iso_date(raw: str | None) -> tuple[str | None, bool]:
    """R-17: (date, ok). A valid ISO 8601 date (YYYY, YYYY-MM or YYYY-MM-DD) is returned as
    given; anything else becomes None with ok False. Null input is (None, True)."""
    if raw is None or not str(raw).strip():
        return None, True
    m = _ISO_PARTIAL.match(str(raw).strip())
    if not m:
        return None, False
    year, month, day = int(m.group(1)), m.group(2), m.group(3)
    try:
        if day is not None:
            date(year, int(month), int(day))
        elif month is not None and not 1 <= int(month) <= 12:
            return None, False
    except ValueError:
        return None, False
    return m.group(0), True


def date_interval(value: str) -> tuple[date, date]:
    m = _ISO_PARTIAL.match(value)
    if not m:
        raise ValueError("not an ISO 8601 date")
    year = int(m.group(1))
    if m.group(3):
        d = date(year, int(m.group(2)), int(m.group(3)))
        return d, d
    if m.group(2):
        month = int(m.group(2))
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    return date(year, 1, 1), date(year, 12, 31)


def before_edges(events: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """BEFORE edges for (event id, ISO date) pairs; undated or invalid dates are skipped."""
    spans = []
    for eid, raw in events:
        value, ok = iso_date(raw)
        if value is not None and ok:
            spans.append((eid, *date_interval(value)))
    edges: list[tuple[str, str]] = []
    for eid, _start, end in spans:
        later = [s for s in spans if s[1] > end]
        if later:
            first = min(s[1] for s in later)
            edges.extend((eid, s[0]) for s in later if s[1] == first)
    return sorted(set(edges))


# One extraction batch after R-15 and R-17 checks (built by extractor.py).


@dataclass
class PartyMention:
    name: str
    kind: str
    role: str
    aliases: list[str]
    chunk_ids: list[str]


@dataclass
class EventMention:
    description: str
    date: str | None
    location: str | None
    participants: list[str]
    chunk_ids: list[str]


@dataclass
class ClaimMention:
    text: str
    claim_type: str
    asserted_by: str | None
    chunk_ids: list[str]


@dataclass
class EvidenceMention:
    kind: str
    description: str
    supports: list[int]  # indexes into the batch's claims
    contradicts: list[int]
    chunk_ids: list[str]


@dataclass
class BatchEntities:
    parties: list[PartyMention] = field(default_factory=list)
    events: list[EventMention] = field(default_factory=list)
    claims: list[ClaimMention | None] = field(default_factory=list)  # None: discarded
    evidence: list[EvidenceMention] = field(default_factory=list)


# Resolved case entities


@dataclass
class PartyRecord:
    id: str
    name: str
    kind: str
    role: str
    aliases: list[str]
    chunk_ids: set[str] = field(default_factory=set)

    def names(self) -> list[str]:
        return [self.name, *self.aliases]


@dataclass
class EntityRecord:
    id: str
    props: dict
    chunk_ids: set[str] = field(default_factory=set)


@dataclass
class ResolvedEntities:
    parties: dict[str, PartyRecord]
    events: dict[str, EntityRecord]
    claims: dict[str, EntityRecord]
    evidence: dict[str, EntityRecord]
    participated: set[tuple[str, str]]
    supports: set[tuple[str, str]]
    contradicts: set[tuple[str, str]]
    warnings: list[str]
    adjudications: int


Adjudicate = Callable[[PartyRecord, PartyMention], Awaitable[bool]]


class EntityResolver:
    def __init__(
        self, case_uuid: str, existing_parties: list[PartyRecord], adjudicate: Adjudicate
    ) -> None:
        self.case_uuid = case_uuid
        self.parties: dict[str, PartyRecord] = {p.id: p for p in existing_parties}
        self._adjudicate = adjudicate
        self._verdicts: dict[tuple[str, str], bool] = {}
        self.adjudications = 0

    # Parties (R-14)

    def lookup(self, name: str) -> str | None:
        """Party id for a participant or asserted_by name: exact normalised name or alias,
        else the best match at AUTO_MERGE or above."""
        norm = normalise_name(name)
        if not norm:
            return None
        best, best_score = None, 0.0
        for party in self.parties.values():
            for candidate in party.names():
                cand = normalise_name(candidate)
                if cand == norm:
                    return party.id
                score = fuzz.token_sort_ratio(norm, cand)
                if score > best_score:
                    best, best_score = party.id, score
        return best if best_score >= AUTO_MERGE else None

    def _best_match(self, mention: PartyMention, core: bool = False) -> tuple[PartyRecord | None, float]:
        """Best same-kind party by R-14 score. With core=True, organisation names are compared
        without their company suffixes (SC-13)."""
        def form(name: str) -> str:
            norm = normalise_name(name)
            return core_name(norm) if core else norm

        names = [n for n in (form(x) for x in [mention.name, *mention.aliases]) if n]
        best, best_score = None, 0.0
        for party in self.parties.values():
            if party.kind != mention.kind:  # R-14: only the same kind merges
                continue
            for candidate in party.names():
                cand = form(candidate)
                for n in names:
                    score = 100.0 if n == cand else fuzz.token_sort_ratio(n, cand)
                    if score > best_score:
                        best, best_score = party, score
        return best, best_score

    async def _same(self, party: PartyRecord, mention: PartyMention, norm: str) -> bool:
        """One Groq yes/no per (party, name), cached (R-14)."""
        key = (party.id, norm)
        if key not in self._verdicts:
            self.adjudications += 1
            self._verdicts[key] = await self._adjudicate(party, mention)
        return self._verdicts[key]

    async def add_party(self, mention: PartyMention) -> str | None:
        norm = normalise_name(mention.name)
        if not norm:
            return None
        match, score = self._best_match(mention)
        same = False
        if match is not None and score >= AUTO_MERGE:
            same = True
        elif match is not None and score >= ADJUDICATE_MIN:
            same = await self._same(match, mention, norm)
        elif mention.kind == "organisation":
            # SC-13: 'QuickKart' and 'QuickKart Retail Private Limited' score far below 80
            # as written. Compared without company suffixes, a score of 80 or more is only
            # a candidate: it always goes to the Groq check and never merges on its own.
            core_match, core_score = self._best_match(mention, core=True)
            if core_match is not None and core_score >= ADJUDICATE_MIN:
                match = core_match
                same = await self._same(match, mention, norm)
        if same and match is not None:
            self._merge(match, mention)
            return match.id
        pid = entity_id(PARTY_PREFIX, self.case_uuid, f"{mention.kind} {norm}")
        if pid in self.parties:  # same normalised name and kind
            self._merge(self.parties[pid], mention)
            return pid
        record = PartyRecord(pid, mention.name, mention.kind, mention.role, [])
        self.parties[pid] = record
        self._merge(record, mention)
        return pid

    @staticmethod
    def _merge(party: PartyRecord, mention: PartyMention) -> None:
        known = {normalise_name(n) for n in party.names()}
        for alias in [mention.name, *mention.aliases]:
            norm = normalise_name(alias)
            if norm and norm not in known:
                party.aliases.append(alias.strip())
                known.add(norm)
        if party.role == "other" and mention.role != "other":
            party.role = mention.role
        party.chunk_ids.update(mention.chunk_ids)

    # Everything

    async def resolve(self, batches: list[BatchEntities]) -> ResolvedEntities:
        events: dict[str, EntityRecord] = {}
        claims: dict[str, EntityRecord] = {}
        evidence: dict[str, EntityRecord] = {}
        participated: set[tuple[str, str]] = set()
        supports: set[tuple[str, str]] = set()
        contradicts: set[tuple[str, str]] = set()
        unresolved = 0

        def record(store: dict[str, EntityRecord], eid: str, props: dict, chunks: list[str]) -> str:
            entry = store.setdefault(eid, EntityRecord(eid, props))
            entry.chunk_ids.update(chunks)
            return eid

        for batch in batches:
            for mention in batch.parties:
                await self.add_party(mention)
            for ev in batch.events:
                eid = entity_id(EVENT_PREFIX, self.case_uuid, f"{ev.date or '-'} {ev.description}")
                record(events, eid, {"description": ev.description, "date": ev.date,
                                     "location": ev.location}, ev.chunk_ids)
                for name in ev.participants:
                    pid = self.lookup(name)
                    if pid is None:
                        unresolved += 1
                    else:
                        participated.add((pid, eid))
            claim_ids: list[str | None] = []
            for cl in batch.claims:
                if cl is None:
                    claim_ids.append(None)
                    continue
                cid = entity_id(CLAIM_PREFIX, self.case_uuid, f"{cl.claim_type} {cl.text}")
                asserted = self.lookup(cl.asserted_by) if cl.asserted_by else None
                if cl.asserted_by and asserted is None:
                    unresolved += 1
                record(claims, cid, {"text": cl.text, "claim_type": cl.claim_type,
                                     "asserted_by": asserted}, cl.chunk_ids)
                claim_ids.append(cid)
            for evd in batch.evidence:
                vid = entity_id(EVIDENCE_PREFIX, self.case_uuid, f"{evd.kind} {evd.description}")
                record(evidence, vid, {"kind": evd.kind, "description": evd.description},
                       evd.chunk_ids)
                for indexes, edges in ((evd.supports, supports), (evd.contradicts, contradicts)):
                    for i in indexes:
                        target = claim_ids[i] if 0 <= i < len(claim_ids) else None
                        if target is None:
                            unresolved += 1
                        else:
                            edges.add((vid, target))

        warnings = []
        if unresolved:
            warnings.append(
                f"extraction: {unresolved} participant, claimant or claim reference(s) "
                "did not match an extracted item and were dropped"
            )
        return ResolvedEntities(
            parties=self.parties, events=events, claims=claims, evidence=evidence,
            participated=participated, supports=supports, contradicts=contradicts,
            warnings=warnings, adjudications=self.adjudications,
        )
