"""Legal linking (B9).

Regex citations -> INVOKES (A-26):
- 'Section 2', 'section 35(1)', 'Sec. 38', 'S. 38', 'u/s 35', 'sections 35 and 36',
  'Sections 35, 36 and 37 of the Act'. A list is read only after the plural keyword, so
  'section 35 and 2 days' is one citation.
- The Act is read from the words right after the citation: 'of the Consumer Protection Act,
  2019', 'CPA', 'IPC', 'of the Code of Criminal Procedure'. A named Act links only when it
  is a non-external Act in the Legal KB with the same title, and the same year when a year
  is given; 'Consumer Protection Act, 1986' does not link to the 2019 Act, and IPC or CrPC
  references are never mapped to BNS or BNSS (R-26). 'of the Act', 'of this Act' and no
  Act at all mean the Legal KB's principal Act, when exactly one non-external Act of kind
  'act' exists.
- One label, as in 'section 35(1)', links to that Clause when it exists, else to the
  Section. Several labels, as in 'Section 2(1)(d)', link to the Section only: CPA 2019 has
  no nested clause nodes, and 2(1)(d) is the 1986 numbering of a definition.
- An edge is created only when the target exists in the Legal KB.

Claims -> ALLEGES_VIOLATION_OF:
- Each Claim the document mentions is embedded as a query (R-07), the top 5 LegalUnits with
  a vector score of at least LINK_MIN_SCORE go to one Groq call at temperature 0 (R-19)
  that answers yes or no for each with a rationale of at most 300 characters (A-27), and
  each yes creates the edge. confidence is the vector score; method is 'vector+llm'. Ids
  in the reply that were not sent are ignored. A split part (2.3, A-11) links to the Clause
  or Section it belongs to.
- R-04: logs carry counts and timings only.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.errors import AppError
from app.ingestion.case.extractor import CachedReplies, ChunkText, neutralise
from app.prompts.link_verify import LINK_VERIFY_SYSTEM, LINK_VERIFY_USER, PROVISION_BLOCK
from app.schemas.extraction import LinkVerification
from app.schemas.graph_schema import CLAUSE_LABEL_RE, clause_id, section_id
from app.services.llm.groq_client import GroqClient
from app.services.retrieval import graph_queries
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)

LINK_TOP_K = 5  # B9
LINK_TEMPERATURE = 0.0  # R-19
LINK_MAX_TOKENS = 800  # A-27: R-19 sets no cap for link verification
TEXT_SPAN_MAX = 200

_NUM = r"\d{1,3}[A-Z]?"
_LABELS = r"(?:\s?\(\s?[0-9A-Za-z]{1,4}\s?\))*"
_ONE = rf"{_NUM}{_LABELS}"
_SINGLE = re.compile(
    rf"(?<![A-Za-z])(?:[Ss]ection|SECTION|[Ss]ec\.?|SEC\.?|S\.|u/s\.?|U/S\.?)\s*(?P<refs>{_ONE})"
)
_PLURAL = re.compile(
    rf"(?<![A-Za-z])(?:[Ss]ections|SECTIONS|[Ss]s\.|[Ss]ecs\.?)\s*"
    rf"(?P<refs>{_ONE}(?:\s*(?:,|and|&)\s*{_ONE})+|{_ONE})"
)
_REF = re.compile(rf"(?P<num>{_NUM})(?P<labels>{_LABELS})")
_LABEL = re.compile(r"\(\s?([0-9A-Za-z]{1,4})\s?\)")

# 'of the Consumer Protection Act, 2019', 'Contract Act', 'of the Code of Criminal
# Procedure, 1973'. The 'of' is optional so 'Section 13 Contract Act' is not read as 'the Act'.
_ACT_NAME = re.compile(
    r"^\s*,?\s*(?:(?:of|under|in)\s+(?:the\s+|this\s+|said\s+)?)?"
    r"(?P<name>(?:[A-Z][\w().&'-]*\s+){0,9}?(?:Act|Code|Sanhita|Adhiniyam)\b"
    r"(?:\s+of\s+(?:[A-Z][\w-]*\s?){1,4})?"
    r"(?:\s*,?\s*(?P<year>(?:18|19|20)\d{2}))?)"
)
_ACT_ABBR = re.compile(
    r"^\s*,?\s*(?:(?:of|under)\s+(?:the\s+)?)?"
    r"(?P<abbr>I\.?P\.?C\.?|Cr\.?\s?P\.?C\.?|C\.?P\.?C\.?|BNSS|BNS|BSA|CPA|C\.P\.A\.)"
    r"(?![A-Za-z])(?:\s*,?\s*(?P<year>(?:18|19|20)\d{2}))?"
)
_ABBREVIATIONS = {
    "ipc": "indian penal code",
    "crpc": "code of criminal procedure",
    "cpc": "code of civil procedure",
    "bns": "bharatiya nyaya sanhita",
    "bnss": "bharatiya nagarik suraksha sanhita",
    "bsa": "bharatiya sakshya adhiniyam",
    "cpa": "consumer protection act",
}
_YEAR = re.compile(r",?\s*(?:18|19|20)\d{2}\s*$")
_PART = re.compile(r"-p\d+$")


@dataclass(frozen=True)
class Citation:
    section: str
    labels: tuple[str, ...]
    act_name: str | None  # normalised title without year; None when no Act was named
    act_year: int | None
    default_act: bool  # 'of the Act', 'this Act' or nothing
    text_span: str


@dataclass
class ActIndex:
    """Non-external Acts of kind 'act' by normalised title."""

    acts: dict[str, list[tuple[str, int | None]]]  # title -> [(act slug, year)]
    principal: str | None  # the only non-external Act of kind 'act', if exactly one

    @classmethod
    def from_rows(cls, rows: list[dict]) -> ActIndex:
        acts: dict[str, list[tuple[str, int | None]]] = {}
        principal = [r for r in rows if r["kind"] == "act" and not r["external"]]
        for row in principal:
            title = row["short_title"] or row["title"] or ""
            year_m = re.search(r"((?:18|19|20)\d{2})\s*$", title)
            slug = row["id"].split(":", 1)[1]
            acts.setdefault(normalise_title(title), []).append(
                (slug, int(year_m.group(1)) if year_m else None)
            )
        return cls(acts, principal[0]["id"].split(":", 1)[1] if len(principal) == 1 else None)

    def resolve(self, citation: Citation) -> str | None:
        if citation.default_act:
            return self.principal
        for slug, year in self.acts.get(citation.act_name or "", []):
            if citation.act_year is None or citation.act_year == year:
                return slug
        return None  # another Act, or another year of this one (R-26)


def normalise_title(title: str) -> str:
    text = _YEAR.sub("", title.lower())
    text = re.sub(r"^the\s+", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _act_after(text: str) -> tuple[str | None, int | None, bool, int]:
    """(act title, year, default, characters consumed) for the words after a citation."""
    abbr = _ACT_ABBR.match(text)
    if abbr:
        key = re.sub(r"[^a-z]", "", abbr.group("abbr").lower())
        year = int(abbr.group("year")) if abbr.group("year") else None
        return _ABBREVIATIONS[key], year, False, abbr.end()
    name = _ACT_NAME.match(text)
    if name:
        raw = name.group("name")
        if normalise_title(raw) == "act":  # 'of the Act', 'of this Act'
            return None, None, True, name.end()
        year = int(name.group("year")) if name.group("year") else None
        return normalise_title(raw), year, False, name.end()
    return None, None, True, 0


def find_citations(text: str) -> list[Citation]:
    found: list[tuple[int, Citation]] = []
    taken: list[tuple[int, int]] = []
    for pattern in (_PLURAL, _SINGLE):
        for m in pattern.finditer(text):
            if any(s <= m.start() < e for s, e in taken):
                continue
            act, year, default, used = _act_after(text[m.end() : m.end() + 160])
            span = text[m.start() : m.end() + used].strip()[:TEXT_SPAN_MAX]
            taken.append((m.start(), m.end()))
            for ref in _REF.finditer(m.group("refs")):
                labels = tuple(_LABEL.findall(ref.group("labels")))
                found.append((m.start(), Citation(ref.group("num"), labels, act, year, default, span)))
    return [c for _, c in sorted(found, key=lambda x: x[0])]


def citation_targets(citation: Citation, act_slug: str) -> list[str]:
    """Candidate ids, most specific first; the caller keeps the first that exists."""
    try:
        section = section_id(act_slug, citation.section)
    except ValueError:
        return []
    if len(citation.labels) == 1 and CLAUSE_LABEL_RE.fullmatch(citation.labels[0]):
        return [clause_id(act_slug, citation.section, citation.labels[0]), section]
    return [section]


def provision_of(unit_id: str) -> str:
    """A split part (2.3, A-11) belongs to the Clause or Section its id starts with."""
    return _PART.sub("", unit_id)


@dataclass
class LinkReport:
    invokes: list[dict] = field(default_factory=list)  # rows for INVOKES_MERGE
    alleges: list[dict] = field(default_factory=list)  # rows for ALLEGES_MERGE
    warnings: list[str] = field(default_factory=list)
    llm_calls: int = 0


class LegalLinker:
    def __init__(
        self,
        neo4j: Neo4jService,
        embeddings: EmbeddingService,
        llm: GroqClient,
        min_score: float,
    ) -> None:
        self.neo4j = neo4j
        self.embeddings = embeddings
        self.llm = llm
        self.min_score = min_score

    async def citation_links(self, chunks: list[ChunkText]) -> list[dict]:
        acts = ActIndex.from_rows(await self.neo4j.read(graph_queries.LEGAL_ACT_TITLES))
        wanted: list[tuple[str, list[str], str]] = []
        for chunk in chunks:
            for citation in find_citations(chunk.text):
                slug = acts.resolve(citation)
                if slug is None:
                    continue
                targets = citation_targets(citation, slug)
                if targets:
                    wanted.append((chunk.id, targets, citation.text_span))
        ids = sorted({t for _, targets, _ in wanted for t in targets})
        if not ids:
            return []
        rows = await self.neo4j.read(graph_queries.LEGAL_PROVISIONS_EXIST, {"ids": ids})
        existing = {r["id"] for r in rows}
        links: dict[tuple[str, str], str] = {}
        for chunk_id, targets, span in wanted:
            target = next((t for t in targets if t in existing), None)
            if target is not None:
                links.setdefault((chunk_id, target), span)
        return [{"chunk_id": c, "target": t, "text_span": s} for (c, t), s in links.items()]

    async def claim_links(
        self,
        claims: list[tuple[str, str]],
        checkpoint: Callable[[], Awaitable[None]],
        llm: GroqClient | CachedReplies | None = None,
    ) -> tuple[list[dict], list[str], int]:
        """(ALLEGES rows, warnings, llm calls) for (claim id, claim text) pairs. llm is the
        document's CachedReplies (SC-12); the shared client is used when it is None."""
        llm = llm or self.llm
        rows: list[dict] = []
        failed = calls = 0
        for claim_id, text in claims:
            await checkpoint()
            vector = await self.embeddings.embed_query(text)
            hits = await self.neo4j.read(
                graph_queries.LEGAL_VECTOR_SEARCH, {"k": LINK_TOP_K, "query_vec": vector}
            )
            scores = {h["id"]: h["score"] for h in hits if h["score"] >= self.min_score}
            if not scores:
                continue
            texts = await self.neo4j.read(graph_queries.LEGAL_UNIT_TEXTS, {"ids": list(scores)})
            provisions = "\n\n".join(
                PROVISION_BLOCK.format(unit_id=r["id"], text=r["search_text"]) for r in texts
            )
            messages = [
                ("system", LINK_VERIFY_SYSTEM),
                ("user", LINK_VERIFY_USER.format(claim=neutralise(text), provisions=provisions)),
            ]
            try:
                reply = await llm.complete_json_resuming(
                    messages,
                    LinkVerification,
                    temperature=LINK_TEMPERATURE,
                    max_tokens=LINK_MAX_TOKENS,
                    checkpoint=checkpoint,
                )
            except AppError as exc:
                if exc.details.get("reason") != "invalid_structured_output":
                    raise
                failed += 1
                calls += 2
                continue
            calls += reply.llm_calls
            best: dict[str, dict] = {}
            for verdict in reply.value.verdicts:
                if not verdict.applies or verdict.unit_id not in scores:
                    continue
                target = provision_of(verdict.unit_id)
                score = round(float(scores[verdict.unit_id]), 4)
                if target not in best or score > best[target]["confidence"]:
                    best[target] = {
                        "claim_id": claim_id, "target": target, "confidence": score,
                        "rationale": verdict.rationale.strip()[:300],
                    }
            rows.extend(best.values())
        warnings = []
        if failed:
            warnings.append(
                f"legal_linking: {failed} claim(s) could not be checked (invalid model output, E-02)"
            )
        return rows, warnings, calls
