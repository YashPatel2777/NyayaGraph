"""Single source of label, relationship and index names, and the id builders (2.1).

Do not type a label, relationship type or index name anywhere else (06 section 2).
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Final

# Node labels (2.3)
ACT: Final = "Act"
CHAPTER: Final = "Chapter"
SECTION: Final = "Section"
CLAUSE: Final = "Clause"
LEGAL_UNIT: Final = "LegalUnit"
LEGAL_CONCEPT: Final = "LegalConcept"
JUDGMENT: Final = "Judgment"
JUDGMENT_CHUNK: Final = "JudgmentChunk"
CASE: Final = "Case"
CASE_DOCUMENT: Final = "CaseDocument"
CASE_CHUNK: Final = "CaseChunk"
PARTY: Final = "Party"
EVENT: Final = "Event"
EVIDENCE: Final = "Evidence"
CLAIM: Final = "Claim"
JOB: Final = "Job"
META: Final = "Meta"

# Relationship types (2.4)
HAS_CHAPTER: Final = "HAS_CHAPTER"
HAS_SECTION: Final = "HAS_SECTION"
HAS_CLAUSE: Final = "HAS_CLAUSE"
REFERS_TO: Final = "REFERS_TO"
REFERS_TO_ACT: Final = "REFERS_TO_ACT"
AMENDS: Final = "AMENDS"
DEFINES: Final = "DEFINES"
ABOUT: Final = "ABOUT"
PRESCRIBES_FOR: Final = "PRESCRIBES_FOR"
INTERPRETS: Final = "INTERPRETS"
CITES: Final = "CITES"
OVERRULES: Final = "OVERRULES"
HAS_CHUNK: Final = "HAS_CHUNK"
HAS_DOCUMENT: Final = "HAS_DOCUMENT"
MENTIONS: Final = "MENTIONS"
PARTICIPATED_IN: Final = "PARTICIPATED_IN"
BEFORE: Final = "BEFORE"
SUPPORTS: Final = "SUPPORTS"
CONTRADICTS: Final = "CONTRADICTS"
INVOKES: Final = "INVOKES"
ALLEGES_VIOLATION_OF: Final = "ALLEGES_VIOLATION_OF"

# Index names (2.2)
LEGAL_UNIT_EMBEDDING_INDEX: Final = "legal_unit_embedding"
CASE_CHUNK_EMBEDDING_INDEX: Final = "case_chunk_embedding"
LEGAL_TEXT_FT_INDEX: Final = "legal_text_ft"
CASE_TEXT_FT_INDEX: Final = "case_text_ft"
CASECHUNK_SCOPE_INDEX: Final = "casechunk_scope"
CASE_OWNER_INDEX: Final = "case_owner"
JOB_OWNER_INDEX: Final = "job_owner"

# The four indexes that must be ONLINE at startup and for /health (1.2, 3.4).
REQUIRED_SEARCH_INDEXES: Final = (
    LEGAL_UNIT_EMBEDDING_INDEX,
    CASE_CHUNK_EMBEDDING_INDEX,
    LEGAL_TEXT_FT_INDEX,
    CASE_TEXT_FT_INDEX,
)

# SC-15 (2.2): properties an index must declare to be ready. The case vector index carries
# the scope properties for the in-index filter of the 4.2 case vector query.
REQUIRED_INDEX_PROPERTIES: Final[dict[str, frozenset[str]]] = {
    CASE_CHUNK_EMBEDDING_INDEX: frozenset({"embedding", "case_id", "user_id"}),
}


def index_missing_properties(name: str, properties: list[str] | None) -> list[str]:
    """Required properties (SC-15) that the index `name` does not declare, sorted."""
    return sorted(REQUIRED_INDEX_PROPERTIES.get(name, frozenset()) - set(properties or []))

# Meta node (2.2). These are the values written by init_neo4j; startup compares the stored
# model and dim with EMBED_MODEL and EMBED_DIM.
META_EMBEDDING_KEY: Final = "embedding"
META_MODEL: Final = "nomic-embed-text"
META_DIM: Final = 768
DOC_PREFIX: Final = "search_document: "
QUERY_PREFIX: Final = "search_query: "
SCHEMA_VERSION: Final = 1

# Id prefixes (2.1)
ACT_PREFIX: Final = "act"
CHAPTER_PREFIX: Final = "ch"
SECTION_PREFIX: Final = "sec"
CLAUSE_PREFIX: Final = "cl"
CONCEPT_PREFIX: Final = "con"
JUDGMENT_PREFIX: Final = "jdg"
JUDGMENT_CHUNK_PREFIX: Final = "jch"
CASE_PREFIX: Final = "case"
DOCUMENT_PREFIX: Final = "doc"
CHUNK_PREFIX: Final = "chk"
PARTY_PREFIX: Final = "pty"
EVENT_PREFIX: Final = "evt"
EVIDENCE_PREFIX: Final = "evd"
CLAIM_PREFIX: Final = "clm"
JOB_PREFIX: Final = "job"

ENTITY_PREFIXES: Final = frozenset({PARTY_PREFIX, EVENT_PREFIX, EVIDENCE_PREFIX, CLAIM_PREFIX})

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")


def slug(value: str) -> str:
    """Lowercase, with every run of non-alphanumeric characters replaced by one hyphen (2.1).

    Leading and trailing hyphens are dropped. An input with no alphanumeric character is
    rejected, because it cannot form a stable id.
    """
    result = _NON_ALNUM.sub("-", value.lower()).strip("-")
    if not result:
        raise ValueError("value has no alphanumeric characters to slug")
    return result


def _uuid4_str(value: str) -> str:
    parsed = uuid.UUID(value)
    if parsed.version != 4:
        raise ValueError("expected a UUID v4")
    return str(parsed)


def new_uuid4() -> str:
    return str(uuid.uuid4())


# Legal KB ids


def act_id(act_slug: str) -> str:
    return f"{ACT_PREFIX}:{slug(act_slug)}"


def chapter_id(act_slug: str, roman: str) -> str:
    # P-04: chapters are keyed by roman numeral only.
    if not re.fullmatch(r"[IVXLC]+", roman):
        raise ValueError("chapter number must be an upper-case roman numeral")
    return f"{CHAPTER_PREFIX}:{slug(act_slug)}:{roman}"


SECTION_NUMBER_RE: Final = re.compile(r"[0-9]{1,3}[A-Z]?")
# A number with an optional capital suffix (2.3), or lowercase letters for rules definitions
# (P-16); either may end in -pN for a split part (2.3).
CLAUSE_LABEL_RE: Final = re.compile(r"(?:[0-9]{1,3}[A-Z]?|[a-z]{1,3})(?:-p[0-9]+)?")


def section_id(act_slug: str, number: str, part: int | None = None) -> str:
    """sec:{act}:{number}; a part of a long Section without Clauses adds -p{part} (2.3)."""
    if not SECTION_NUMBER_RE.fullmatch(number):
        raise ValueError("section number must be 1 to 3 digits with an optional capital letter")
    suffix = f"-p{part}" if part is not None else ""
    return f"{SECTION_PREFIX}:{slug(act_slug)}:{number}{suffix}"


def clause_id(act_slug: str, section: str, label: str) -> str:
    if not CLAUSE_LABEL_RE.fullmatch(label):
        raise ValueError("clause label must be a number with optional capital, or lowercase letters")
    section_id(act_slug, section)  # validates the section number
    return f"{CLAUSE_PREFIX}:{slug(act_slug)}:{section}:{label}"


def concept_id(term: str) -> str:
    return f"{CONCEPT_PREFIX}:{slug(term)}"


def judgment_id(judgment_slug: str) -> str:
    return f"{JUDGMENT_PREFIX}:{slug(judgment_slug)}"


def judgment_chunk_id(judgment_slug: str, order: int) -> str:
    if order < 0:
        raise ValueError("order must be non-negative")
    return f"{JUDGMENT_CHUNK_PREFIX}:{slug(judgment_slug)}:{order}"


# Case KB and system ids. API-facing ids are bare UUID v4 strings; node ids add the
# prefix (3.1).


def case_node_id(case_uuid: str) -> str:
    return f"{CASE_PREFIX}:{_uuid4_str(case_uuid)}"


def document_node_id(doc_uuid: str) -> str:
    return f"{DOCUMENT_PREFIX}:{_uuid4_str(doc_uuid)}"


def job_node_id(job_uuid: str) -> str:
    return f"{JOB_PREFIX}:{_uuid4_str(job_uuid)}"


def chunk_id(doc_uuid: str, order: int) -> str:
    if not 0 <= order <= 9999:
        raise ValueError("chunk order must fit in 4 digits")
    return f"{CHUNK_PREFIX}:{_uuid4_str(doc_uuid)}:{order:04d}"


def normalise_key(key: str) -> str:
    """Lowercase, trimmed, whitespace collapsed. Name-specific normalisation (R-14) is applied
    by the entity resolver before the key reaches the id builder."""
    return _WHITESPACE.sub(" ", key.strip().lower())


def entity_id(prefix: str, case_uuid: str, key: str) -> str:
    """pty, evt, evd or clm id: {prefix}:{case_uuid}:{first 12 hex of sha1(normalised key)}."""
    if prefix not in ENTITY_PREFIXES:
        raise ValueError("prefix must be one of pty, evt, evd, clm")
    normalised = normalise_key(key)
    if not normalised:
        raise ValueError("entity key is empty")
    digest = hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{_uuid4_str(case_uuid)}:{digest}"


def bare_uuid(node_id: str, prefix: str) -> str:
    """Strip a node id prefix to the API-facing bare UUID (3.1)."""
    head, sep, tail = node_id.partition(":")
    if sep != ":" or head != prefix:
        raise ValueError("node id does not carry the expected prefix")
    return _uuid4_str(tail)


# Provisions in answers (3.5, A-35): a split part (A-11) is cited as its whole Clause or
# Section, so answers and the evaluation set use Section and Clause ids only.
_PART_SUFFIX_RE: Final = re.compile(r"-p[0-9]+$")


def provision_base_id(unit_id: str) -> str:
    """sec:a:5-p2 -> sec:a:5 and cl:a:2:47-p1 -> cl:a:2:47; other ids are unchanged."""
    return _PART_SUFFIX_RE.sub("", unit_id)


def format_citation(
    section_number: str, label: str | None, act_short_title: str, act_kind: str | None
) -> str:
    """'Section 38(2), Consumer Protection Act, 2019' (3.5); rules documents use 'Rule'
    (SC-5). A part's label suffix (-pN) is dropped."""
    unit = "Rule" if act_kind in ("rule", "regulation") else "Section"
    clause = f"({_PART_SUFFIX_RE.sub('', label)})" if label else ""
    return f"{unit} {section_number}{clause}, {act_short_title}"
