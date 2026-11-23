"""Legal ingestion, steps A1 to A8 (1.4 A).

A3 text extraction (pypdf per page; pages under 40 non-space characters go to OCR), A4/A5
parsing and references (parser.py, reference_extractor.py), A6 retrieval units and
embeddings (legal_splitter.py, EmbeddingService), A7 one write transaction per Act, A8
verification. Legal nodes are written only here (R-18).

R-04: logs carry ids, counts and timings only.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import anyio

from app.errors import AppError, ErrorCode
from app.ingestion.chunking.legal_splitter import LegalSplitter, Unit
from app.ingestion.legal.parser import (
    ParseError,
    ParseResult,
    apply_substitution,
    extract_substitutions,
    parse_amending,
    parse_statute,
)
from app.ingestion.legal.reference_extractor import ReferenceExtractor
from app.schemas.api import LegalIngestMetadata
from app.schemas.graph_schema import (
    REQUIRED_SEARCH_INDEXES,
    act_id as act_node_id,
    slug,
    chapter_id,
    clause_id,
    concept_id,
    format_citation,
    job_node_id,
    section_id,
)
from app.services.media.ocr_service import OcrService
from app.services.retrieval import graph_queries
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)

OCR_MIN_CHARS = 40  # 1.4 A3
LEGAL_WRITE_TIMEOUT_S = 120.0  # A-15: one transaction per Act exceeds the 15 s query budget
MAX_JOB_WARNINGS = 200


@dataclass
class LegalIngestReport:
    act_id: str
    parse_report: dict[str, int]
    units: int
    counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)
    ocr_pages: int = 0


class VerificationError(Exception):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass
class _Prepared:
    """One document after A3 to A6: its write steps and what A8 should find."""

    act_id: str
    parsed: ParseResult
    units: list[Unit]
    steps: list
    expected: dict[str, int]
    warnings: list[str]
    ocr_pages: int
    new_substitutions: list[dict[str, Any]]


def _title_key(title: str | None) -> str:
    return slug(re.sub(r"(?i)^the\s+", "", title or "x"))


def _dmy(d: date) -> str:
    return f"{d.day}-{d.month}-{d.year}"  # as amendment footnotes print it (17-5-2021)


ProgressFn = Callable[[str, int], Awaitable[None]]


def extract_pages(pdf_path: str, ocr: OcrService | None) -> tuple[list[str], int]:
    """A3: pypdf per page; a page with fewer than 40 non-space characters goes to OCR."""
    import logging as _logging

    from pypdf import PdfReader

    _logging.getLogger("pypdf").setLevel(_logging.ERROR)
    reader = PdfReader(pdf_path)
    pages: list[str] = []
    ocr_pages = 0
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if len(re.sub(r"\s", "", text)) < OCR_MIN_CHARS:
            if ocr is None:
                raise ParseError("OCR_UNAVAILABLE", f"page {index + 1} needs OCR and none is configured")
            text = ocr.ocr_pdf_page(pdf_path, index)
            ocr_pages += 1
        pages.append(text)
    return pages, ocr_pages


class LegalProcessor:
    def __init__(
        self,
        neo4j: Neo4jService,
        embeddings: EmbeddingService,
        ocr: OcrService | None = None,
    ) -> None:
        self.neo4j = neo4j
        self.embeddings = embeddings
        self.ocr = ocr

    async def ingest(
        self,
        pdf_path: str,
        meta: LegalIngestMetadata,
        progress: ProgressFn | None = None,
    ) -> LegalIngestReport:
        async def step(stage: str, pct: int) -> None:
            if progress:
                await progress(stage, pct)

        timings: dict[str, int] = {}
        t0 = time.perf_counter()
        await step("text_extraction", 5)

        if meta.amends is None:
            # P-21: substitutions stored on amending Acts are re-applied on every ingest.
            subs = await self._stored_substitutions(meta.act_id, exclude=None)
            main = await self._prepare(pdf_path, meta, subs, step)
            preps = [main]
        else:
            main = await self._prepare(pdf_path, meta, [], step, amending=True)
            principal = await self._prepare_principal(meta, main.new_substitutions, step)
            preps = [main, principal]
        timings["prepare"] = int((time.perf_counter() - t0) * 1000)

        # A7: one transaction (for an amendment: the notification and the rebuilt principal)
        await step("writing", 75)
        t = time.perf_counter()
        steps = [s for p in preps for s in p.steps]
        await anyio.to_thread.run_sync(
            lambda: self.neo4j.write_transaction_sync(steps, timeout_s=LEGAL_WRITE_TIMEOUT_S)
        )
        timings["writing"] = int((time.perf_counter() - t) * 1000)

        # A8
        await step("verification", 90)
        t = time.perf_counter()
        counts = await self._verify(main.act_id, main.expected, len(main.units))
        for p in preps[1:]:
            await self._verify(p.act_id, p.expected, len(p.units))
        timings["verification"] = int((time.perf_counter() - t) * 1000)
        timings["total"] = int((time.perf_counter() - t0) * 1000)

        report = LegalIngestReport(
            act_id=meta.act_id,
            parse_report=main.parsed.report,
            units=sum(len(p.units) for p in preps),
            counts=counts,
            warnings=[w for p in preps for w in p.warnings],
            timings_ms=timings,
            ocr_pages=sum(p.ocr_pages for p in preps),
        )
        await step("done", 100)
        logger.info(
            "legal_ingest_ok act_id=%s sections=%d units=%d warnings=%d ms=%d",
            meta.act_id, len(main.parsed.sections), report.units, len(report.warnings), timings["total"],
        )
        return report

    async def _prepare(
        self,
        pdf_path: str,
        meta: LegalIngestMetadata,
        substitutions: list[dict[str, Any]],
        step: Callable[[str, int], Awaitable[None]],
        amending: bool = False,
    ) -> _Prepared:
        """A3 to A6 for one document, ending with its write steps (nothing is written)."""
        pages, ocr_pages = await anyio.to_thread.run_sync(extract_pages, pdf_path, self.ocr)  # A3
        await step("parsing", 25)
        if amending:  # P-19
            parsed = await anyio.to_thread.run_sync(parse_amending, pages, meta.expected_sections)
        else:
            parsed = await anyio.to_thread.run_sync(parse_statute, pages, meta.kind, meta.expected_sections)
        stored_meta = meta
        extras: dict[str, Any] = {
            "ingest_metadata": meta.model_dump_json(exclude={"replace"}),
            "source_path": pdf_path,
        }
        new_subs: list[dict[str, Any]] = []
        sub_amends: list[dict[str, Any]] = []
        if amending:
            new_subs = self._substitution_records(parsed, meta)
            extras |= {"amends_act_id": meta.amends.act_id, "substitutions": json.dumps(new_subs)}
        elif substitutions:
            meta, sub_amends = self._apply_substitutions(parsed, meta, substitutions)  # P-19, P-20

        parent = await self._parent_context(stored_meta)
        graph = self._build_graph(parsed, meta, parent, references=not amending)  # A5
        graph["amends"].extend(sub_amends)

        await step("embedding", 45)  # A6; an amending notification has no units (SC-9)
        units = [] if amending else LegalSplitter(parsed, meta.act_id, meta.short_title).units()
        vectors = await self.embeddings.embed_documents([u.search_text for u in units])

        steps, keep_ids, expected = self._write_steps(parsed, meta, graph, units, vectors, extras)
        if meta.replace:
            steps.insert(0, (graph_queries.LEGAL_DELETE_STALE, {"act_id": meta.act_id, "keep_ids": keep_ids}, None))
        return _Prepared(meta.act_id, parsed, units, steps, expected,
                         parsed.warnings + graph["warnings"], ocr_pages, new_subs)

    async def _prepare_principal(
        self,
        meta: LegalIngestMetadata,
        new_subs: list[dict[str, Any]],
        step: Callable[[str, int], Awaitable[None]],
    ) -> _Prepared:
        """SC-9: rebuild the principal rules from their own PDF with every substitution
        (stored ones plus this notification's), exactly as a re-ingest would (P-21)."""
        principal = meta.amends.act_id
        rows = await self.neo4j.read(graph_queries.LEGAL_ACT, {"act_node_id": act_node_id(principal)})
        if not rows or rows[0]["external"]:
            raise ParseError("PARSE_VALIDATION_FAILED", f"principal rules {principal} are not ingested",
                             [f"ingest {principal} before {meta.act_id}"])
        row = rows[0]
        if not row.get("ingest_metadata") or not row.get("source_path"):
            raise ParseError("PARSE_VALIDATION_FAILED", f"{principal} has no stored source",
                             [f"re-ingest {principal} once so its metadata and file path are recorded"])
        if not Path(row["source_path"]).exists():
            raise ParseError("PARSE_VALIDATION_FAILED", f"source file of {principal} is missing",
                             [str(row["source_path"])])
        titles = {_title_key(row["title"]), _title_key(row["short_title"])}
        wrong = [s["principal_title"] for s in new_subs if _title_key(s["principal_title"]) not in titles]
        if wrong:
            raise ParseError("PARSE_VALIDATION_FAILED", "the notification amends different rules (P-19)", wrong)
        principal_meta = LegalIngestMetadata.model_validate_json(row["ingest_metadata"])
        principal_meta = principal_meta.model_copy(update={"replace": True})
        stored = await self._stored_substitutions(principal, exclude=act_node_id(meta.act_id))
        return await self._prepare(row["source_path"], principal_meta, stored + new_subs, step)

    async def _stored_substitutions(self, act_id: str, exclude: str | None) -> list[dict[str, Any]]:
        rows = await self.neo4j.read(graph_queries.AMENDMENT_SUBSTITUTIONS, {"act_id": act_id, "exclude": exclude})
        return [s for r in rows for s in json.loads(r["substitutions"] or "[]")]

    @staticmethod
    def _substitution_records(parsed: ParseResult, meta: LegalIngestMetadata) -> list[dict[str, Any]]:
        effective = meta.amends.effective_on or meta.as_on_date
        return [
            {
                "amending_act_id": act_node_id(meta.act_id),
                "notification": meta.number,
                "published": meta.as_on_date.isoformat(),
                "effective_on": effective.isoformat(),
                "principal_title": s.principal_title,
                "rule": s.rule,
                "sub": s.sub,
                "text": s.text,
            }
            for s in extract_substitutions(parsed)
        ]

    @staticmethod
    def _apply_substitutions(
        parsed: ParseResult, meta: LegalIngestMetadata, subs: list[dict[str, Any]]
    ) -> tuple[LegalIngestMetadata, list[dict[str, Any]]]:
        """P-19/P-20: apply in publication order; note, AMENDS row with previous_text, and the
        principal's as_on_date moves to the latest amendment."""
        amends: list[dict[str, Any]] = []
        as_on = meta.as_on_date
        for s in sorted(subs, key=lambda x: (x["published"], x["rule"], x["sub"] or "")):
            previous = apply_substitution(parsed, s["rule"], s["sub"], s["text"])
            published = date.fromisoformat(s["published"])
            effective = date.fromisoformat(s["effective_on"])
            note = (f"Subs. by {s['notification']}, dated {_dmy(published)} (w.e.f. {_dmy(effective)})"
                    + (f", sub-rule ({s['sub']})." if s["sub"] else "."))
            sec = next(x for x in parsed.sections if x.number == s["rule"])
            if note not in sec.amendment_notes:
                sec.amendment_notes.append(note)
            amends.append({"act_id": s["amending_act_id"], "section_id": section_id(meta.act_id, s["rule"]),
                           "note": note, "effective_on": effective, "previous_text": previous})
            as_on = max(as_on, published)
        return meta.model_copy(update={"as_on_date": as_on}), amends

    async def _parent_context(self, meta: LegalIngestMetadata) -> dict[str, Any] | None:
        """For rules: the parent Act must be ingested; its provision ids resolve P-17."""
        if meta.issued_under is None:
            return None
        parent_slug = meta.issued_under.act_id
        rows = await self.neo4j.read(graph_queries.LEGAL_ACT, {"act_node_id": act_node_id(parent_slug)})
        if not rows or rows[0]["external"]:
            raise ParseError(
                "PARSE_VALIDATION_FAILED",
                f"parent act {parent_slug} is not ingested",
                [f"ingest {parent_slug} before {meta.act_id}"],
            )
        ids = await self.neo4j.read(graph_queries.LEGAL_PROVISION_IDS, {"act_id": parent_slug})
        return {
            "slug": parent_slug,
            "titles": [rows[0]["title"], rows[0]["short_title"]],
            "ids": {r["id"] for r in ids},
        }

    def _build_graph(
        self,
        parsed: ParseResult,
        meta: LegalIngestMetadata,
        parent: dict[str, Any] | None,
        references: bool = True,
    ) -> dict[str, Any]:
        """A5: reference edges, stub Acts, amendments and PRESCRIBES_FOR (P-09 to P-11, P-17,
        SC-6). Validation failures raise ParseError so nothing is written (E-14)."""
        act = meta.act_id
        own_ids = {chapter_id(act, c.number) for c in parsed.chapters}
        for s in parsed.sections:
            own_ids.add(section_id(act, s.number))
            own_ids.update(clause_id(act, s.number, c.label) for c in s.clauses)

        known = own_ids | (parent["ids"] if parent else set())
        extractor = ReferenceExtractor(
            act,
            [meta.title, meta.short_title],
            known,
            parent_slug=parent["slug"] if parent else None,
            parent_titles=parent["titles"] if parent else (),
        )
        for s in parsed.sections if references else []:  # an amending notification: none (SC-9)
            sid = section_id(act, s.number)
            extractor.extract(sid, sid, f"{s.title}. {s.text}")
            for c in s.clauses:
                extractor.extract(clause_id(act, s.number, c.label), sid, c.text)
        refs = extractor.result

        stub_acts = dict(refs.stub_acts)
        amends = []
        for a in parsed.amendments:
            if a.act_number:  # P-09: an Act number gives a stub Act and an AMENDS edge
                stub = act_node_id(a.act_number)
                stub_acts.setdefault(stub, f"Act {a.act_number}")
                amends.append({
                    "act_id": stub,
                    "section_id": section_id(act, a.section_number),
                    "note": a.note,
                    "effective_on": a.effective_on,
                })

        prescribes = []
        problems = []
        rule_numbers = {s.number for s in parsed.sections}
        for p in meta.prescribes_for or []:
            target = section_id(parent["slug"], p.section) if parent else None
            if p.rule not in rule_numbers:
                problems.append(f"prescribes_for rule {p.rule} is not in the parsed document")
            if parent is None or target not in parent["ids"]:
                problems.append(f"prescribes_for section {p.section} is not in the parent act")
            prescribes.append({"rule_section_id": section_id(act, p.rule), "section_id": target, "note": p.note})
        if problems:
            raise ParseError("PARSE_VALIDATION_FAILED", "prescribes_for failed validation (SC-6)", problems)

        act_refs: dict[tuple[str, str], list[str]] = defaultdict(list)
        for e in refs.refers_to_act:
            act_refs[(e.source, e.target)].append(e.section_ref or "")
        return {
            "refers_to": sorted(refs.refers_to, key=lambda e: (e.source, e.target)),
            "refers_to_act": [
                {"source": s, "target": t, "section_ref": "; ".join(sorted({r for r in v if r})) or None}
                for (s, t), v in sorted(act_refs.items())
            ],
            "stub_acts": stub_acts,
            "amends": amends,
            "prescribes_for": prescribes,
            "warnings": refs.warnings,
        }

    def _write_steps(
        self,
        parsed: ParseResult,
        meta: LegalIngestMetadata,
        graph: dict[str, Any],
        units: list[Unit],
        vectors: list[list[float]],
        extras: dict[str, Any],
    ) -> tuple[list, list[str], dict[str, int]]:
        act = meta.act_id
        node_id = act_node_id(act)
        splitter = LegalSplitter(parsed, act, meta.short_title)
        unit_data = {
            u.id: {"embedding": v, "embed_model": self.embeddings.model, "search_text": u.search_text}
            for u, v in zip(units, vectors)
        }
        no_unit = {"embedding": None, "embed_model": None, "search_text": None}

        act_props = {
            "kind": meta.kind, "title": meta.title, "short_title": meta.short_title,
            "year": meta.year, "number": meta.number, "jurisdiction": meta.jurisdiction,
            "as_on_date": meta.as_on_date, "status": "in_force", "external": False,
            "issued_under_act_id": meta.issued_under.act_id if meta.issued_under else None,
            "issued_under_section": meta.issued_under.sections if meta.issued_under else None,
            "objects_text": parsed.objects_text, "enabling_text": parsed.enabling_text,
            "source_file": Path(extras["source_path"]).name,
            **extras,
        }
        chapters = [
            {"id": chapter_id(act, c.number), "props": {"act_id": act, "number": c.number, "title": c.title}}
            for c in parsed.chapters
        ]
        sections, clauses, concepts = [], [], []
        for s in parsed.sections:
            sid = section_id(act, s.number)
            is_unit = sid in unit_data
            sections.append({
                "id": sid,
                "chapter_id": chapter_id(act, s.chapter_number) if s.chapter_number else None,
                "is_unit": is_unit,
                "props": {
                    "act_id": act, "number": s.number, "title": s.title,
                    "chapter_number": s.chapter_number, "text": s.text, "status": s.status,
                    "in_force_from": s.in_force_from, "amendment_notes": s.amendment_notes,
                    "is_unit": is_unit, **unit_data.get(sid, no_unit),
                },
            })
            for c in s.clauses:
                cid = clause_id(act, s.number, c.label)
                c_unit = cid in unit_data
                clauses.append({
                    "id": cid, "section_id": sid, "is_unit": c_unit,
                    "props": {
                        "act_id": act, "section_number": s.number, "label": c.label,
                        "kind": c.kind, "term": c.term, "text": c.text,
                        **unit_data.get(cid, no_unit),
                    },
                })
                # P-08: definitions in an Act create LegalConcepts; rules do not (P-16).
                if c.kind == "definition" and c.term and meta.kind == "act":
                    concepts.append({
                        "id": concept_id(c.term), "clause_id": cid,
                        "props": {"act_id": act, "name": c.term, "definition_text": c.text},
                    })
        parts = [
            {
                "id": u.id,
                "section_id": section_id(act, u.section_number),
                "props": {
                    "act_id": act, "section_number": u.section_number, "label": u.label,
                    "kind": "part", "text": u.text, **unit_data[u.id],
                },
            }
            for u in units
            if u.kind == "part"
        ]
        if len({c["id"] for c in concepts}) != len(concepts):
            graph["warnings"].append("two definitions share a concept id")

        stubs = [
            {"id": sid, "title": name, "number": name.removeprefix("Act ") if name.startswith("Act ") else None}
            for sid, name in sorted(graph["stub_acts"].items())
            if sid != node_id
        ]
        refers_to = [
            {"source": e.source, "target": e.target, "qualifier": e.qualifier}
            for e in graph["refers_to"]
        ]
        keep_ids = (
            [c["id"] for c in chapters] + [s["id"] for s in sections] + [c["id"] for c in clauses]
            + [p["id"] for p in parts] + [k["id"] for k in concepts]
        )
        p_act = {"act_id": act, "act_node_id": node_id}
        steps = [
            (graph_queries.LEGAL_DELETE_DERIVED_EDGES, p_act, None),
            (graph_queries.LEGAL_DELETE_ACT_EDGES, p_act, None),
            (graph_queries.LEGAL_DELETE_AMENDS, p_act, None),
            (graph_queries.LEGAL_MERGE_ACT, {**p_act, "props": act_props}, None),
            (graph_queries.LEGAL_MERGE_CHAPTERS, p_act, chapters),
            (graph_queries.LEGAL_MERGE_SECTIONS, p_act, sections),
            (graph_queries.LEGAL_MERGE_CLAUSES, p_act, clauses),
            (graph_queries.LEGAL_MERGE_PARTS, p_act, parts),
            (graph_queries.LEGAL_MERGE_CONCEPTS, p_act, concepts),
            (graph_queries.LEGAL_MERGE_STUB_ACTS, p_act, stubs),
            (graph_queries.LEGAL_MERGE_REFERS_TO, p_act, refers_to),
            (graph_queries.LEGAL_MERGE_REFERS_TO_ACT, p_act, graph["refers_to_act"]),
            (graph_queries.LEGAL_MERGE_AMENDS, p_act, graph["amends"]),
            (graph_queries.LEGAL_MERGE_PRESCRIBES_FOR, p_act, graph["prescribes_for"]),
        ]
        expected = {
            "chapters": len(chapters), "sections": len(sections), "clauses": len(clauses),
            "units": len(units), "concepts": len(concepts),
        }
        return steps, keep_ids, expected

    async def _verify(self, act_id: str, expected: dict[str, int], n_units: int) -> dict[str, int]:
        """A8: node counts match the parse, every unit has a vector, indexes are ONLINE."""
        rows = await self.neo4j.read(graph_queries.LEGAL_COUNTS, {"act_id": act_id, "dim": self.embeddings.dim})
        counts = {k: int(v or 0) for k, v in rows[0].items()}
        problems = [
            f"{label}: expected {want}, found {counts.get(label)}"
            for label, want in expected.items()
            if counts.get(label) != want
        ]
        if counts.get("embedded") != n_units:
            problems.append(f"embedded units: expected {n_units}, found {counts.get('embedded')}")
        await anyio.to_thread.run_sync(
            lambda: self.neo4j.run_autocommit_sync(graph_queries.AWAIT_INDEXES, {"timeout_s": 300})
        )
        states = await self.neo4j.read(graph_queries.INDEX_STATES, {"names": list(REQUIRED_SEARCH_INDEXES)})
        offline = [r["name"] for r in states if r["state"] != "ONLINE"]
        if offline or len(states) != len(REQUIRED_SEARCH_INDEXES):
            problems.append(f"indexes not ONLINE: {offline or 'missing'}")
        if problems:
            raise VerificationError(problems)
        return counts


async def list_acts(neo4j: Neo4jService, limit: int, cursor: str | None) -> tuple[list[dict], str | None]:
    """GET /v1/admin/legal/acts: one page of Act nodes ordered by id (3.1 cursor = last id)."""
    rows = await neo4j.read(graph_queries.LIST_ACTS, {"limit": limit + 1, "cursor": cursor})
    page = rows[:limit]
    for row in page:
        row["act_id"] = row.pop("id").removeprefix("act:")
        row["external"] = bool(row["external"])
        if row["as_on_date"] is not None:
            row["as_on_date"] = row["as_on_date"].to_native()
    next_cursor = f"act:{page[-1]['act_id']}" if len(rows) > limit and page else None
    return page, next_cursor


async def get_provision(
    neo4j: Neo4jService,
    provision_id: str,
    terms_in: Callable[[str, set[str]], list[tuple[str, str]]],
) -> dict:
    """GET /v1/legal/provisions/{provision_id} (3.4, A-45). Citations are built from stored
    numbers, labels and Act titles (3.5 format). `terms_in` is the cached R-13 concept list's
    matcher. Unknown ids and split parts give 404 PROVISION_NOT_FOUND."""
    rows = await neo4j.read(graph_queries.PROVISION_GET, {"id": provision_id})
    if not rows:
        raise AppError(ErrorCode.PROVISION_NOT_FOUND)
    row = rows[0]
    linked = list(dict.fromkeys([*row["refers_to"], *row["referenced_by"], *([row["parent_id"]] if row["parent_id"] else [])]))
    refs = {r["id"]: r for r in await neo4j.read(graph_queries.LEGAL_PROVISIONS_HYDRATE, {"ids": linked})} if linked else {}

    def citation(r: dict) -> str:
        label = None if r["is_section"] else r.get("label")
        return format_citation(r.get("section_number") or "", label, r.get("act_short_title") or "", r.get("act_kind"))

    def ref(pid: str) -> dict | None:
        return {"id": pid, "citation": citation(refs[pid])} if pid in refs else None

    text = row["text"] or ""
    in_force = row["in_force_from"]
    return {
        "provision_id": row["id"],
        "citation": citation(row),
        "text": text,
        "status": row["status"] or "in_force",
        "in_force_from": in_force.to_native() if hasattr(in_force, "to_native") else in_force,
        "amendment_notes": list(row["amendment_notes"] or []),
        "parent": ref(row["parent_id"]) if row["parent_id"] else None,
        "refers_to": [r for pid in row["refers_to"] if (r := ref(pid))],
        "referenced_by": [r for pid in row["referenced_by"] if (r := ref(pid))],
        "definitions": [{"term": term, "id": cid} for term, cid in terms_in(text, {row["id"]})],
    }


# Job lifecycle (A2, 3.4 stages). Phase 3 moves scheduling onto the job queue (1.2).


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def create_legal_job(neo4j: Neo4jService, job_uuid: str, user_id: str) -> None:
    await neo4j.write(
        graph_queries.CREATE_JOB,
        {"id": job_node_id(job_uuid), "type": "legal_ingest", "user_id": user_id,
         "case_id": None, "doc_id": None, "created_at": _now()},
    )


async def run_legal_job(
    processor: LegalProcessor, job_uuid: str, pdf_path: str, meta: LegalIngestMetadata
) -> LegalIngestReport | None:
    """Run one legal_ingest job and record its outcome on the Job node."""
    neo4j = processor.neo4j
    jid = job_node_id(job_uuid)

    async def update(**props: Any) -> None:
        await neo4j.write(graph_queries.UPDATE_JOB, {"id": jid, "props": props})

    async def progress(stage: str, pct: int) -> None:
        await update(stage=stage, progress=pct)

    await update(status="running", started_at=_now(), stage="text_extraction", progress=0)
    try:
        report = await processor.ingest(pdf_path, meta, progress=progress)
    except ParseError as exc:
        await update(status="failed", finished_at=_now(), error_code=exc.code,
                     error_message="; ".join([exc.message, *exc.problems])[:2000])
        logger.warning("legal_ingest_failed act_id=%s code=%s", meta.act_id, exc.code)
        return None
    except VerificationError as exc:
        await update(status="failed", finished_at=_now(), error_code="VERIFICATION_FAILED",
                     error_message="; ".join(exc.problems)[:2000])
        return None
    except AppError as exc:
        dep = str(exc.details.get("dependency", "")).upper()
        code = f"{dep}_UNAVAILABLE" if dep else exc.code.value
        await update(status="failed", finished_at=_now(), error_code=code, error_message=exc.message)
        return None
    except Exception as exc:  # the stack goes to logs only
        logger.exception("legal_ingest_error act_id=%s type=%s", meta.act_id, type(exc).__name__)
        await update(status="failed", finished_at=_now(), error_code="INTERNAL",
                     error_message="Internal error.")
        return None
    status = "succeeded_with_warnings" if report.warnings else "succeeded"
    await update(status=status, finished_at=_now(), stage="done", progress=100,
                 warnings=report.warnings[:MAX_JOB_WARNINGS])
    return report
