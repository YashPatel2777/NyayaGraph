"""Case documents, B1 to B10 (1.4 B).

B1 and B2 (store_document, at the end of this file): validate the upload, check the case,
detect a duplicate (E-09), store the file under UPLOAD_DIR/{user_id}/{case_id}/{doc_id}/
and create the CaseDocument and its Job in one transaction. The case records behind
/v1/cases live there too.

CaseProcessor runs the job's steps: B3 text extraction, B4 chunking, B5 embedding and CaseChunk writes, B6 extraction
(extractor.py), B7 entity resolution (entity_resolver.py), B8 entity, MENTIONS,
relationship and BEFORE writes, B9 legal linking (legal_linker.py), B10 the document's
final status.

- 1.2: OCR and Whisper run inside the GPU semaphore; every blocking call goes through
  anyio.to_thread.run_sync.
- R-01: every read and write goes through ScopedGraph with the job's case_id and user_id,
  which come from the Job node the API created (R-02).
- R-07, R-08: only EmbeddingService makes vectors.
- R-16: chunk ids are deterministic and writes MERGE, so a re-run changes nothing; chunks
  a re-run no longer produces are deleted. Before extraction re-runs, the document's
  MENTIONS and INVOKES edges go, then any case entity left with no MENTIONS.
- E-10: the case status is checked at every stage boundary and before each write; a case
  that is deleting or gone raises CaseGone, and the job worker cancels the job and calls
  purge_document.
- A-22: any job warning makes the document ready_with_warnings.
- R-04: logs carry ids, counts and timings, never text.

The CaseDocument's state is owned here; the Job node is owned by app/ingestion/jobs.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import anyio

from app.errors import AppError, ErrorCode
from app.ingestion.case.entity_resolver import (
    EntityResolver,
    PartyMention,
    PartyRecord,
    ResolvedEntities,
    before_edges,
)
from app.ingestion.case.extractor import CachedReplies, ChunkText, extract_document
from app.ingestion.case.legal_linker import LegalLinker
from app.ingestion.chunking.splitter import Chunk, split_units
from app.ingestion.loaders import LoadedDocument, LoaderError
from app.ingestion.loaders import pdf as pdf_loader
from app.ingestion.loaders.audio import load_audio
from app.ingestion.loaders.image import load_image
from app.ingestion.loaders.router import (
    UploadType,
    check_content_limits,
    check_signature,
    check_size,
    route_stored,
)
from app.ingestion.loaders.text import load_text
from app.prompts.entity_adjudicate import ENTITY_ADJUDICATE_SYSTEM, ENTITY_ADJUDICATE_USER
from app.schemas.extraction import SameEntityVerdict
from app.schemas.graph_schema import (
    DOCUMENT_PREFIX,
    bare_uuid,
    case_node_id,
    chunk_id,
    document_node_id,
    job_node_id,
    new_uuid4,
)
from app.services.llm.groq_client import GroqClient
from app.services.media.ocr_service import OcrService
from app.services.media.transcription_service import TranscriptionService
from app.services.retrieval import graph_queries
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.neo4j_service import Neo4jService
from app.services.retrieval.scope import CaseScope, ScopedGraph

logger = logging.getLogger(__name__)

Progress = Callable[[str, int], Awaitable[None]]

# Stage names from 3.4 and their progress percentages.
STAGE_PROGRESS = {
    "text_extraction": 10,
    "chunking": 40,
    "embedding": 50,
    "entity_extraction": 65,
    "entity_resolution": 80,
    "legal_linking": 90,
}


ADJUDICATE_TEMPERATURE = 0.0  # R-19 (0 for every ingestion call)
ADJUDICATE_MAX_TOKENS = 200  # A-27: R-19 sets no cap for this check


class CaseGone(Exception):
    """E-10: the case is deleting or no longer exists."""


class DocumentFailed(Exception):
    """The document cannot be processed; code is the job and document error_code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class CaseJob:
    job_uuid: str
    case_uuid: str
    doc_uuid: str
    user_id: str


@dataclass
class DocumentResult:
    media_type: str
    chunk_count: int
    processing_status: str
    warnings: list[str] = field(default_factory=list)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _no_progress(stage: str, pct: int) -> None:
    return None


def dependency_error_code(exc: AppError) -> str:
    """OLLAMA_UNAVAILABLE, NEO4J_UNAVAILABLE and so on (E-03, E-04); else the AppError code."""
    dep = str(exc.details.get("dependency", "")).upper()
    return f"{dep}_UNAVAILABLE" if dep else exc.code.value


class CaseProcessor:
    def __init__(
        self,
        neo4j: Neo4jService,
        embeddings: EmbeddingService,
        ocr: OcrService,
        transcriber: TranscriptionService,
        gpu: asyncio.Semaphore,
        llm: GroqClient,
        link_min_score: float,
    ) -> None:
        self.neo4j = neo4j
        self.embeddings = embeddings
        self.ocr = ocr
        self.transcriber = transcriber
        self.gpu = gpu  # 1.2: at most one job in a GPU stage
        self.llm = llm
        self.linker = LegalLinker(neo4j, embeddings, llm, link_min_score)

    def graph(self, job: CaseJob) -> ScopedGraph:
        return ScopedGraph(self.neo4j, CaseScope(job.case_uuid, job.user_id))

    async def ensure_active(self, graph: ScopedGraph) -> None:
        rows = await graph.read(graph_queries.CASE_STATUS)
        if not rows or rows[0]["status"] != "active":
            raise CaseGone()

    async def _update_doc(self, graph: ScopedGraph, job: CaseJob, **props: object) -> None:
        await graph.write(
            graph_queries.DOC_UPDATE,
            {"doc_node_id": document_node_id(job.doc_uuid), "props": props},
        )

    async def process(self, job: CaseJob, progress: Progress = _no_progress) -> DocumentResult:
        """Run B3 to B10 for one document. Raises CaseGone (E-10) or DocumentFailed; on
        DocumentFailed the document is already marked failed."""
        graph = self.graph(job)
        start = time.perf_counter()
        try:
            return await self._process(job, graph, progress, start)
        except (CaseGone, asyncio.CancelledError):
            raise
        except LoaderError as exc:
            await self._fail(graph, job, exc.code)
            raise DocumentFailed(exc.code, exc.message) from exc
        except AppError as exc:
            code = dependency_error_code(exc)
            await self._fail(graph, job, code)
            raise DocumentFailed(code, exc.message) from exc

    async def _fail(self, graph: ScopedGraph, job: CaseJob, code: str) -> None:
        try:
            await self._update_doc(graph, job, processing_status="failed", error_code=code)
        except AppError:  # Neo4j is down too; the job worker records what it can
            logger.warning("doc_fail_unrecorded doc_id=%s code=%s", job.doc_uuid, code)

    async def _process(
        self, job: CaseJob, graph: ScopedGraph, progress: Progress, start: float
    ) -> DocumentResult:
        doc_node = document_node_id(job.doc_uuid)
        rows = await graph.read(graph_queries.DOC_FOR_PROCESSING, {"doc_node_id": doc_node})
        if not rows:
            raise CaseGone()
        record = rows[0]
        await self.ensure_active(graph)
        await self._update_doc(graph, job, processing_status="processing", error_code=None)

        # B3
        await progress("text_extraction", STAGE_PROGRESS["text_extraction"])
        loaded = await self._load(record["mime"], record["stored_path"], record["language"])
        await self.ensure_active(graph)

        # B4
        await progress("chunking", STAGE_PROGRESS["chunking"])
        chunks = split_units(loaded.units, loaded.kind)
        await self.ensure_active(graph)

        # B5
        await progress("embedding", STAGE_PROGRESS["embedding"])
        vectors = await self.embeddings.embed_documents([c.text for c in chunks])
        await self.ensure_active(graph)
        await self._write_chunks(graph, job, chunks, vectors)

        warnings = list(loaded.warnings)
        if chunks:
            folder = Path(record["stored_path"]).parent
            warnings += await self._entities_and_links(job, graph, chunks, progress, folder)

        # B10
        status = "ready_with_warnings" if warnings else "ready"
        await self.ensure_active(graph)
        await self._update_doc(
            graph,
            job,
            processing_status=status,
            error_code=None,
            page_count=loaded.page_count,
            duration_sec=loaded.duration_sec,
            language=loaded.language or record["language"],
        )
        logger.info(
            "case_doc_done doc_id=%s media=%s chunks=%d warnings=%d ms=%d",
            job.doc_uuid, loaded.media_type, len(chunks), len(warnings),
            (time.perf_counter() - start) * 1000,
        )
        return DocumentResult(loaded.media_type, len(chunks), status, warnings)

    async def _load(self, mime: str, path: str, language: str | None) -> LoadedDocument:
        kind = await anyio.to_thread.run_sync(route_stored, mime, path)
        if kind.media_type == "pdf":
            pages = await anyio.to_thread.run_sync(pdf_loader.read_pdf_text, path)
            if pdf_loader.needs_ocr(pages):
                async with self.gpu:
                    await anyio.to_thread.run_sync(pdf_loader.ocr_pages, path, pages, self.ocr)
            return pdf_loader.to_document(pages)
        if kind.media_type == "image":
            async with self.gpu:
                return await anyio.to_thread.run_sync(load_image, path, self.ocr)
        if kind.media_type == "audio":
            async with self.gpu:
                return await anyio.to_thread.run_sync(
                    partial(load_audio, path, self.transcriber, language)
                )
        return await anyio.to_thread.run_sync(load_text, path)

    async def _write_chunks(
        self, graph: ScopedGraph, job: CaseJob, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        ids = [chunk_id(job.doc_uuid, c.order) for c in chunks]
        rows = [
            {
                "id": cid,
                "props": {
                    "order": c.order,
                    "text": c.text,
                    "embedding": vec,
                    "embed_model": self.embeddings.model,
                    "page": c.page,
                    "start_sec": c.start_sec,
                    "end_sec": c.end_sec,
                },
            }
            for cid, c, vec in zip(ids, chunks, vectors, strict=True)
        ]
        params = {"doc_node_id": document_node_id(job.doc_uuid), "doc_id": job.doc_uuid}
        await graph.write_batches(graph_queries.CHUNKS_MERGE, rows, params)
        await graph.write(
            graph_queries.CHUNKS_DELETE_STALE, {"doc_id": job.doc_uuid, "keep_ids": ids}
        )

    async def _entities_and_links(
        self,
        job: CaseJob,
        graph: ScopedGraph,
        chunks: list[Chunk],
        progress: Progress,
        folder: Path,
    ) -> list[str]:
        """B6 to B9. Returns the job warnings they produce. Every Groq reply goes through
        the document's reply cache (SC-12)."""
        replies = await CachedReplies.open(self.llm, folder)

        async def checkpoint() -> None:
            await self.ensure_active(graph)

        doc = {"doc_id": job.doc_uuid}
        texts = [ChunkText(chunk_id(job.doc_uuid, c.order), c.text) for c in chunks]

        # R-16: start this document's extraction from a clean slate.
        await graph.write(graph_queries.DOC_MENTIONS_DELETE, doc)
        await graph.write(graph_queries.DOC_INVOKES_DELETE, doc)
        await graph.write(graph_queries.CASE_ORPHAN_ENTITIES_DELETE)

        # B6
        await progress("entity_extraction", STAGE_PROGRESS["entity_extraction"])
        extraction = await extract_document(replies, texts, checkpoint)

        # B7
        await progress("entity_resolution", STAGE_PROGRESS["entity_resolution"])
        await checkpoint()
        existing = [
            PartyRecord(r["id"], r["name"], r["kind"], r["role"], list(r["aliases"] or []))
            for r in await graph.read(graph_queries.CASE_PARTIES)
        ]
        failed_adjudications = 0

        async def adjudicate(party: PartyRecord, mention: PartyMention) -> bool:
            nonlocal failed_adjudications
            user = ENTITY_ADJUDICATE_USER.format(
                kind=party.kind,
                name_a=party.name, aliases_a=", ".join(party.aliases) or "none",
                name_b=mention.name, aliases_b=", ".join(mention.aliases) or "none",
            )
            try:
                reply = await replies.complete_json_resuming(
                    [("system", ENTITY_ADJUDICATE_SYSTEM), ("user", user)],
                    SameEntityVerdict,
                    temperature=ADJUDICATE_TEMPERATURE,
                    max_tokens=ADJUDICATE_MAX_TOKENS,
                    checkpoint=checkpoint,
                )
            except AppError as exc:
                if exc.details.get("reason") != "invalid_structured_output":
                    raise
                failed_adjudications += 1
                return False  # R-14: not shown to be the same, so they stay separate
            return reply.value.same

        resolver = EntityResolver(job.case_uuid, existing, adjudicate)
        resolved = await resolver.resolve(extraction.batches)

        # B8
        await checkpoint()
        await self._write_entities(graph, resolved)
        events = await graph.read(graph_queries.CASE_EVENT_DATES)
        await graph.write(graph_queries.CASE_BEFORE_DELETE)
        edges = before_edges((r["id"], r["date"]) for r in events)
        await graph.write_batches(
            graph_queries.BEFORE_MERGE, [{"source": a, "target": b} for a, b in edges]
        )

        # B9
        await progress("legal_linking", STAGE_PROGRESS["legal_linking"])
        invokes = await self.linker.citation_links(texts)
        claims = [(cid, rec.props["text"]) for cid, rec in resolved.claims.items()]
        alleges, link_warnings, link_calls = await self.linker.claim_links(claims, checkpoint, replies)
        await checkpoint()
        await graph.write_batches(graph_queries.INVOKES_MERGE, invokes)
        await graph.write(graph_queries.CLAIM_ALLEGES_DELETE, {"claim_ids": [c for c, _ in claims]})
        await graph.write_batches(graph_queries.ALLEGES_MERGE, alleges)
        await replies.prune()

        logger.info(
            "case_doc_entities doc_id=%s batches=%d failed_batches=%d parties=%d events=%d "
            "claims=%d evidence=%d adjudications=%d before=%d invokes=%d alleges=%d llm_calls=%d "
            "cache_hits=%d",
            job.doc_uuid, len(extraction.batches) + extraction.failed_batches,
            extraction.failed_batches, len(resolved.parties), len(resolved.events),
            len(resolved.claims), len(resolved.evidence), resolved.adjudications, len(edges),
            len(invokes), len(alleges), extraction.llm_calls + resolved.adjudications + link_calls,
            replies.hits,
        )
        warnings = extraction.warnings + resolved.warnings + link_warnings
        if failed_adjudications:
            warnings.append(
                f"entity_resolution: {failed_adjudications} name check(s) failed and the "
                "names were kept separate"
            )
        return warnings

    async def _write_entities(self, graph: ScopedGraph, resolved: ResolvedEntities) -> None:
        parties = [
            {"id": p.id, "chunk_ids": sorted(p.chunk_ids),
             "props": {"name": p.name, "aliases": p.aliases, "kind": p.kind, "role": p.role}}
            for p in resolved.parties.values()
        ]
        await graph.write_batches(graph_queries.ENTITY_MERGE_PARTIES, parties)
        for query, store in (
            (graph_queries.ENTITY_MERGE_EVENTS, resolved.events),
            (graph_queries.ENTITY_MERGE_CLAIMS, resolved.claims),
            (graph_queries.ENTITY_MERGE_EVIDENCE, resolved.evidence),
        ):
            rows = [{"id": e.id, "props": e.props, "chunk_ids": sorted(e.chunk_ids)}
                    for e in store.values()]
            await graph.write_batches(query, rows)
        for query, pairs in (
            (graph_queries.PARTICIPATED_IN_MERGE, resolved.participated),
            (graph_queries.SUPPORTS_MERGE, resolved.supports),
            (graph_queries.CONTRADICTS_MERGE, resolved.contradicts),
        ):
            await graph.write_batches(query, [{"source": a, "target": b} for a, b in sorted(pairs)])

    async def purge_document(self, job: CaseJob) -> None:
        """E-10: remove the partial nodes of a cancelled document: its chunks (with their
        MENTIONS and INVOKES edges) and any entity no other document mentions."""
        graph = self.graph(job)
        await graph.write(graph_queries.CHUNKS_DELETE_FOR_DOC, {"doc_id": job.doc_uuid})
        await graph.write(graph_queries.CASE_ORPHAN_ENTITIES_DELETE)


# B1, B2 and the case records behind /v1/cases (3.4). user_id always comes from
# deps.get_current_user() (R-02).

MAX_DOCUMENTS_PER_CASE = 200  # 3.4
FILENAME_MAX = 255


def _native(value: object) -> object:
    """neo4j.time values to Python datetimes and dates."""
    to_native = getattr(value, "to_native", None)
    return to_native() if callable(to_native) else value


def _record(row: dict) -> dict:
    return {k: _native(v) for k, v in row.items()}


def _scoped(neo4j: Neo4jService, case_uuid: str, user_id: str) -> ScopedGraph:
    return ScopedGraph(neo4j, CaseScope(case_uuid, user_id))


async def create_case(neo4j: Neo4jService, user_id: str, title: str) -> dict:
    case_uuid = new_uuid4()
    graph = _scoped(neo4j, case_uuid, user_id)
    rows = await graph.write(
        graph_queries.CASE_CREATE,
        {"case_node_id": case_node_id(case_uuid), "title": title, "created_at": _utcnow()},
    )
    return _record(rows[0])


async def get_case(neo4j: Neo4jService, user_id: str, case_uuid: str) -> dict:
    rows = await _scoped(neo4j, case_uuid, user_id).read(graph_queries.CASE_GET)
    if not rows:
        raise AppError(ErrorCode.CASE_NOT_FOUND)
    return _record(rows[0])


async def list_cases(
    neo4j: Neo4jService, user_id: str, limit: int, cursor: str | None
) -> tuple[list[dict], str | None]:
    rows = await neo4j.read(
        graph_queries.CASE_LIST, {"user_id": user_id, "limit": limit + 1, "cursor": cursor}
    )
    page = [_record(r) for r in rows[:limit]]
    return page, (page[-1]["case_id"] if len(rows) > limit else None)


async def require_active_case(graph: ScopedGraph) -> None:
    rows = await graph.read(graph_queries.CASE_STATUS)
    if not rows:
        raise AppError(ErrorCode.CASE_NOT_FOUND)
    if rows[0]["status"] != "active":
        raise AppError(ErrorCode.CASE_DELETING)  # R-05


@dataclass(frozen=True)
class DocumentAccepted:
    doc_id: str
    job_id: str | None
    processing_status: str
    duplicate: bool


def _write_upload(folder: Path, name: str, data: bytes) -> Path:
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / name
    path.write_bytes(data)
    return path


async def store_document(
    neo4j: Neo4jService,
    user_id: str,
    case_uuid: str,
    filename: str | None,
    kind: UploadType,
    data: bytes,
    language: str | None,
    upload_dir: Path,
) -> DocumentAccepted:
    """B1 and B2. Nothing is written before every check passes (E-13). The caller enqueues
    the returned job."""
    graph = _scoped(neo4j, case_uuid, user_id)
    await require_active_case(graph)
    check_size(kind, len(data))
    check_signature(kind, data)
    await anyio.to_thread.run_sync(check_content_limits, kind, data)
    rows = await graph.read(graph_queries.DOC_COUNT)
    if rows and rows[0]["n"] >= MAX_DOCUMENTS_PER_CASE:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,  # A-30: 3.3 has no code for this limit
            details={"fields": [{"loc": ["path", "case_id"],
                                 "msg": f"a case holds at most {MAX_DOCUMENTS_PER_CASE} documents"}]},
        )
    sha256 = await anyio.to_thread.run_sync(lambda: hashlib.sha256(data).hexdigest())

    # E-09: the same file in the same case returns the existing document and no job.
    dup = await graph.read(graph_queries.DOC_BY_SHA256, {"sha256": sha256})
    if dup:
        return DocumentAccepted(
            bare_uuid(dup[0]["id"], DOCUMENT_PREFIX), None, dup[0]["processing_status"], True
        )

    doc_uuid, job_uuid = new_uuid4(), new_uuid4()
    folder = upload_dir / user_id / case_uuid / doc_uuid  # B2
    stored = await anyio.to_thread.run_sync(
        _write_upload, folder, f"original.{kind.extension}", data
    )
    created = await graph.write(
        graph_queries.DOC_CREATE_WITH_JOB,
        {
            "doc_node_id": document_node_id(doc_uuid), "doc_id": doc_uuid,
            "job_node_id": job_node_id(job_uuid), "created_at": _utcnow(),
            "doc_props": {
                "filename": Path(filename or "upload").name[:FILENAME_MAX] or "upload",
                "media_type": kind.media_type, "mime": kind.mime, "size_bytes": len(data),
                "sha256": sha256, "stored_path": str(stored), "language": language,
                "uploaded_at": _utcnow(),
            },
        },
    )
    if not created:  # the case started deleting after the first check
        await anyio.to_thread.run_sync(lambda: shutil.rmtree(folder, ignore_errors=True))
        raise AppError(ErrorCode.CASE_DELETING)
    logger.info("doc_stored doc_id=%s media=%s bytes=%d", doc_uuid, kind.media_type, len(data))
    return DocumentAccepted(doc_uuid, job_uuid, "queued", False)


async def get_document(neo4j: Neo4jService, user_id: str, case_uuid: str, doc_uuid: str) -> dict:
    graph = _scoped(neo4j, case_uuid, user_id)
    rows = await graph.read(graph_queries.DOC_GET, {"doc_node_id": document_node_id(doc_uuid)})
    if not rows:
        await get_case(neo4j, user_id, case_uuid)  # 404 CASE_NOT_FOUND first
        raise AppError(ErrorCode.DOCUMENT_NOT_FOUND)
    return _doc(rows[0])


async def list_documents(
    neo4j: Neo4jService, user_id: str, case_uuid: str, limit: int, cursor: str | None
) -> tuple[list[dict], str | None]:
    await get_case(neo4j, user_id, case_uuid)
    graph = _scoped(neo4j, case_uuid, user_id)
    node_cursor = document_node_id(cursor) if cursor else None
    rows = await graph.read(graph_queries.DOC_LIST, {"limit": limit + 1, "cursor": node_cursor})
    page = [_doc(r) for r in rows[:limit]]
    return page, (page[-1]["doc_id"] if len(rows) > limit else None)


def _doc(row: dict) -> dict:
    record = _record(row)
    record["doc_id"] = bare_uuid(record.pop("id"), DOCUMENT_PREFIX)
    return record
