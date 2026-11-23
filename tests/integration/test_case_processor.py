"""Case processing B3 to B10 against a throwaway Neo4j database and the real Ollama
(07 T3.7 checkpoint and T3.8 machinery): chunks are written, vector and full-text search
over the document work, a re-run changes no node or relationship count (R-16), scope keeps
other users out (R-01), and E-10 clean-up removes the document's nodes.

The LLM is scripted here so re-runs are exactly repeatable; tests/integration/
test_case_linking.py runs the real Groq model."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import get_settings
from app.errors import AppError
from app.ingestion.case.case_processor import CaseGone, CaseJob, CaseProcessor, DocumentFailed
from app.schemas.extraction import ExtractionResult, LinkVerification, SameEntityVerdict
from app.schemas.graph_schema import case_node_id, document_node_id, job_node_id, new_uuid4
from app.services.llm.groq_client import LLMResult
from app.services.media.ocr_service import OcrService
from app.services.media.transcription_service import TranscriptionService
from app.services.retrieval import graph_queries
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.neo4j_service import Neo4jService
from app.services.retrieval.scope import CaseScope, ScopedGraph
from scripts.init_neo4j import init_schema

pytestmark = [pytest.mark.neo4j, pytest.mark.ollama]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "case"
USER = "local"


@pytest.fixture(scope="module")
def db(neo4j_test_db: Neo4jService) -> Neo4jService:
    init_schema(neo4j_test_db)
    return neo4j_test_db


@pytest.fixture
async def embeddings() -> AsyncIterator[EmbeddingService]:
    s = get_settings()
    service = EmbeddingService(s.OLLAMA_URL, s.EMBED_MODEL, s.EMBED_DIM)
    try:
        await service.check_model_served()
    except AppError:
        await service.aclose()
        pytest.skip("Ollama does not serve EMBED_MODEL")
    yield service
    await service.aclose()


class NoGpuModels:
    """Stands in for OCR and Whisper in tests that must not touch the GPU."""

    def __getattr__(self, name: str):
        raise AssertionError(f"GPU model used unexpectedly ({name})")


class ScriptedLLM:
    """Deterministic stand-in for Groq. Extraction cites the first chunk of each batch."""

    model = "scripted"

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    async def complete_json_resuming(self, messages, schema, *, temperature, max_tokens, checkpoint):
        self.calls[schema.__name__] = self.calls.get(schema.__name__, 0) + 1
        if schema is ExtractionResult:
            first = "C1"  # A-29: the batch label of its first chunk
            value = ExtractionResult.model_validate({
                "parties": [
                    {"name": "Ravi Mehta", "kind": "person", "role": "complainant"},
                    {"name": "QuickKart Seller Support", "kind": "organisation", "role": "opposite_party"},
                ],
                "events": [{"description": "Seller refused a refund", "date": "2026-01-12",
                            "participants": ["Ravi Mehta"], "chunk_ids": [first]},
                           {"description": "Return requested", "date": "2026-01-07", "chunk_ids": [first]}],
                "claims": [{"text": "The seller refused to refund a defective phone",
                            "claim_type": "deficiency", "asserted_by": "Ravi Mehta", "chunk_ids": [first]}],
                "evidence": [{"kind": "message", "description": "WhatsApp chat with the seller",
                              "supports": [0], "chunk_ids": [first]}],
            })
        elif schema is LinkVerification:
            value = LinkVerification()
        else:
            value = SameEntityVerdict(same=False)
        return LLMResult(value, 0, 0, 1)


def processor(db: Neo4jService, embeddings: EmbeddingService, ocr=None, transcriber=None, llm=None) -> CaseProcessor:
    return CaseProcessor(
        db, embeddings, ocr or NoGpuModels(), transcriber or NoGpuModels(), asyncio.Semaphore(1),
        llm or ScriptedLLM(), get_settings().LINK_MIN_SCORE,
    )


async def new_case(db: Neo4jService, user: str = USER) -> ScopedGraph:
    case_uuid = new_uuid4()
    graph = ScopedGraph(db, CaseScope(case_uuid, user))
    await graph.write(
        graph_queries.CASE_CREATE,
        {"case_node_id": case_node_id(case_uuid), "title": "Zenith X2 refund",
         "created_at": datetime.now(timezone.utc)},
    )
    return graph


async def add_document(
    graph: ScopedGraph, upload_root: Path, fixture: Path, mime: str, media_type: str,
    language: str | None = None,
) -> CaseJob:
    doc_uuid, job_uuid = new_uuid4(), new_uuid4()
    scope = graph.scope
    folder = upload_root / scope.user_id / scope.case_id / doc_uuid
    folder.mkdir(parents=True)
    stored = folder / f"original{fixture.suffix}"
    shutil.copyfile(fixture, stored)
    await graph.write(
        graph_queries.DOC_CREATE_WITH_JOB,
        {
            "doc_node_id": document_node_id(doc_uuid), "doc_id": doc_uuid,
            "job_node_id": job_node_id(job_uuid), "created_at": datetime.now(timezone.utc),
            "doc_props": {
                "filename": fixture.name, "media_type": media_type, "mime": mime,
                "size_bytes": stored.stat().st_size, "sha256": "0" * 64,
                "stored_path": str(stored), "language": language,
                "uploaded_at": datetime.now(timezone.utc),
            },
        },
    )
    return CaseJob(job_uuid, scope.case_id, doc_uuid, scope.user_id)


async def counts(graph: ScopedGraph) -> dict[str, int]:
    rows = await graph.read(graph_queries.CASE_NODE_COUNTS)
    return {":".join(sorted(r["labels"])): r["n"] for r in rows}


async def rel_counts(graph: ScopedGraph) -> dict[str, int]:
    rows = await graph.read(
        "MATCH (a {case_id: $case_id, user_id: $user_id})-[r]->() "
        "RETURN type(r) AS t, count(*) AS n"
    )
    return {r["t"]: r["n"] for r in rows}


async def doc_status(graph: ScopedGraph, job: CaseJob) -> str:
    rows = await graph.read(
        "MATCH (d:CaseDocument {id: $doc_node_id, case_id: $case_id, user_id: $user_id}) "
        "RETURN d.processing_status AS s",
        {"doc_node_id": document_node_id(job.doc_uuid)},
    )
    return rows[0]["s"]


async def test_pdf_is_chunked_embedded_and_searchable(db, embeddings, tmp_path: Path) -> None:
    graph = await new_case(db)
    job = await add_document(
        graph, tmp_path, FIXTURES / "clean" / "invoice_and_letter.pdf", "application/pdf", "pdf"
    )
    result = await processor(db, embeddings).process(job)
    assert result.media_type == "pdf"
    assert result.chunk_count == 2  # one short page each; chunks never cross a page (A-2)
    # The scripted party 'QuickKart Seller Support' is not in the PDF, so R-15 drops it.
    assert result.warnings == ["extraction: 1 item(s) without a valid chunk id were discarded (R-15)"]
    assert await doc_status(graph, job) == "ready_with_warnings"

    query_vec = await embeddings.embed_query("seller refused to refund the defective phone")
    hits = await graph.read(graph_queries.CASE_VECTOR_SEARCH, {"fetch_k": 50, "k": 5, "query_vec": query_vec})
    assert hits and all(h["doc_id"] == job.doc_uuid for h in hits)
    assert hits[0]["page"] == 2  # the complaint letter page
    assert hits[0]["id"] == f"chk:{job.doc_uuid}:0001"

    ft = await graph.read(graph_queries.CASE_FULLTEXT_SEARCH, {"lucene": '"Zenith"', "fetch_k": 50, "k": 5})
    assert {h["page"] for h in ft} == {1, 2}
    ft = await graph.read(graph_queries.CASE_FULLTEXT_SEARCH, {"lucene": '"warranty"', "fetch_k": 50, "k": 5})
    assert [h["page"] for h in ft] == [1]

    # R-01: another user's scope over the same case id sees nothing.
    other = ScopedGraph(db, CaseScope(graph.scope.case_id, "someone-else"))
    assert await other.read(graph_queries.CASE_VECTOR_SEARCH, {"fetch_k": 50, "k": 5, "query_vec": query_vec}) == []
    assert await other.read(graph_queries.CASE_FULLTEXT_SEARCH, {"lucene": '"Zenith"', "fetch_k": 50, "k": 5}) == []


async def test_rerun_leaves_node_counts_unchanged(db, embeddings, tmp_path: Path) -> None:
    graph = await new_case(db)
    job = await add_document(graph, tmp_path, FIXTURES / "clean" / "chat_seller.txt", "text/plain", "text")
    proc = processor(db, embeddings)
    first = await proc.process(job)
    before, rels_before = await counts(graph), await rel_counts(graph)
    second = await proc.process(job)
    assert await counts(graph) == before
    assert await rel_counts(graph) == rels_before
    assert first.chunk_count == second.chunk_count >= 1
    assert before == {"Case": 1, "CaseDocument": 1, "Job": 1, "CaseChunk": first.chunk_count,
                      "Party": 2, "Event": 2, "Claim": 1, "Evidence": 1}
    assert rels_before["PARTICIPATED_IN"] == 1
    assert rels_before["SUPPORTS"] == 1
    assert rels_before["BEFORE"] == 1  # 2026-01-07 before 2026-01-12, computed in code
    assert rels_before["MENTIONS"] >= 6


async def test_entities_are_case_wide_and_r16_cleans_up_after_a_document(db, embeddings, tmp_path: Path) -> None:
    graph = await new_case(db)
    proc = processor(db, embeddings)
    chat = await add_document(graph, tmp_path, FIXTURES / "clean" / "chat_seller.txt", "text/plain", "text")
    pdf = await add_document(graph, tmp_path, FIXTURES / "clean" / "invoice_and_letter.pdf", "application/pdf", "pdf")
    await proc.process(chat)
    await proc.process(pdf)
    one_party = await graph.read(
        "MATCH (p:Party {name: 'Ravi Mehta'}) WHERE p.case_id = $case_id AND p.user_id = $user_id "
        "MATCH (k:CaseChunk)-[:MENTIONS]->(p) RETURN count(DISTINCT p) AS parties, count(DISTINCT k.doc_id) AS docs"
    )
    assert one_party == [{"parties": 1, "docs": 2}]  # one node per case, mentioned by both
    # E-10 for the PDF: its chunks go, the shared entities stay.
    await proc.purge_document(pdf)
    after = await counts(graph)
    assert after["Party"] == 2 and after["CaseChunk"] == 1


async def test_chat_chunks_keep_sender_and_timestamp(db, embeddings, tmp_path: Path) -> None:
    graph = await new_case(db)
    job = await add_document(graph, tmp_path, FIXTURES / "clean" / "chat_seller.txt", "text/plain", "text")
    await processor(db, embeddings).process(job)
    ft = await graph.read(graph_queries.CASE_FULLTEXT_SEARCH, {"lucene": '"non-refundable"', "fetch_k": 50, "k": 5})
    assert ft
    assert "[2026-01-12 14:36:00] QuickKart Seller Support: Sorry for the trouble." in ft[0]["text"]
    assert ft[0]["page"] is None and ft[0]["start_sec"] is None


async def test_deleting_case_stops_the_job_and_purge_removes_chunks(db, embeddings, tmp_path: Path) -> None:
    graph = await new_case(db)
    job = await add_document(graph, tmp_path, FIXTURES / "clean" / "chat_seller.txt", "text/plain", "text")
    proc = processor(db, embeddings)
    await proc.process(job)
    assert (await counts(graph))["CaseChunk"] >= 1
    await graph.write("MATCH (c:Case {case_id: $case_id, user_id: $user_id}) SET c.status = 'deleting'")
    with pytest.raises(CaseGone):
        await proc.process(job)
    await proc.purge_document(job)
    left = await counts(graph)
    assert not {"CaseChunk", "Party", "Event", "Claim", "Evidence"} & set(left)


async def test_loader_failure_marks_the_document_failed(db, embeddings, tmp_path: Path) -> None:
    blank = tmp_path / "blank.txt"
    blank.write_text("   \n\n  ", encoding="utf-8")
    graph = await new_case(db)
    job = await add_document(graph, tmp_path / "uploads", blank, "text/plain", "text")
    with pytest.raises(DocumentFailed) as exc:
        await processor(db, embeddings).process(job)
    assert exc.value.code == "TEXT_EMPTY"
    assert await doc_status(graph, job) == "failed"


@pytest.mark.gpu
async def test_image_and_audio_through_the_gpu_semaphore(db, embeddings, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    s = get_settings()
    ocr, whisper = OcrService(s.OCR_LANGS), TranscriptionService(s.WHISPER_MODEL)
    proc = processor(db, embeddings, ocr, whisper)
    graph = await new_case(db)
    image = await add_document(graph, tmp_path, FIXTURES / "noisy" / "service_job_sheet.png", "image/png", "image")
    audio = await add_document(graph, tmp_path, FIXTURES / "clean" / "voice_note.wav", "audio/wav", "audio")
    # Run both at once: the semaphore serialises the GPU stages (1.2).
    img_res, aud_res = await asyncio.gather(proc.process(image), proc.process(audio))
    # Scripted extraction names a party that is not in these files: R-15 warnings only.
    assert img_res.processing_status.startswith("ready") and img_res.chunk_count == 1
    assert aud_res.processing_status.startswith("ready") and aud_res.chunk_count >= 1
    assert all("R-15" in w for w in img_res.warnings + aud_res.warnings)
    rows = await graph.read(
        "MATCH (d:CaseDocument {case_id: $case_id, user_id: $user_id}) "
        "RETURN d.media_type AS m, d.page_count AS p, d.duration_sec AS dur, d.language AS lang"
    )
    by_media = {r["m"]: r for r in rows}
    assert by_media["image"]["p"] == 1
    assert by_media["audio"]["lang"] == "en" and 15 < by_media["audio"]["dur"] < 60
    query_vec = await embeddings.embed_query("service centre confirmed manufacturing defect")
    hits = await graph.read(graph_queries.CASE_VECTOR_SEARCH, {"fetch_k": 50, "k": 3, "query_vec": query_vec})
    assert {h["doc_id"] for h in hits} == {image.doc_uuid, audio.doc_uuid}
    audio_hit = next(h for h in hits if h["doc_id"] == audio.doc_uuid)
    assert audio_hit["start_sec"] is not None and audio_hit["end_sec"] > audio_hit["start_sec"]
