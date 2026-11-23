"""B6 to B9 with the real Groq model and the real CPA 2019 Legal KB (07 T3.8, T3.9).

A throwaway database gets the CPA (as in the golden tests) and the synthetic complaint
notes and seller chat. Regex citations are exact; LLM output is checked for shape and for
the facts the material states plainly. Slow: CPA ingestion plus a few Groq calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.config import get_settings
from app.ingestion.case.entity_resolver import normalise_name
from app.ingestion.legal.legal_processor import LegalProcessor
from app.schemas.api import LegalIngestMetadata
from app.services.llm.groq_client import GroqClient
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.neo4j_service import Neo4jService
from scripts.ingest_legal import PRESETS
from scripts.init_neo4j import init_schema
from tests.integration.test_case_processor import (
    FIXTURES,
    add_document,
    counts,
    new_case,
    processor,
    rel_counts,
)

pytestmark = [pytest.mark.neo4j, pytest.mark.ollama, pytest.mark.groq]

CPA_PDF = Path(__file__).resolve().parents[2] / "data" / "legal" / "CPA2019.pdf"


@pytest.fixture(scope="module")
def kb(neo4j_test_db: Neo4jService) -> Neo4jService:
    if not CPA_PDF.exists():
        pytest.skip("data/legal/CPA2019.pdf is not present")
    init_schema(neo4j_test_db)

    async def ingest() -> None:
        s = get_settings()
        emb = EmbeddingService(s.OLLAMA_URL, s.EMBED_MODEL, s.EMBED_DIM)
        try:
            meta = LegalIngestMetadata.model_validate(PRESETS["cpa-2019"])
            await LegalProcessor(neo4j_test_db, emb).ingest(str(CPA_PDF), meta)
        finally:
            await emb.aclose()

    asyncio.run(ingest())
    return neo4j_test_db


@pytest.fixture
async def services() -> AsyncIterator[tuple[EmbeddingService, GroqClient]]:
    s = get_settings()
    if s.GROQ_API_KEY.get_secret_value() == "unit-test-placeholder":
        pytest.skip("GROQ_API_KEY is not set")
    emb = EmbeddingService(s.OLLAMA_URL, s.EMBED_MODEL, s.EMBED_DIM)
    yield emb, GroqClient(s.GROQ_API_KEY.get_secret_value(), s.GROQ_MODEL)
    await emb.aclose()


async def test_notes_and_chat_build_the_case_graph(kb, services, tmp_path: Path) -> None:
    embeddings, llm = services
    proc = processor(kb, embeddings, llm=llm)
    graph = await new_case(kb)
    notes = await add_document(graph, tmp_path, FIXTURES / "clean" / "complaint_notes.txt", "text/plain", "text")
    chat = await add_document(graph, tmp_path, FIXTURES / "clean" / "chat_seller.txt", "text/plain", "text")
    for job in (notes, chat):
        result = await proc.process(job)
        assert result.processing_status in {"ready", "ready_with_warnings"}

    # B9 regex citations: exact. Section 420 IPC links nowhere (R-26).
    invokes = await graph.read(
        "MATCH (k:CaseChunk {case_id: $case_id, user_id: $user_id})-[r:INVOKES]->(t) "
        "RETURN t.id AS target, r.text_span AS span ORDER BY target"
    )
    assert [r["target"] for r in invokes] == [
        "cl:cpa-2019:2:10", "cl:cpa-2019:2:11", "cl:cpa-2019:2:47", "cl:cpa-2019:2:7",
        "sec:cpa-2019:35", "sec:cpa-2019:69",
    ]
    assert not any("420" in r["span"] for r in invokes)

    # B6 to B8: one complainant node for Ravi Mehta across both documents (R-14).
    parties = await graph.read(
        "MATCH (p:Party) WHERE p.case_id = $case_id AND p.user_id = $user_id "
        "OPTIONAL MATCH (k:CaseChunk)-[:MENTIONS]->(p) "
        "RETURN p.name AS name, p.aliases AS aliases, p.role AS role, p.kind AS kind, "
        "collect(DISTINCT k.doc_id) AS docs"
    )
    ravi = [p for p in parties if "ravi mehta" in normalise_name(" ".join([p["name"], *p["aliases"]]))]
    assert len(ravi) == 1, parties
    assert ravi[0]["role"] == "complainant" and ravi[0]["kind"] == "person"
    assert set(ravi[0]["docs"]) == {notes.doc_uuid, chat.doc_uuid}
    assert any(p["role"] == "opposite_party" for p in parties)

    claims = await graph.read(
        "MATCH (c:Claim) WHERE c.case_id = $case_id AND c.user_id = $user_id "
        "OPTIONAL MATCH (c)-[r:ALLEGES_VIOLATION_OF]->(t) "
        "RETURN c.id AS id, c.claim_type AS type, "
        "collect({target: t.id, confidence: r.confidence, method: r.method, rationale: r.rationale}) AS links"
    )
    assert claims
    links = [l for c in claims for l in c["links"] if l["target"]]
    assert links, "no claim was linked to a provision"
    for link in links:
        assert link["target"].startswith(("sec:cpa-2019:", "cl:cpa-2019:"))
        assert "-p" not in link["target"]
        assert link["method"] == "vector+llm"
        assert get_settings().LINK_MIN_SCORE <= link["confidence"] <= 1
        assert 0 < len(link["rationale"]) <= 300

    # Every entity has at least one MENTIONS edge (R-15).
    orphans = await graph.read(
        "MATCH (e:Party|Event|Claim|Evidence) WHERE e.case_id = $case_id AND e.user_id = $user_id "
        "AND NOT ()-[:MENTIONS]->(e) RETURN count(e) AS n"
    )
    assert orphans == [{"n": 0}]


async def test_rerun_with_the_live_model_leaves_counts_unchanged(kb, services, tmp_path: Path) -> None:
    # Phase 3 exit with SC-12: the second run replays the cached replies.
    embeddings, llm = services
    proc = processor(kb, embeddings, llm=llm)
    graph = await new_case(kb)
    jobs = [
        await add_document(graph, tmp_path, FIXTURES / "clean" / "complaint_notes.txt", "text/plain", "text"),
        await add_document(graph, tmp_path, FIXTURES / "clean" / "chat_seller.txt", "text/plain", "text"),
    ]
    for job in jobs:
        await proc.process(job)
    nodes, rels = await counts(graph), await rel_counts(graph)
    for job in jobs:
        await proc.process(job)
    assert await counts(graph) == nodes
    assert await rel_counts(graph) == rels
    cache = tmp_path / graph.scope.user_id / graph.scope.case_id / jobs[0].doc_uuid / "llm_cache.json"
    assert cache.exists()
