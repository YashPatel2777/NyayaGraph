"""Golden tests G1 to G10 (05-roadmap.md, Phase 2): the Phase 2 acceptance test.

Ingests data/legal/CPA2019.pdf, the Jurisdiction Rules, 2021 and the E-Commerce Rules, 2020
(Gazette original, text layer; SC-8 option a) into a throwaway database with real
embeddings (Ollama), then checks the graph.

The assertion queries below read the graph to check it; they are test code, not application
Cypher (the R-01 rule and the scan in test_cypher_safety.py cover app/).
"""

from __future__ import annotations

import asyncio
import re
from datetime import date
from pathlib import Path

import pytest

from app.config import get_settings
from app.ingestion.legal.legal_processor import LegalProcessor
from app.schemas.api import LegalIngestMetadata
from app.services.retrieval import graph_queries
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.neo4j_service import Neo4jService
from scripts.ingest_legal import PRESETS
from scripts.init_neo4j import init_schema

pytestmark = pytest.mark.neo4j

ROOT = Path(__file__).resolve().parents[2]
CPA_PDF = ROOT / "data" / "legal" / "CPA2019.pdf"
RULES_PDF = ROOT / "data" / "legal" / (
    "Consumer Protection (Jurisdiction of the District Commission, the State Commission "
    "and the National Commission) Rules, 2021.pdf"
)
ECOM_PDF = ROOT / "data" / "legal" / "E commerce rules.pdf"
CPA = "cpa-2019"
RULES = "cp-jurisdiction-rules-2021"
ECOM = "cp-e-commerce-rules-2020"
AMEND_PDF = ROOT / "data" / "legal" / "2021 May 17.pdf"
AMEND = "cp-e-commerce-amendment-rules-2021"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _embeddings() -> EmbeddingService:
    s = get_settings()
    return EmbeddingService(s.OLLAMA_URL, s.EMBED_MODEL, s.EMBED_DIM)


async def _ingest(db: Neo4jService, pdf: Path, preset: str, replace: bool = False):
    emb = _embeddings()
    try:
        meta = LegalIngestMetadata.model_validate({**PRESETS[preset], "replace": replace})
        return await LegalProcessor(db, emb).ingest(str(pdf), meta)
    finally:
        await emb.aclose()


@pytest.fixture(scope="module")
def kb(neo4j_test_db: Neo4jService) -> Neo4jService:
    for pdf in (CPA_PDF, RULES_PDF, ECOM_PDF, AMEND_PDF):
        if not pdf.exists():
            pytest.skip(f"missing {pdf.name}")
    init_schema(neo4j_test_db)
    asyncio.run(_ingest(neo4j_test_db, CPA_PDF, CPA))
    asyncio.run(_ingest(neo4j_test_db, RULES_PDF, RULES))
    asyncio.run(_ingest(neo4j_test_db, ECOM_PDF, ECOM))
    asyncio.run(_ingest(neo4j_test_db, AMEND_PDF, AMEND))
    return neo4j_test_db


def q(db: Neo4jService, cypher: str, **params) -> list[dict]:
    return db.read_sync(cypher, params)


def _refers(db: Neo4jService, source_section: str, target_section: str) -> bool:
    """A REFERS_TO edge from the source Section (or one of its Clauses) to the target
    Section (or one of its Clauses)."""
    rows = q(
        db,
        """
        MATCH (s:Section {id: $src})
        OPTIONAL MATCH (s)-[:HAS_CLAUSE]->(sc)
        WITH collect(s) + collect(sc) AS sources
        MATCH (t:Section {id: $dst})
        OPTIONAL MATCH (t)-[:HAS_CLAUSE]->(tc)
        WITH sources, collect(t) + collect(tc) AS targets
        UNWIND sources AS a
        MATCH (a)-[:REFERS_TO]->(b)
        WHERE b IN targets
        RETURN count(*) AS n
        """,
        src=f"sec:{CPA}:{source_section}",
        dst=f"sec:{CPA}:{target_section}",
    )
    return rows[0]["n"] > 0


def test_g1_chapters_and_sections(kb) -> None:
    chapters = q(kb, "MATCH (c:Chapter {act_id: $a}) RETURN c.id AS id", a=CPA)
    assert len(chapters) == 8
    ids = [r["id"] for r in q(kb, "MATCH (s:Section {act_id: $a}) RETURN s.id AS id", a=CPA)]
    assert len(ids) == 107
    assert sorted(ids) == sorted(f"sec:{CPA}:{n}" for n in range(1, 108))
    title = q(kb, "MATCH (c:Chapter {id: $id}) RETURN c.title AS t", id=f"ch:{CPA}:IV")[0]["t"]
    assert title == "CONSUMER DISPUTER REDRESSAL COMMISSION"  # P-04: source typo kept


def test_g2_definitions_and_concepts(kb) -> None:
    defs = q(
        kb,
        "MATCH (:Section {id: $s})-[:HAS_CLAUSE]->(c:Clause {kind: 'definition'}) RETURN count(c) AS n",
        s=f"sec:{CPA}:2",
    )
    assert defs[0]["n"] == 47
    concepts = q(kb, "MATCH (k:LegalConcept {act_id: $a}) RETURN count(k) AS n", a=CPA)
    assert concepts[0]["n"] == 47
    consumer = q(
        kb,
        "MATCH (c:Clause)-[:DEFINES]->(k:LegalConcept {id: 'con:consumer'}) "
        "RETURN k.definition_text AS d, c.id AS clause",
    )
    assert consumer and consumer[0]["clause"] == f"cl:{CPA}:2:7"
    text = consumer[0]["d"].replace("“", "'").replace("”", "'")
    assert text.startswith("'consumer' means any person who")


@pytest.mark.parametrize(
    "src, dst",
    [("24", "20"), ("24", "21"), ("49", "35"), ("49", "36"), ("49", "37"), ("49", "38"),
     ("49", "39"), ("73", "72"), ("92", "88"), ("92", "89"), ("97", "21"), ("97", "96")],
)
def test_g3_refers_to(kb, src, dst) -> None:
    assert _refers(kb, src, dst), f"no REFERS_TO from {src} to {dst}"


def test_g4_other_act_reference(kb) -> None:
    ipc = q(
        kb,
        "MATCH (:Section {id: $s})-[r:REFERS_TO_ACT]->(a:Act {id: 'act:indian-penal-code'}) "
        "RETURN a.external AS external, r.section_ref AS ref",
        s=f"sec:{CPA}:38",
    )
    assert ipc and ipc[0]["external"] is True
    assert "193" in ipc[0]["ref"]
    wrong = q(
        kb,
        """
        MATCH (s:Section {id: $s})
        OPTIONAL MATCH (s)-[:HAS_CLAUSE]->(c)
        WITH collect(s) + collect(c) AS sources
        UNWIND sources AS src
        MATCH (src)-[:REFERS_TO]->(t)
        WHERE t.number IN ['193', '228'] OR t.section_number IN ['193', '228']
        RETURN count(*) AS n
        """,
        s=f"sec:{CPA}:38",
    )
    assert wrong[0]["n"] == 0


def test_g5_amendment(kb) -> None:
    notes = q(kb, "MATCH (s:Section {id: $s}) RETURN s.amendment_notes AS n", s=f"sec:{CPA}:55")[0]["n"]
    assert any("Act 33 of 2021" in n for n in notes)
    amends = q(
        kb,
        "MATCH (a:Act {id: 'act:33-of-2021'})-[r:AMENDS]->(:Section {id: $s}) "
        "RETURN r.effective_on AS eff, a.external AS external",
        s=f"sec:{CPA}:55",
    )
    assert amends and amends[0]["external"] is True
    assert amends[0]["eff"].to_native() == date(2021, 4, 4)
    clause = q(kb, "MATCH (c:Clause {id: $c}) RETURN c.text AS t", c=f"cl:{CPA}:55:1A")
    assert clause, "sub-section (1A) after the 1[ marker"


@pytest.mark.parametrize(
    "section, expected",
    [("34", date(2020, 7, 20)), ("90", date(2020, 7, 20)), ("10", date(2020, 7, 24)),
     ("88", date(2020, 7, 24)), ("94", date(2020, 7, 24)), ("1", None)],
)
def test_g6_in_force_from(kb, section, expected) -> None:
    value = q(kb, "MATCH (s:Section {id: $s}) RETURN s.in_force_from AS d", s=f"sec:{CPA}:{section}")[0]["d"]
    assert (value.to_native() if value is not None else None) == expected


def test_g7_objects_text(kb) -> None:
    act = q(kb, "MATCH (a:Act {id: $id}) RETURN a.objects_text AS t", id=f"act:{CPA}")[0]
    assert act["t"] and act["t"].startswith("STATEMENT OF OBJECTS AND REASONS")
    leaked = q(
        kb,
        "MATCH (u:LegalUnit) WHERE u.text CONTAINS 'STATEMENT OF OBJECTS' "
        "OR u.search_text CONTAINS 'STATEMENT OF OBJECTS' RETURN count(u) AS n",
    )
    assert leaked[0]["n"] == 0


def test_g8_no_watermark(kb) -> None:
    rows = q(
        kb,
        """
        MATCH (n)
        WHERE any(k IN keys(n) WHERE k <> 'embedding' AND toStringOrNull(n[k]) CONTAINS 'IndiaCode')
           OR any(x IN coalesce(n.amendment_notes, []) WHERE x CONTAINS 'IndiaCode')
        RETURN count(n) AS n
        """,
    )
    assert rows[0]["n"] == 0


def test_g9_embeddings_indexes_and_search(kb) -> None:
    bad = q(
        kb,
        "MATCH (u:LegalUnit {act_id: $a}) WHERE u.embedding IS NULL OR size(u.embedding) <> 768 "
        "RETURN count(u) AS n",
        a=CPA,
    )
    assert bad[0]["n"] == 0
    units = q(kb, "MATCH (u:LegalUnit {act_id: $a}) RETURN count(u) AS n", a=CPA)
    assert units[0]["n"] > 100
    states = q(kb, graph_queries.INDEX_STATES, names=["legal_unit_embedding", "case_chunk_embedding",
                                                      "legal_text_ft", "case_text_ft"])
    assert {r["state"] for r in states} == {"ONLINE"} and len(states) == 4

    async def search() -> list[str]:
        emb = _embeddings()
        try:
            vec = await emb.embed_query("limitation period for filing a consumer complaint")
        finally:
            await emb.aclose()
        rows = await kb.read(graph_queries.LEGAL_VECTOR_SEARCH, {"k": 3, "query_vec": vec})
        return [r["id"] for r in rows]

    assert f"sec:{CPA}:69" in asyncio.run(search())


def test_g10_jurisdiction_rules(kb) -> None:
    rules = q(kb, "MATCH (s:Section {act_id: $a}) RETURN s.id AS id", a=RULES)
    assert sorted(r["id"] for r in rules) == [f"sec:{RULES}:{n}" for n in range(1, 6)]
    edges = q(
        kb,
        "MATCH (r:Section {act_id: $a})-[:PRESCRIBES_FOR]->(s:Section) "
        "RETURN r.number AS rule, s.id AS section ORDER BY rule",
        a=RULES,
    )
    assert [(e["rule"], e["section"]) for e in edges] == [
        ("3", f"sec:{CPA}:34"), ("4", f"sec:{CPA}:47"), ("5", f"sec:{CPA}:58"),
    ]
    texts = q(
        kb,
        "MATCH (n) WHERE n.act_id IN [$a, $b] OR n.id IN [$ra, $ca] "
        "RETURN n.id AS id, [k IN keys(n) WHERE k <> 'embedding' | toStringOrNull(n[k])] AS vals",
        a=RULES, b=CPA, ra=f"act:{RULES}", ca=f"act:{CPA}",
    )
    with_devanagari = [t["id"] for t in texts if any(v and DEVANAGARI.search(v) for v in t["vals"])]
    assert with_devanagari == []
    rule3 = q(kb, "MATCH (s:Section {id: $s}) RETURN s.text AS t", s=f"sec:{RULES}:3")[0]["t"]
    assert "fifty lakh rupees" in rule3
    act = q(kb, "MATCH (a:Act {id: $id}) RETURN a.kind AS kind, a.issued_under_section AS secs, "
                "a.enabling_text AS en", id=f"act:{RULES}")[0]
    assert act["kind"] == "rule" and act["secs"] == ["34", "47", "58"]
    assert act["en"].startswith("In exercise of the powers conferred by")  # P-14


def test_g11_e_commerce_rules(kb) -> None:
    rules = q(kb, "MATCH (s:Section {act_id: $a}) RETURN s.id AS id", a=ECOM)
    assert sorted(r["id"] for r in rules) == [f"sec:{ECOM}:{n}" for n in range(1, 9)]
    it_act = q(
        kb,
        "MATCH (:Section {id: $s})-[:REFERS_TO_ACT]->(a:Act {id: 'act:information-technology-act-2000'}) "
        "RETURN a.external AS external",
        s=f"sec:{ECOM}:5",
    )
    assert it_act and it_act[0]["external"] is True
    # Beyond G11: every sub-rule and definition label survives (the OCR scan lost ~25).
    labels = q(kb, "MATCH (:Section {id: $s})-[:HAS_CLAUSE]->(c:Clause) RETURN c.label AS l", s=f"sec:{ECOM}:4")
    assert sorted(int(r["l"]) for r in labels) == list(range(1, 12))
    terms = q(kb, "MATCH (:Section {id: $s})-[:HAS_CLAUSE]->(c:Clause {kind: 'definition'}) "
                  "RETURN c.label AS l, c.term AS t ORDER BY l", s=f"sec:{ECOM}:3")
    assert [r["l"] for r in terms] == list("abcdefghijkl")
    assert {r["t"] for r in terms} >= {"PAN", "inventory e-commerce entity", "seller"}
    # P-17: 'clause (37) of section 2 of the Act' is the CPA's definition of product seller.
    seller = q(kb, "MATCH (:Clause {id: $c})-[:REFERS_TO]->(t) RETURN t.id AS id", c=f"cl:{ECOM}:3:k")
    assert {r["id"] for r in seller} == {f"cl:{CPA}:2:37"}
    gazette = q(kb, "MATCH (n) WHERE n.act_id = $a AND n.text CONTAINS 'GAZETTE OF INDIA' RETURN count(n) AS n", a=ECOM)
    assert gazette[0]["n"] == 0
    act = q(kb, "MATCH (a:Act {id: $id}) RETURN a.issued_under_section AS s, a.as_on_date AS d", id=f"act:{ECOM}")[0]
    # 2020-07-23 as ingested; the fixture then applies G.S.R. 328(E), which moves it (P-20).
    assert act["s"] == ["101"] and act["d"].to_native() == date(2021, 5, 17)


def _rule_4_1(db: Neo4jService) -> dict:
    return q(
        db,
        """
        MATCH (s:Section {id: $s})-[:HAS_CLAUSE]->(c:Clause {id: $c})
        MATCH (a:Act {id: $act})
        OPTIONAL MATCH (n:Act {id: $amend})-[r:AMENDS]->(s)
        RETURN c.text AS clause, s.text AS section, s.amendment_notes AS notes,
               a.as_on_date AS as_on, r.effective_on AS eff, r.previous_text AS prev,
               c.embedding IS NOT NULL AS embedded
        """,
        s=f"sec:{ECOM}:4", c=f"cl:{ECOM}:4:1", act=f"act:{ECOM}", amend=f"act:{AMEND}",
    )[0]


def test_g12_amendment_substitutes_rule_4_1(kb) -> None:
    """SC-9: G.S.R. 328(E) of 17 May 2021 substitutes rule 4(1) of the E-Commerce Rules."""
    r = _rule_4_1(kb)
    assert r["clause"].startswith("where an e-commerce entity is a company incorporated")
    assert "nodal officer" in r["clause"] and r["embedded"]
    assert "be a company incorporated" not in r["section"]  # the 2020 wording is gone
    assert r["section"].startswith("(1) where an e-commerce entity")
    assert r["notes"] == ["Subs. by G.S.R. 328(E), dated 17-5-2021 (w.e.f. 17-5-2021), sub-rule (1)."]
    assert r["eff"].to_native() == date(2021, 5, 17)
    assert r["prev"].startswith("An e-commerce entity shall:")  # P-20 keeps the old text
    assert r["as_on"].to_native() == date(2021, 5, 17)
    notification = q(kb, "MATCH (a:Act {id: $id}) OPTIONAL MATCH (a)-[:HAS_SECTION]->(s) "
                         "RETURN a.kind AS kind, a.amends_act_id AS amends, count(s) AS n, "
                         "sum(CASE WHEN s:LegalUnit THEN 1 ELSE 0 END) AS units", id=f"act:{AMEND}")[0]
    assert (notification["kind"], notification["amends"], notification["n"], notification["units"]) ==         ("rule", ECOM, 2, 0)  # stored for provenance, never embedded


def test_g12_principal_reingest_and_notification_reingest_change_nothing(kb) -> None:
    """P-21: a --replace re-ingest of the principal re-applies the substitution; R-16:
    re-ingesting the notification is idempotent."""
    before = _rule_4_1(kb)
    totals = "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS rels"
    counts = q(kb, totals)[0]
    asyncio.run(_ingest(kb, ECOM_PDF, ECOM, replace=True))
    assert _rule_4_1(kb) == before
    asyncio.run(_ingest(kb, AMEND_PDF, AMEND))
    assert _rule_4_1(kb) == before
    assert q(kb, totals)[0] == counts


def test_reingest_is_idempotent_and_keeps_rule_links(kb) -> None:
    """R-16: re-running a legal job changes nothing, including PRESCRIBES_FOR edges that
    point into the re-ingested Act from a rules document (A-12)."""
    count_all = "MATCH (n) RETURN count(n) AS nodes"
    count_rels = "MATCH ()-[r]->() RETURN count(r) AS rels"
    before = (q(kb, count_all)[0]["nodes"], q(kb, count_rels)[0]["rels"])
    asyncio.run(_ingest(kb, CPA_PDF, CPA, replace=True))
    after = (q(kb, count_all)[0]["nodes"], q(kb, count_rels)[0]["rels"])
    assert after == before
    edges = q(kb, "MATCH (:Section {act_id: $a})-[r:PRESCRIBES_FOR]->() RETURN count(r) AS n", a=RULES)
    assert edges[0]["n"] == 3
