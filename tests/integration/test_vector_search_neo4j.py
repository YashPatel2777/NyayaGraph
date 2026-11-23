"""Case vector search against a real Neo4j (SC-15, Phase 4 isolation test at service level).

Two users hold identical chunks: queries scoped to user B never return user A's chunks.
A small case whose chunks are all outranked by 1,000 other-user chunks (more than the R-09
cap of 800) still gets its exact top k: the case the R-09 loop missed in the O-1 probe.
"""

from __future__ import annotations

import math
import random
import uuid

import pytest

from app.services.retrieval.neo4j_service import Neo4jService
from app.services.retrieval.scope import CaseScope, ScopedGraph
from app.services.retrieval.vector_search import case_vector_search
from scripts.init_neo4j import init_schema

pytestmark = pytest.mark.neo4j

CASE_A = str(uuid.uuid4())  # user local
CASE_B = str(uuid.uuid4())  # user other, same chunks and vectors as case A
CASE_SMALL = str(uuid.uuid4())  # user local, 20 chunks
BASES = 20
COPIES = 1000  # other-user near-copies of the query, above the R-09 fetch_k cap of 800


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


@pytest.fixture(scope="module")
def db(neo4j_test_db: Neo4jService) -> tuple[Neo4jService, list[list[float]]]:
    init_schema(neo4j_test_db)
    rnd = random.Random(11)
    bases = [_unit([rnd.gauss(0, 1) for _ in range(768)]) for _ in range(BASES)]
    rows = []
    for i, vec in enumerate(bases[:5]):
        rows.append({"id": f"chk:a:{i}", "case_id": CASE_A, "user_id": "local", "emb": vec})
        rows.append({"id": f"chk:b:{i}", "case_id": CASE_B, "user_id": "other", "emb": vec})
    query = bases[0]
    for j in range(BASES):
        # Small-case chunk j has cosine exactly 0.95 - 0.04 j to the query: a direction
        # orthogonal to the query is mixed in, so the exact order is unambiguous.
        other = [rnd.gauss(0, 1) for _ in range(768)]
        dot = sum(a * b for a, b in zip(other, query))
        ortho = _unit([a - dot * b for a, b in zip(other, query)])
        sim = 0.95 - 0.04 * j
        rows.append({"id": f"chk:s:{j:02d}", "case_id": CASE_SMALL, "user_id": "local",
                     "emb": _unit([sim * a + math.sqrt(1 - sim * sim) * b for a, b in zip(query, ortho)])})
    for c in range(COPIES):
        # Other users' near-copies (cosine about 0.99 to the query) outrank every small-case
        # chunk, so an over-fetch of 800 holds none of them.
        rows.append({"id": f"chk:n:{c}", "case_id": str(uuid.uuid4()), "user_id": "other",
                     "emb": _unit([x + rnd.gauss(0, 0.005) for x in query])})
    for start in range(0, len(rows), 1000):
        neo4j_test_db.write_sync(
            "UNWIND $rows AS r CREATE (c:CaseChunk {id: r.id, case_id: r.case_id, user_id: r.user_id, "
            "doc_id: 'd', text: r.id}) WITH c, r CALL db.create.setNodeVectorProperty(c, 'embedding', r.emb)",
            {"rows": rows[start:start + 1000]},
        )
    neo4j_test_db.write_sync("CALL db.awaitIndexes(600)")
    return neo4j_test_db, bases


def _graph(db: Neo4jService, case_id: str, user_id: str) -> ScopedGraph:
    return ScopedGraph(db, CaseScope(case_id, user_id))


async def test_user_b_never_sees_user_a_chunks(db) -> None:
    neo, bases = db
    for vec in bases[:5]:
        rows = await case_vector_search(_graph(neo, CASE_B, "other"), vec, k=15)
        assert {r["id"] for r in rows} == {f"chk:b:{i}" for i in range(5)}
        assert rows[0]["score"] == pytest.approx(1.0, abs=1e-4)


async def test_right_case_wrong_user_returns_nothing(db) -> None:
    neo, bases = db
    assert await case_vector_search(_graph(neo, CASE_A, "other"), bases[0], k=5) == []


@pytest.mark.parametrize("k", [5, 15])
async def test_small_case_gets_its_exact_top_k(db, k: int) -> None:
    neo, bases = db
    rows = await case_vector_search(_graph(neo, CASE_SMALL, "local"), bases[0], k=k)
    assert [r["id"] for r in rows] == [f"chk:s:{j:02d}" for j in range(k)]
    for j, row in enumerate(rows):
        assert row["score"] == pytest.approx((1 + 0.95 - 0.04 * j) / 2, abs=1e-3)  # Neo4j cosine score
