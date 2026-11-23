"""Startup checks (1.2): a model mismatch, or any other failed check, aborts startup with
exit code 2 (Phase 1 exit)."""

from __future__ import annotations

import pytest

import app.main as main
from app.errors import AppError, ErrorCode
from app.schemas.graph_schema import REQUIRED_SEARCH_INDEXES
from app.services.retrieval import graph_queries
from app.services.retrieval.embeddings import EmbeddingService

GOOD_META = {"model": "nomic-embed-text", "dim": 768}


class FakeNeo4j:
    def __init__(self, meta=GOOD_META, offline: tuple[str, ...] = (), reachable: bool = True,
                 properties: dict[str, list[str]] | None = None):
        self.meta = meta
        self.properties = properties or {}
        self.offline = offline
        self.reachable = reachable
        self.closed = False

    async def verify_connectivity(self) -> None:
        if not self.reachable:
            raise AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, details={"dependency": "neo4j"})

    async def read(self, query, params=None):
        if query == graph_queries.INDEX_STATES:
            return [
                {"name": n, "state": "POPULATING" if n in self.offline else "ONLINE",
                 "properties": self.properties.get(n, ["embedding", "case_id", "user_id"])}
                for n in params["names"]
            ]
        if query == graph_queries.GET_META:
            return [self.meta] if self.meta is not None else []
        raise AssertionError("unexpected query")

    def close(self) -> None:
        self.closed = True


class FakeEmbeddings(EmbeddingService):
    def __init__(self, model="nomic-embed-text", dim=768, served=True):
        self.model, self.dim, self.served = model, dim, served
        self.closed = False

    async def check_model_served(self) -> None:
        if not self.served:
            raise AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, details={"dependency": "ollama"})

    async def aclose(self) -> None:
        self.closed = True


async def test_all_checks_pass() -> None:
    await main.run_startup_checks(FakeNeo4j(), FakeEmbeddings())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "neo4j, embeddings, match",
    [
        (FakeNeo4j(reachable=False), FakeEmbeddings(), "Neo4j is unreachable"),
        (FakeNeo4j(), FakeEmbeddings(served=False), "Ollama does not serve"),
        (FakeNeo4j(offline=("legal_text_ft",)), FakeEmbeddings(), "not ONLINE"),
        (FakeNeo4j(meta=None), FakeEmbeddings(), "Meta node 'embedding' is missing"),
        (FakeNeo4j(meta={"model": "other", "dim": 768}), FakeEmbeddings(), "model='other'"),
        (FakeNeo4j(meta={"model": "nomic-embed-text", "dim": 384}), FakeEmbeddings(), "dim=384"),
        (FakeNeo4j(), FakeEmbeddings(model="mxbai-embed-large"), "mxbai-embed-large"),
    ],
    ids=["neo4j", "ollama", "index", "meta-missing", "meta-model", "meta-dim", "config-model"],
)
async def test_failed_check_raises(neo4j, embeddings, match) -> None:
    with pytest.raises(main.StartupCheckError, match=match):
        await main.run_startup_checks(neo4j, embeddings)


async def test_every_required_index_is_checked() -> None:
    for name in REQUIRED_SEARCH_INDEXES:
        with pytest.raises(main.StartupCheckError, match=name):
            await main.run_startup_checks(FakeNeo4j(offline=(name,)), FakeEmbeddings())  # type: ignore[arg-type]


async def test_case_vector_index_without_filter_properties_fails() -> None:
    # SC-15 (2.2): a pre-SC-15 index is ONLINE but cannot serve the in-index scope filter.
    neo = FakeNeo4j(properties={"case_chunk_embedding": ["embedding"]})
    with pytest.raises(main.StartupCheckError, match=r"case_chunk_embedding lacks filter properties \['case_id', 'user_id'\]"):
        await main.run_startup_checks(neo, FakeEmbeddings())  # type: ignore[arg-type]


async def test_model_mismatch_aborts_lifespan_with_exit_code_2(monkeypatch) -> None:
    neo = FakeNeo4j(meta={"model": "some-other-model", "dim": 768})
    emb = FakeEmbeddings()
    exits: list[int] = []

    def fake_abort(code: int):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(main, "Neo4jService", lambda *a, **k: neo)
    monkeypatch.setattr(main, "EmbeddingService", lambda *a, **k: emb)
    monkeypatch.setattr(main, "configure_observability", lambda settings: None)
    monkeypatch.setattr(main, "_abort", fake_abort)

    application = main.create_app()
    with pytest.raises(SystemExit):
        async with main.lifespan(application):
            pytest.fail("the app must not start")
    assert exits == [2]
    assert neo.closed and emb.closed


class FakeModel:
    def __init__(self, *args, **kwargs) -> None:
        self.loaded = 0

    def load(self) -> None:
        self.loaded += 1


class FakeJobs:
    def __init__(self, *args) -> None:
        self.events: list[str] = []

    async def start(self) -> None:
        self.events.append("start")

    async def stop(self) -> None:
        self.events.append("stop")


async def test_successful_lifespan_exposes_services(monkeypatch) -> None:
    neo, emb = FakeNeo4j(), FakeEmbeddings()
    monkeypatch.setattr(main, "Neo4jService", lambda *a, **k: neo)
    monkeypatch.setattr(main, "EmbeddingService", lambda *a, **k: emb)
    monkeypatch.setattr(main, "configure_observability", lambda settings: None)
    monkeypatch.setattr(main, "_abort", lambda code: pytest.fail("must not abort"))
    models: list[FakeModel] = []

    def model(*args, **kwargs):
        models.append(FakeModel())
        return models[-1]

    monkeypatch.setattr(main, "OcrService", model)
    monkeypatch.setattr(main, "TranscriptionService", model)
    monkeypatch.setattr(main, "JobQueue", FakeJobs)
    rankers: list[str] = []
    monkeypatch.setattr(main, "RankingService", lambda name: rankers.append(name) or object())

    class FakeConcepts:
        @classmethod
        async def load(cls, neo4j):
            return cls()

    monkeypatch.setattr(main, "ConceptIndex", FakeConcepts)

    application = main.create_app()
    async with main.lifespan(application):
        assert application.state.neo4j is neo
        assert application.state.embeddings is emb
        assert application.state.llm is not None
        # 1.2: FlashRank loads once; the query graph is compiled with its services.
        assert len(rankers) == 1
        agent = application.state.query_agent
        assert agent.deps.neo4j is neo and isinstance(agent.deps.concepts, FakeConcepts)
        # 1.2: EasyOCR and Whisper load once at startup; E-11 recovery runs in start().
        assert [m.loaded for m in models] == [1, 1]
        assert application.state.jobs.events == ["start"]
    assert application.state.jobs.events == ["start", "stop"]
    assert neo.closed and emb.closed
