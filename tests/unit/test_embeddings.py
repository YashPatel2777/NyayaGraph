"""EmbeddingService: the two prefixes, the 768 check, batching, retry, and the Meta check
that aborts startup (Phase 1 exit; R-07, R-08, 2.2, E-03, 1.2)."""

from __future__ import annotations

import json
import math

import httpx
import pytest

from app.errors import AppError, ErrorCode
from app.schemas.graph_schema import DOC_PREFIX, QUERY_PREFIX
from app.services.retrieval.embeddings import (
    EMBED_BATCH_SIZE,
    EmbeddingError,
    EmbeddingMetaMismatch,
    EmbeddingService,
)

DIM = 768


def _vec(seed: float = 0.1, dim: int = DIM) -> list[float]:
    return [seed] * dim


class FakeOllama:
    """httpx transport that records requests and answers like Ollama /api/embed."""

    def __init__(self, make_vector=lambda text: _vec(), status: int = 200, fail_times: int = 0):
        self.requests: list[dict] = []
        self.make_vector = make_vector
        self.status = status
        self.fail_times = fail_times

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise httpx.ConnectError("connection refused", request=request)
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "model not found"})
        payload = {"model": body["model"], "embeddings": [self.make_vector(t) for t in body["input"]]}
        # json.dumps emits NaN/Infinity tokens, as a misbehaving server could.
        return httpx.Response(
            200, content=json.dumps(payload), headers={"content-type": "application/json"}
        )


class Sleeps:
    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def _service(fake: FakeOllama, sleep: Sleeps | None = None) -> EmbeddingService:
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake))
    return EmbeddingService(
        "http://ollama.test", "nomic-embed-text", DIM, client=client, sleep=sleep or Sleeps()
    )


# R-07: prefixes


async def test_documents_get_search_document_prefix() -> None:
    fake = FakeOllama()
    vectors = await _service(fake).embed_documents(["alpha", "beta"])
    assert fake.requests[0]["input"] == ["search_document: alpha", "search_document: beta"]
    assert fake.requests[0]["model"] == "nomic-embed-text"
    assert len(vectors) == 2 and all(len(v) == DIM for v in vectors)


async def test_query_gets_search_query_prefix() -> None:
    fake = FakeOllama()
    vector = await _service(fake).embed_query("limitation period")
    assert fake.requests[0]["input"] == ["search_query: limitation period"]
    assert len(vector) == DIM


def test_prefix_constants_match_spec() -> None:
    assert DOC_PREFIX == "search_document: "
    assert QUERY_PREFIX == "search_query: "


async def test_prefix_is_added_exactly_once() -> None:
    fake = FakeOllama()
    await _service(fake).embed_documents(["search_document: already"])
    # The service does not strip or deduplicate; callers must pass raw text (R-07).
    assert fake.requests[0]["input"] == ["search_document: search_document: already"]


# R-08: 768 finite floats


@pytest.mark.parametrize(
    "bad",
    [
        _vec(dim=767),
        _vec(dim=769),
        [],
        [math.nan] + _vec(dim=767),
        [math.inf] + _vec(dim=767),
        [-math.inf] + _vec(dim=767),
        ["0.1"] + _vec(dim=767),
        [None] + _vec(dim=767),
        [True] + _vec(dim=767),
    ],
    ids=["767", "769", "empty", "nan", "inf", "-inf", "string", "none", "bool"],
)
async def test_invalid_vector_raises(bad) -> None:
    fake = FakeOllama(make_vector=lambda text: bad)
    with pytest.raises(EmbeddingError):
        await _service(fake).embed_query("q")


async def test_nan_is_rejected_after_json_roundtrip() -> None:
    # Ollama would send NaN as a JSON token; make sure it is caught either way.
    def handler(request: httpx.Request) -> httpx.Response:
        payload = '{"embeddings": [[NaN' + ", 0.1" * (DIM - 1) + "]]}"
        return httpx.Response(200, content=payload, headers={"content-type": "application/json"})

    svc = EmbeddingService(
        "http://ollama.test", "nomic-embed-text", DIM,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), sleep=Sleeps(),
    )
    with pytest.raises(EmbeddingError):
        await svc.embed_query("q")


async def test_wrong_vector_count_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [_vec()]})

    svc = EmbeddingService(
        "http://ollama.test", "nomic-embed-text", DIM,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), sleep=Sleeps(),
    )
    with pytest.raises(EmbeddingError, match="expected 2 vectors"):
        await svc.embed_documents(["a", "b"])


async def test_vectors_are_plain_floats() -> None:
    fake = FakeOllama(make_vector=lambda text: [1] * DIM)
    vector = await _service(fake).embed_query("q")
    assert all(type(x) is float for x in vector)


# Batching (R-08, 4.2)


async def test_batches_hold_at_most_32_texts_and_keep_order() -> None:
    fake = FakeOllama(make_vector=lambda text: _vec(float(text.rsplit(" ", 1)[1])))
    texts = [f"t {i}" for i in range(70)]
    vectors = await _service(fake).embed_documents(texts)
    assert [len(r["input"]) for r in fake.requests] == [32, 32, 6]
    assert max(len(r["input"]) for r in fake.requests) <= EMBED_BATCH_SIZE
    assert [v[0] for v in vectors] == [float(i) for i in range(70)]


async def test_empty_input_makes_no_call() -> None:
    fake = FakeOllama()
    assert await _service(fake).embed_documents([]) == []
    assert fake.requests == []


# Retry and failure (R-08, E-03)


async def test_connection_errors_retried_three_times_with_1_2_4_seconds() -> None:
    fake = FakeOllama(fail_times=3)
    sleeps = Sleeps()
    vector = await _service(fake, sleeps).embed_query("q")
    assert len(vector) == DIM
    assert sleeps.waits == [1.0, 2.0, 4.0]


async def test_fourth_connection_failure_is_dependency_unavailable() -> None:
    fake = FakeOllama(fail_times=4)
    sleeps = Sleeps()
    with pytest.raises(AppError) as info:
        await _service(fake, sleeps).embed_query("q")
    assert info.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert info.value.http_status == 503
    assert info.value.details == {"dependency": "ollama"}
    assert sleeps.waits == [1.0, 2.0, 4.0]


async def test_model_not_pulled_is_dependency_unavailable_without_retry() -> None:
    fake = FakeOllama(status=404)
    sleeps = Sleeps()
    with pytest.raises(AppError) as info:
        await _service(fake, sleeps).check_model_served()
    assert info.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert sleeps.waits == []


# Meta check (2.2)


def test_meta_match_passes() -> None:
    _service(FakeOllama()).check_meta({"model": "nomic-embed-text", "dim": 768})


@pytest.mark.parametrize(
    "meta",
    [
        None,
        {"model": "mxbai-embed-large", "dim": 768},
        {"model": "nomic-embed-text", "dim": 1024},
        {"model": "nomic-embed-text"},
        {},
    ],
    ids=["missing", "model", "dim", "no-dim", "empty"],
)
def test_meta_mismatch_raises(meta) -> None:
    with pytest.raises(EmbeddingMetaMismatch):
        _service(FakeOllama()).check_meta(meta)


def test_config_model_mismatch_raises_against_stored_meta() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(FakeOllama()))
    svc = EmbeddingService("http://ollama.test", "other-model", DIM, client=client)
    with pytest.raises(EmbeddingMetaMismatch, match="other-model"):
        svc.check_meta({"model": "nomic-embed-text", "dim": 768})


async def test_verify_meta_reads_meta_node_by_parameter() -> None:
    from app.services.retrieval import graph_queries

    class FakeNeo4j:
        def __init__(self) -> None:
            self.calls: list = []

        async def read(self, query, params=None):
            self.calls.append((query, params))
            return [{"model": "nomic-embed-text", "dim": 768}]

    neo = FakeNeo4j()
    await _service(FakeOllama()).verify_meta(neo)  # type: ignore[arg-type]
    assert neo.calls == [(graph_queries.GET_META, {"key": "embedding"})]
