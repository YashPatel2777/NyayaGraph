"""EmbeddingService: the only code that creates vectors (R-07, R-08).

- R-07: adds 'search_document: ' to stored text and 'search_query: ' to queries. No other
  code adds or removes these prefixes.
- R-08: every vector must be exactly EMBED_DIM finite floats, otherwise EmbeddingError.
  There is no fallback model. Batches hold at most 32 texts. Connection errors are retried
  3 times after 1, 2 and 4 s, then 503 DEPENDENCY_UNAVAILABLE (E-03).
- 2.2: the Meta node must carry model == EMBED_MODEL and dim == EMBED_DIM, otherwise
  EmbeddingMetaMismatch, which aborts startup.
- R-04: logs carry counts, lengths and timings only.

Ollama is called over async HTTP, so nothing blocks the event loop (1.2).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
)

from app.errors import AppError, ErrorCode
from app.schemas.graph_schema import DOC_PREFIX, META_EMBEDDING_KEY, QUERY_PREFIX
from app.services.retrieval import graph_queries
from app.services.retrieval.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 32  # R-08, 4.2
RETRY_WAITS_S = (1.0, 2.0, 4.0)  # R-08, E-03
EMBED_TIMEOUT_S = 60.0
# Fixed probe used by the startup check; not case data.
PROBE_TEXT = "startup check"


class EmbeddingError(ValueError):
    """A vector broke R-08 (wrong length, non-finite or non-numeric values, wrong count)."""


class EmbeddingMetaMismatch(RuntimeError):
    """The Meta node disagrees with EMBED_MODEL or EMBED_DIM (2.2). Startup must abort."""


class _OllamaServerError(Exception):
    pass


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (httpx.TransportError, _OllamaServerError))


class EmbeddingService:
    def __init__(
        self,
        ollama_url: str,
        model: str,
        dim: int,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.model = model
        self.dim = dim
        self._url = ollama_url.rstrip("/") + "/api/embed"
        self._client = client or httpx.AsyncClient(timeout=EMBED_TIMEOUT_S)
        self._sleep = sleep

    async def aclose(self) -> None:
        await self._client.aclose()

    # Public API

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Vectors for stored text (legal units, case chunks), prefixed 'search_document: '."""
        return await self._embed([DOC_PREFIX + t for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        """Vector for a search query, prefixed 'search_query: '."""
        return (await self._embed([QUERY_PREFIX + text]))[0]

    async def check_model_served(self) -> None:
        """Startup and /health: Ollama serves EMBED_MODEL and returns valid vectors (1.2)."""
        await self.embed_query(PROBE_TEXT)

    def check_meta(self, meta: Mapping[str, Any] | None) -> None:
        """2.2: the Meta node must have model == EMBED_MODEL and dim == EMBED_DIM."""
        if meta is None:
            raise EmbeddingMetaMismatch(
                "Meta node 'embedding' is missing; run python -m scripts.init_neo4j"
            )
        if meta.get("model") != self.model or meta.get("dim") != self.dim:
            raise EmbeddingMetaMismatch(
                f"Meta node has model={meta.get('model')!r} dim={meta.get('dim')!r}, "
                f"config has EMBED_MODEL={self.model!r} EMBED_DIM={self.dim}. Changing the "
                "embedding model requires a new database or a full re-embed."
            )

    async def verify_meta(self, neo4j: Neo4jService) -> None:
        rows = await neo4j.read(graph_queries.GET_META, {"key": META_EMBEDDING_KEY})
        self.check_meta(rows[0] if rows else None)

    # Internals

    def _validate(self, vector: Any) -> list[float]:
        # R-08: exactly dim finite floats.
        if not isinstance(vector, list) or len(vector) != self.dim:
            got = len(vector) if isinstance(vector, list) else type(vector).__name__
            raise EmbeddingError(f"expected {self.dim} floats, got {got}")
        out: list[float] = []
        for x in vector:
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise EmbeddingError("vector contains a non-finite or non-numeric value")
            out.append(float(x))
        return out

    async def _embed(self, prefixed: list[str]) -> list[list[float]]:
        if not prefixed:
            return []
        start = time.perf_counter()
        vectors: list[list[float]] = []
        for offset in range(0, len(prefixed), EMBED_BATCH_SIZE):
            batch = prefixed[offset : offset + EMBED_BATCH_SIZE]
            raw = await self._post_with_retry(batch)
            if len(raw) != len(batch):
                raise EmbeddingError(f"expected {len(batch)} vectors, got {len(raw)}")
            vectors.extend(self._validate(v) for v in raw)
        logger.info(
            "embed_ok model=%s texts=%d chars=%d ms=%d",
            self.model,
            len(prefixed),
            sum(len(t) for t in prefixed),
            (time.perf_counter() - start) * 1000,
        )
        return vectors

    async def _post(self, batch: list[str]) -> list[Any]:
        resp = await self._client.post(self._url, json={"model": self.model, "input": batch})
        if resp.status_code >= 500:
            raise _OllamaServerError(f"HTTP {resp.status_code}")
        if resp.status_code != 200:
            # 404: the model is not pulled (E-03). Retrying cannot fix it.
            logger.error("embed_http_error model=%s status=%d", self.model, resp.status_code)
            raise AppError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                details={"dependency": "ollama"},
            )
        embeddings = resp.json().get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingError("Ollama response has no embeddings list")
        return embeddings

    async def _post_with_retry(self, batch: list[str]) -> list[Any]:
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            wait=wait_chain(*(wait_fixed(s) for s in RETRY_WAITS_S)),
            stop=stop_after_attempt(len(RETRY_WAITS_S) + 1),
            sleep=self._sleep,
            reraise=True,
            before_sleep=lambda rs: logger.warning(
                "embed_retry attempt=%d error=%s",
                rs.attempt_number,
                type(rs.outcome.exception()).__name__ if rs.outcome else "unknown",
            ),
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await self._post(batch)
        except (httpx.TransportError, _OllamaServerError) as exc:
            raise AppError(
                ErrorCode.DEPENDENCY_UNAVAILABLE, details={"dependency": "ollama"}
            ) from exc
        raise AssertionError("unreachable")  # pragma: no cover
