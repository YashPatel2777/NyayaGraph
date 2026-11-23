"""FastAPI app and lifespan (1.2).

Startup aborts with exit code 2 if Neo4j is unreachable, Ollama does not serve EMBED_MODEL,
any required index is not ONLINE, or the Meta node check in 2.2 fails. The EasyOCR reader
and the Whisper model are loaded here once, then the job queue recovers interrupted jobs
(E-11) and starts its worker. The FlashRank model and the R-13 concept list also load once,
and the query graph is compiled with its MemorySaver (1.4 C).

Run: python -m app.main (binds APP_HOST), or uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import NoReturn

import anyio
import logfire
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.graph import QueryAgent, build_graph
from app.agents.nodes.graph_expander import ConceptIndex
from app.agents.state import QueryDeps
from app.api import admin_legal, cases, documents, health, jobs as jobs_api, legal, query as query_api
from app.config import Settings, get_settings
from app.errors import AppError, install_exception_handlers
from app.ingestion.case.case_processor import CaseProcessor
from app.ingestion.jobs import JobQueue
from app.ingestion.legal.legal_processor import LegalProcessor
from app.middleware import RateLimitMiddleware, RequestIDMiddleware
from app.observability import configure_observability
from app.schemas.graph_schema import REQUIRED_SEARCH_INDEXES, index_missing_properties
from app.services.llm.groq_client import GroqClient
from app.services.media.ocr_service import OcrService
from app.services.media.transcription_service import TranscriptionService
from app.services.retrieval import graph_queries
from app.services.retrieval.embeddings import (
    EmbeddingError,
    EmbeddingMetaMismatch,
    EmbeddingService,
)
from app.services.retrieval.neo4j_service import Neo4jService
from app.services.retrieval.ranking_service import RankingService

logger = logging.getLogger(__name__)

STARTUP_EXIT_CODE = 2  # 1.2


class StartupCheckError(RuntimeError):
    pass


async def run_startup_checks(neo4j: Neo4jService, embeddings: EmbeddingService) -> None:
    """The four startup checks of 1.2, in order. Raises StartupCheckError on the first failure."""
    try:
        await neo4j.verify_connectivity()
    except Exception as exc:  # AppError after E-04 retries, or an auth/config error
        raise StartupCheckError(f"Neo4j is unreachable ({type(exc).__name__})") from exc

    try:
        await embeddings.check_model_served()
    except (AppError, EmbeddingError) as exc:
        raise StartupCheckError(
            f"Ollama does not serve {embeddings.model} with {embeddings.dim}-float vectors "
            f"({type(exc).__name__})"
        ) from exc

    rows = await neo4j.read(graph_queries.INDEX_STATES, {"names": list(REQUIRED_SEARCH_INDEXES)})
    states = {row["name"]: row["state"] for row in rows}
    not_online = [name for name in REQUIRED_SEARCH_INDEXES if states.get(name) != "ONLINE"]
    if not_online:
        raise StartupCheckError(
            f"indexes not ONLINE: {not_online}; run python -m scripts.init_neo4j"
        )
    # SC-15 (2.2): the case vector index must declare the scope filter properties.
    for row in rows:
        missing = index_missing_properties(row["name"], row.get("properties"))
        if missing:
            raise StartupCheckError(
                f"index {row['name']} lacks filter properties {missing}; "
                "run python -m scripts.init_neo4j"
            )

    try:
        await embeddings.verify_meta(neo4j)
    except EmbeddingMetaMismatch as exc:
        raise StartupCheckError(str(exc)) from exc


async def dependency_status(neo4j: Neo4jService, embeddings: EmbeddingService) -> dict[str, str]:
    """GET /health (3.4): the 1.2 checks run independently, each 'ok' or 'fail'. Exceptions
    are caught, so /health answers even when a dependency is down. Logs carry only the
    exception type (R-04)."""
    checks = dict.fromkeys(("neo4j", "indexes", "embedding_meta", "ollama"), "fail")
    try:
        await neo4j.verify_connectivity()
        checks["neo4j"] = "ok"
    except Exception as exc:
        logger.warning("health_check_failed check=neo4j error=%s", type(exc).__name__)
    if checks["neo4j"] == "ok":
        try:
            rows = await neo4j.read(graph_queries.INDEX_STATES, {"names": list(REQUIRED_SEARCH_INDEXES)})
            states = {row["name"]: row["state"] for row in rows}
            online = all(states.get(name) == "ONLINE" for name in REQUIRED_SEARCH_INDEXES)
            complete = not any(index_missing_properties(r["name"], r.get("properties")) for r in rows)
            checks["indexes"] = "ok" if online and complete else "fail"  # SC-15
        except Exception as exc:
            logger.warning("health_check_failed check=indexes error=%s", type(exc).__name__)
        try:
            await embeddings.verify_meta(neo4j)
            checks["embedding_meta"] = "ok"
        except Exception as exc:
            logger.warning("health_check_failed check=embedding_meta error=%s", type(exc).__name__)
    try:
        await embeddings.check_model_served()
        checks["ollama"] = "ok"
    except Exception as exc:
        logger.warning("health_check_failed check=ollama error=%s", type(exc).__name__)
    return checks


def _abort(code: int) -> NoReturn:
    # Uvicorn turns any exception raised during lifespan startup into its own exit code (3),
    # so the process exits directly to honour exit code 2 (1.2).
    for handler in logging.getLogger().handlers:
        handler.flush()
    logfire.force_flush()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_observability(settings)

    neo4j = Neo4jService(
        settings.NEO4J_URI,
        settings.NEO4J_USER,
        settings.NEO4J_PASSWORD.get_secret_value(),
        settings.NEO4J_DATABASE,
    )
    embeddings = EmbeddingService(settings.OLLAMA_URL, settings.EMBED_MODEL, settings.EMBED_DIM)
    try:
        await run_startup_checks(neo4j, embeddings)
    except StartupCheckError as exc:
        logger.critical("startup_aborted reason=%s", exc)
        print(f"Startup aborted: {exc}", file=sys.stderr)
        neo4j.close()
        await embeddings.aclose()
        _abort(STARTUP_EXIT_CODE)

    app.state.neo4j = neo4j
    app.state.embeddings = embeddings
    app.state.check_dependencies = partial(dependency_status, neo4j, embeddings)  # GET /health
    llm = GroqClient(settings.GROQ_API_KEY.get_secret_value(), settings.GROQ_MODEL)
    app.state.llm = llm

    # 1.2: media models load once, outside the event loop.
    start = time.perf_counter()
    ocr = OcrService(settings.OCR_LANGS)
    transcriber = TranscriptionService(settings.WHISPER_MODEL)
    await anyio.to_thread.run_sync(ocr.load)
    await anyio.to_thread.run_sync(transcriber.load)
    logger.info("media_models_loaded ms=%d", (time.perf_counter() - start) * 1000)

    # 1.2: the reranker loads once, off the event loop; R-13: the concept list is cached.
    ranker = await anyio.to_thread.run_sync(RankingService, settings.FLASHRANK_MODEL)
    concepts = await ConceptIndex.load(neo4j)
    query_deps = QueryDeps(neo4j=neo4j, embeddings=embeddings, llm=llm, ranker=ranker,
                           concepts=concepts, rerank_high=settings.RERANK_HIGH)
    app.state.query_agent = QueryAgent(build_graph(query_deps), query_deps, settings.ENABLE_GUARDRAILS)

    gpu = asyncio.Semaphore(1)  # 1.2: at most one job in a GPU stage
    case_processor = CaseProcessor(
        neo4j, embeddings, ocr, transcriber, gpu, llm, settings.LINK_MIN_SCORE
    )
    jobs = JobQueue(
        neo4j, case_processor, LegalProcessor(neo4j, embeddings, ocr=ocr), Path(settings.UPLOAD_DIR)
    )
    await jobs.start()  # E-11 recovery, then the worker
    app.state.jobs = jobs
    logger.info("startup_ok env=%s", settings.APP_ENV)
    try:
        yield
    finally:
        await jobs.stop()
        neo4j.close()
        await embeddings.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    prod = settings.APP_ENV == "prod"
    app = FastAPI(
        title="Legal GraphRAG Platform",
        version="1.0.0",
        lifespan=lifespan,
        # 1.3: in prod, docs endpoints are disabled.
        docs_url=None if prod else "/docs",
        redoc_url=None if prod else "/redoc",
        openapi_url=None if prod else "/openapi.json",
    )
    app.state.settings = settings

    if settings.CORS_ORIGINS:  # empty means CORS is off (1.3)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "Retry-After"],
        )
    # 3.2: inside RequestIDMiddleware, so a 429 carries the request id (3.1, 3.3).
    app.add_middleware(RateLimitMiddleware)
    # Added last so it is the outermost user middleware and sees every response (3.1).
    app.add_middleware(RequestIDMiddleware)
    install_exception_handlers(app)
    app.include_router(cases.router)
    app.include_router(documents.router)
    app.include_router(jobs_api.router)
    app.include_router(admin_legal.router)
    app.include_router(query_api.router)
    app.include_router(health.router)
    app.include_router(legal.router)
    return app


app = create_app()

APP_PORT = 8000  # 1.1 component table


def serve(settings: Settings | None = None) -> None:
    """Run Uvicorn on APP_HOST, default 127.0.0.1 (1.3, 3.2; Phase 5 exit). Keep it loopback
    while the service has no authentication."""
    import uvicorn

    settings = settings or get_settings()
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=APP_PORT)


if __name__ == "__main__":
    serve()
