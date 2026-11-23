"""Request dependencies.

get_current_user() is the ONLY place identity is decided (3.2). v1 has no authentication:
every request runs as DEFAULT_USER_ID and no Authorization header is read. Adding login
later changes this one function and nothing else.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import Request

from app.agents.graph import QueryAgent
from app.agents.nodes.graph_expander import ConceptIndex
from app.config import get_settings
from app.ingestion.jobs import JobQueue
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.neo4j_service import Neo4jService


def get_current_user() -> str:
    # R-02: a request can never set user_id.
    return get_settings().DEFAULT_USER_ID


# Shared services, created once in the lifespan handler (app/main.py, 1.2).


def get_neo4j(request: Request) -> Neo4jService:
    return request.app.state.neo4j


def get_embeddings(request: Request) -> EmbeddingService:
    return request.app.state.embeddings


def get_jobs(request: Request) -> JobQueue:
    return request.app.state.jobs


def get_query_agent(request: Request) -> QueryAgent:
    return request.app.state.query_agent


def get_dependency_check(request: Request) -> Callable[[], Awaitable[dict[str, str]]]:
    """GET /health: the 1.2 checks bound to this app's services (app/main.py)."""
    return request.app.state.check_dependencies


def get_concepts(request: Request) -> ConceptIndex:
    """The R-13 concept list cached at startup (GET /v1/legal/provisions definitions, A-45)."""
    return request.app.state.query_agent.deps.concepts
