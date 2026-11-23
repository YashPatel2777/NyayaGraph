"""Case scoping (R-01, R-02).

Every case-scoped read and write goes through ScopedGraph, which supplies case_id and
user_id as query parameters. Callers cannot set or override either value, and a query that
does not reference both $case_id and $user_id is refused, so a missing scope filter fails
loudly instead of leaking another case's data.

user_id always comes from deps.get_current_user() (or, in the job worker, from the Job node
that the API created with it); a request can never set it (R-02, 3.2).

The case_id property on case-scoped nodes holds the bare UUID v4 (A-8 in docs/README.md).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.services.retrieval.neo4j_service import WRITE_BATCH_SIZE, Neo4jService

SCOPE_KEYS = frozenset({"case_id", "user_id"})
_PLACEHOLDERS = (re.compile(r"\$case_id\b"), re.compile(r"\$user_id\b"))


@dataclass(frozen=True)
class CaseScope:
    case_id: str
    user_id: str

    def __post_init__(self) -> None:
        parsed = uuid.UUID(str(self.case_id))
        if parsed.version != 4:
            raise ValueError("case_id must be a UUID v4")
        object.__setattr__(self, "case_id", str(parsed))
        if not isinstance(self.user_id, str) or not self.user_id:
            raise ValueError("user_id must be a non-empty string")


def scoped_params(scope: CaseScope, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return params with the scope's case_id and user_id added. Overriding either is an
    error, never a silent replacement (R-01, R-02)."""
    given = dict(params or {})
    clash = SCOPE_KEYS & given.keys()
    if clash:
        raise ValueError(f"scope parameters may not be supplied by the caller: {sorted(clash)}")
    return {**given, "case_id": scope.case_id, "user_id": scope.user_id}


def require_scope_filter(query: str) -> None:
    missing = [p.pattern for p in _PLACEHOLDERS if not p.search(query)]
    if missing:
        raise ValueError("case-scoped query must reference both $case_id and $user_id")


class ScopedGraph:
    """Case-scoped facade over Neo4jService. Queries come from graph_queries.py."""

    def __init__(self, neo4j: Neo4jService, scope: CaseScope) -> None:
        self._neo4j = neo4j
        self.scope = scope

    def _prepare(self, query: str, params: Mapping[str, Any] | None) -> dict[str, Any]:
        require_scope_filter(query)
        return scoped_params(self.scope, params)

    async def read(self, query: str, params: Mapping[str, Any] | None = None) -> list[dict]:
        return await self._neo4j.read(query, self._prepare(query, params))

    async def write(self, query: str, params: Mapping[str, Any] | None = None) -> list[dict]:
        return await self._neo4j.write(query, self._prepare(query, params))

    async def write_batches(
        self,
        query: str,
        rows: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any] | None = None,
        batch_size: int = WRITE_BATCH_SIZE,
    ) -> int:
        return await self._neo4j.write_batches(
            query, rows, self._prepare(query, params), batch_size
        )

    # Sync variants for code already running in a worker thread.

    def read_sync(self, query: str, params: Mapping[str, Any] | None = None) -> list[dict]:
        return self._neo4j.read_sync(query, self._prepare(query, params))

    def write_sync(self, query: str, params: Mapping[str, Any] | None = None) -> list[dict]:
        return self._neo4j.write_sync(query, self._prepare(query, params))

    def write_batches_sync(
        self,
        query: str,
        rows: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any] | None = None,
        batch_size: int = WRITE_BATCH_SIZE,
    ) -> int:
        return self._neo4j.write_batches_sync(
            query, rows, self._prepare(query, params), batch_size
        )
