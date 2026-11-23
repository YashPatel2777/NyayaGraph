"""Neo4j driver, sessions, batched UNWIND writes and retry (E-04).

- Queries come from graph_queries.py; every value is passed as a parameter (R-01).
- Transient failures (ServiceUnavailable, SessionExpired, TransientError) are retried 3 times
  after 0.5, 1 and 2 s, then raise 503 DEPENDENCY_UNAVAILABLE (E-04). Writes are MERGE-based
  and idempotent, so a retry is safe (R-16).
- Transaction timeout 15 s (4.2).
- The driver is blocking: async callers use the async methods, which run the sync ones
  through anyio.to_thread.run_sync (1.2).
- R-04: logs carry query names, row counts and timings only, never parameters.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from typing import Any, TypeVar

import anyio
from neo4j import (
    Driver,
    GraphDatabase,
    ManagedTransaction,
    NotificationDisabledClassification,
    unit_of_work,
)
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_chain, wait_fixed

from app.errors import AppError, ErrorCode

logger = logging.getLogger(__name__)

T = TypeVar("T")

TX_TIMEOUT_S = 15.0  # 4.2
WRITE_BATCH_SIZE = 200  # 1.4 A7: UNWIND batches of 200
RETRY_WAITS_S = (0.5, 1.0, 2.0)  # E-04
TRANSIENT_ERRORS = (ServiceUnavailable, SessionExpired, TransientError)


def batched(rows: Sequence[Mapping[str, Any]], size: int) -> Iterator[list[Mapping[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


class Neo4jService:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
        *,
        driver: Driver | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.database = database
        # max_transaction_retry_time=0: the driver's own retry loop is off so E-04 governs.
        # UNRECOGNIZED notifications ("label/property does not exist") fire on every query
        # against an empty database, and DEPRECATION fires on every vector search
        # (db.index.vector.queryNodes, recorded in docs/README.md); the driver logs each with
        # the full query text.
        self._driver = driver or GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_transaction_retry_time=0,
            notifications_disabled_classifications=[
                NotificationDisabledClassification.UNRECOGNIZED,
                NotificationDisabledClassification.DEPRECATION,
            ],
        )
        self._sleep = sleep

    def close(self) -> None:
        self._driver.close()

    def _retrying(self) -> Retrying:
        kwargs: dict[str, Any] = {}
        if self._sleep is not None:
            kwargs["sleep"] = self._sleep
        return Retrying(
            retry=retry_if_exception_type(TRANSIENT_ERRORS),
            wait=wait_chain(*(wait_fixed(s) for s in RETRY_WAITS_S)),
            stop=stop_after_attempt(len(RETRY_WAITS_S) + 1),
            reraise=True,
            before_sleep=lambda rs: logger.warning(
                "neo4j_retry attempt=%d error=%s",
                rs.attempt_number,
                type(rs.outcome.exception()).__name__ if rs.outcome else "unknown",
            ),
            **kwargs,
        )

    def _with_retry(self, fn: Callable[[], T]) -> T:
        try:
            for attempt in self._retrying():
                with attempt:
                    return fn()
        except TRANSIENT_ERRORS as exc:
            raise AppError(
                ErrorCode.DEPENDENCY_UNAVAILABLE, details={"dependency": "neo4j"}
            ) from exc
        raise AssertionError("unreachable")  # pragma: no cover

    # Sync API (scripts, worker threads)

    def verify_connectivity_sync(self) -> None:
        self._with_retry(self._driver.verify_connectivity)

    def read_sync(self, query: str, params: Mapping[str, Any] | None = None) -> list[dict]:
        @unit_of_work(timeout=TX_TIMEOUT_S)
        def work(tx: ManagedTransaction) -> list[dict]:
            return [record.data() for record in tx.run(query, dict(params or {}))]

        def call() -> list[dict]:
            with self._driver.session(database=self.database) as session:
                return session.execute_read(work)

        return self._with_retry(call)

    def write_sync(self, query: str, params: Mapping[str, Any] | None = None) -> list[dict]:
        @unit_of_work(timeout=TX_TIMEOUT_S)
        def work(tx: ManagedTransaction) -> list[dict]:
            return [record.data() for record in tx.run(query, dict(params or {}))]

        def call() -> list[dict]:
            with self._driver.session(database=self.database) as session:
                return session.execute_write(work)

        return self._with_retry(call)

    def write_batches_sync(
        self,
        query: str,
        rows: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any] | None = None,
        batch_size: int = WRITE_BATCH_SIZE,
    ) -> int:
        """Run query once per batch of rows, all inside one transaction (1.4 A7).

        The query receives the batch as $rows and must UNWIND it. Returns the row count.
        """
        base = dict(params or {})
        if "rows" in base:
            raise ValueError("'rows' is reserved for the batch parameter")

        @unit_of_work(timeout=TX_TIMEOUT_S)
        def work(tx: ManagedTransaction) -> int:
            for chunk in batched(rows, batch_size):
                tx.run(query, {**base, "rows": chunk}).consume()
            return len(rows)

        def call() -> int:
            with self._driver.session(database=self.database) as session:
                return session.execute_write(work)

        return self._with_retry(call)

    def write_transaction_sync(
        self,
        steps: Sequence[tuple[str, Mapping[str, Any], Sequence[Mapping[str, Any]] | None]],
        timeout_s: float = TX_TIMEOUT_S,
        batch_size: int = WRITE_BATCH_SIZE,
    ) -> None:
        """Run several statements in ONE transaction (1.4 A7: one transaction per Act).

        Each step is (query, params, rows). When rows is not None the query runs once per
        batch with the batch as $rows (UNWIND batches of 200); an empty rows list skips it.
        """

        @unit_of_work(timeout=timeout_s)
        def work(tx: ManagedTransaction) -> None:
            for query, params, rows in steps:
                if rows is None:
                    tx.run(query, dict(params)).consume()
                    continue
                for chunk in batched(rows, batch_size):
                    tx.run(query, {**params, "rows": chunk}).consume()

        def call() -> None:
            with self._driver.session(database=self.database) as session:
                session.execute_write(work)

        self._with_retry(call)

    def run_autocommit_sync(
        self, query: str, params: Mapping[str, Any] | None = None, database: str | None = None
    ) -> dict[str, int]:
        """Auto-commit statement for schema DDL and admin commands. Returns change counters."""

        def call() -> dict[str, int]:
            with self._driver.session(database=database or self.database) as session:
                summary = session.run(query, dict(params or {})).consume()
            counters = summary.counters
            return {
                "constraints_added": counters.constraints_added,
                "constraints_removed": counters.constraints_removed,
                "indexes_added": counters.indexes_added,
                "indexes_removed": counters.indexes_removed,
                "nodes_created": counters.nodes_created,
                "properties_set": counters.properties_set,
                "system_updates": counters.system_updates,
            }

        return self._with_retry(call)

    # Async API (request handlers, agent nodes). Blocking work runs in a thread (1.2).

    async def verify_connectivity(self) -> None:
        await anyio.to_thread.run_sync(self.verify_connectivity_sync)

    async def read(self, query: str, params: Mapping[str, Any] | None = None) -> list[dict]:
        return await anyio.to_thread.run_sync(partial(self.read_sync, query, params))

    async def write(self, query: str, params: Mapping[str, Any] | None = None) -> list[dict]:
        return await anyio.to_thread.run_sync(partial(self.write_sync, query, params))

    async def write_batches(
        self,
        query: str,
        rows: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any] | None = None,
        batch_size: int = WRITE_BATCH_SIZE,
    ) -> int:
        return await anyio.to_thread.run_sync(
            partial(self.write_batches_sync, query, rows, params, batch_size)
        )
