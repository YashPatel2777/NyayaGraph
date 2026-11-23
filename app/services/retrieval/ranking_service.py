"""FlashRank cross-encoder rerank (C4, C8).

The Ranker loads once in the FastAPI lifespan (1.2). Reranking is blocking ONNX inference,
so async callers go through anyio.to_thread.run_sync (1.2). FLASHRANK_MODEL empty means the
library default model (1.3). R-04: logs carry counts and timings only.

The score range of FlashRank is not verified (README open item); RERANK_HIGH (R-24) is
calibrated against the scores this service returns.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from functools import partial
from typing import Any

import anyio

logger = logging.getLogger(__name__)


class RankingService:
    def __init__(self, model_name: str = "", *, ranker: Any | None = None) -> None:
        if ranker is None:
            from flashrank import Ranker

            ranker = Ranker(model_name=model_name) if model_name else Ranker()
        self._ranker = ranker

    def rerank_sync(
        self, query: str, items: Sequence[dict], *, text_key: str = "text", top_n: int | None = None
    ) -> list[dict]:
        """Return copies of `items` with `rerank_score` added, best first, cut to top_n.
        Items need a unique `id`; ties keep the input order."""
        if not items:
            return []
        from flashrank import RerankRequest

        started = time.perf_counter()
        passages = [{"id": index, "text": item[text_key]} for index, item in enumerate(items)]
        scored = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        by_index = {int(p["id"]): float(p["score"]) for p in scored}
        ranked = sorted(range(len(items)), key=lambda i: (-by_index[i], i))
        if top_n is not None:
            ranked = ranked[:top_n]
        logger.info(
            "rerank: items=%d kept=%d ms=%d",
            len(items), len(ranked), int((time.perf_counter() - started) * 1000),
        )
        return [{**items[i], "rerank_score": by_index[i]} for i in ranked]

    async def rerank(
        self, query: str, items: Sequence[dict], *, text_key: str = "text", top_n: int | None = None
    ) -> list[dict]:
        return await anyio.to_thread.run_sync(
            partial(self.rerank_sync, query, items, text_key=text_key, top_n=top_n)
        )
