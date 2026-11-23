"""FlashRank wrapper: ordering, top_n, and that blocking work runs off the event loop (1.2)."""

from __future__ import annotations

import threading

from app.services.retrieval.ranking_service import RankingService


class FakeRanker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.thread: threading.Thread | None = None

    def rerank(self, request):
        self.thread = threading.current_thread()
        out = [{**p, "score": self.scores[p["text"]]} for p in request.passages]
        return sorted(out, key=lambda p: -p["score"])


def test_rerank_orders_by_score_and_cuts_to_top_n() -> None:
    svc = RankingService(ranker=FakeRanker({"x": 0.1, "y": 0.9, "z": 0.5}))
    items = [{"id": "1", "text": "x"}, {"id": "2", "text": "y"}, {"id": "3", "text": "z"}]
    out = svc.rerank_sync("q", items, top_n=2)
    assert [(o["id"], o["rerank_score"]) for o in out] == [("2", 0.9), ("3", 0.5)]
    assert "rerank_score" not in items[0]


def test_ties_keep_input_order_and_text_key_is_configurable() -> None:
    svc = RankingService(ranker=FakeRanker({"a": 0.5, "b": 0.5}))
    out = svc.rerank_sync("q", [{"id": "1", "t": "a"}, {"id": "2", "t": "b"}], text_key="t")
    assert [o["id"] for o in out] == ["1", "2"]


def test_empty_input() -> None:
    assert RankingService(ranker=FakeRanker({})).rerank_sync("q", []) == []


async def test_async_rerank_runs_in_a_worker_thread() -> None:
    ranker = FakeRanker({"x": 0.3})
    out = await RankingService(ranker=ranker).rerank("q", [{"id": "1", "text": "x"}])
    assert out[0]["rerank_score"] == 0.3
    assert ranker.thread is not threading.main_thread()
