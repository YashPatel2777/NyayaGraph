"""Reciprocal rank fusion, constant 60 (R-10)."""

from __future__ import annotations

import pytest

from app.services.retrieval.hybrid import RRF_K, rrf_fuse


def test_constant_is_60() -> None:
    assert RRF_K == 60


def test_scores_are_sums_of_reciprocal_ranks() -> None:
    vector = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]
    fulltext = [{"id": "b", "score": 7.0}, {"id": "c", "score": 3.0}]
    fused = {r["id"]: r for r in rrf_fuse([vector, fulltext])}
    assert fused["b"]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"]["rrf_score"] == pytest.approx(1 / 61)
    assert fused["c"]["rrf_score"] == pytest.approx(1 / 62)
    assert fused["b"]["ranks"] == [2, 1] and fused["c"]["ranks"] == [None, 2]


def test_order_and_tie_break() -> None:
    vector = [{"id": "a"}, {"id": "b"}]
    fulltext = [{"id": "b"}, {"id": "a"}, {"id": "c"}]
    # a and b tie on score; both have best rank 1, so id decides.
    assert [r["id"] for r in rrf_fuse([vector, fulltext])] == ["a", "b", "c"]


def test_first_list_wins_field_conflicts_and_fields_are_merged() -> None:
    fused = rrf_fuse([[{"id": "a", "score": 0.9}], [{"id": "a", "score": 7.0, "text": "t"}]])
    assert fused[0]["score"] == 0.9 and fused[0]["text"] == "t"


def test_repeated_id_in_one_list_counts_once() -> None:
    fused = rrf_fuse([[{"id": "a"}, {"id": "a"}]])
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61)


def test_inputs_are_not_mutated_and_empty_lists_work() -> None:
    row = {"id": "a"}
    rrf_fuse([[row], []])
    assert row == {"id": "a"}
    assert rrf_fuse([]) == [] and rrf_fuse([[], []]) == []
