"""Reciprocal rank fusion of ranked result lists (R-10, constant 60).

Each list is ordered best first. An item's fused score is the sum of 1 / (60 + rank) over the
lists it appears in, with rank starting at 1. Ties are broken by the best single rank, then
by id, so the order is deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

RRF_K: Final = 60  # R-10, 4.2


def rrf_fuse(
    lists: Sequence[Sequence[dict]], *, rrf_k: int = RRF_K, id_key: str = "id"
) -> list[dict]:
    """Fuse ranked lists of dicts keyed by `id_key`. The returned dicts merge the fields of
    every occurrence (earlier lists win on conflicts) and add `rrf_score` and `ranks`, the
    1-based rank in each input list or None where the item is absent. The input `score`
    fields stay as they were, because vector and full-text scores are not comparable."""
    fused: dict[str, dict] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, list[int | None]] = {}
    for list_index, rows in enumerate(lists):
        seen: set[str] = set()
        for position, row in enumerate(rows, start=1):
            item_id = row[id_key]
            if item_id in seen:  # a repeated id within one list counts once, at its best rank
                continue
            seen.add(item_id)
            if item_id not in fused:
                fused[item_id] = dict(row)
                scores[item_id] = 0.0
                ranks[item_id] = [None] * len(lists)
            else:
                for key, value in row.items():
                    fused[item_id].setdefault(key, value)
            scores[item_id] += 1.0 / (rrf_k + position)
            ranks[item_id][list_index] = position

    def order(item_id: str) -> tuple[float, int, str]:
        best = min(r for r in ranks[item_id] if r is not None)
        return (-scores[item_id], best, item_id)

    result = []
    for item_id in sorted(fused, key=order):
        item = fused[item_id]
        item["rrf_score"] = scores[item_id]
        item["ranks"] = ranks[item_id]
        result.append(item)
    return result
