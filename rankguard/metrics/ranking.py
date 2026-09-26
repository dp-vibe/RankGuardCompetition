"""Ranking construction (spec section 五.4).

Sort key (all descending):
    overall_score, educational_value, informativeness, trustworthiness,
    coherence, writing_quality

Ties (all six fields equal) are kept as groups; in the distance metrics they
receive *average* ranks (spec 五.3). The deterministic ordering returned by
``build_ranking`` breaks ties by ``sample_id`` ascending only so that Top-K
sets are well-defined; this neutral, content-independent break does not favor
either side (spec 五.4 "不使用样本编号进行有利于任一参赛方的随机打破").
"""

from __future__ import annotations

from typing import Mapping

from ..schema import RANK_TIEBREAKERS, Score

# A packed single integer so numpy can compare in one shot. Base 101 keeps the
# six 0..100 fields collision-free (overall 0..100, axes 0..20).
_PACK_BASE = 101


def sort_value(score: Score) -> int:
    """Pack the tiebreaker tuple into one comparable int (larger = better)."""
    v = 0
    for field in RANK_TIEBREAKERS:
        v = v * _PACK_BASE + int(getattr(score, field, 0))
    return v


def build_ranking(scores_by_id: Mapping[str, Score]) -> list[str]:
    """Return sample_ids ordered best -> worst.

    Primary order: sort_value descending. Stable tiebreak: sample_id ascending
    (deterministic, neutral).
    """
    items = [(sid, sort_value(s)) for sid, s in scores_by_id.items()]
    items.sort(key=lambda t: (-t[1], t[0]))
    return [sid for sid, _ in items]


def average_ranks(scores_by_id: Mapping[str, Score]) -> dict[str, float]:
    """1-based ranks where tied samples share the average of their positions."""
    order = build_ranking(scores_by_id)
    ranks: dict[str, float] = {}
    i = 0
    n = len(order)
    while i < n:
        j = i
        val_i = sort_value(scores_by_id[order[i]])
        while j + 1 < n and sort_value(scores_by_id[order[j + 1]]) == val_i:
            j += 1
        # positions i+1 .. j+1 (1-based) -> average
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def top_k(ranking: list[str], k: int) -> set[str]:
    """First ``k`` sample_ids of an ordered ranking."""
    return set(ranking[:k])
