"""Rank-shift distance metrics (spec section 五).

Three complementary distances, each normalized to [0, 1]:

  D_pair : pairwise order distortion           (spec 五.2)
  D_foot  : normalized Spearman footrule        (spec 五.3)
  D_top   : Top-K set replacement               (spec 五.4)

Final per-group distance:
    D   = 0.60 * D_pair + 0.25 * D_foot + 0.15 * D_top
    RankShift = 100 * D
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..schema import Score
from .ranking import average_ranks, build_ranking, sort_value, top_k

DEFAULT_WEIGHTS = {"w_pair": 0.60, "w_foot": 0.25, "w_top": 0.15}
DEFAULT_TOPK = {"top_k_ratio": 0.10, "top_k_min": 10}


@dataclass
class ShiftBreakdown:
    d_pair: float
    d_foot: float
    d_top: float
    d: float
    rank_shift: float
    n: int
    k: int

    def to_dict(self) -> dict:
        return {
            "D_pair": round(self.d_pair, 6),
            "D_foot": round(self.d_foot, 6),
            "D_top": round(self.d_top, 6),
            "D": round(self.d, 6),
            "RankShift": round(self.rank_shift, 4),
            "N": self.n,
            "K": self.k,
        }


def _pair_distance(
    ref_values: np.ndarray, new_values: np.ndarray
) -> float:
    """D_pair: mean pairwise cost over all C(N,2) unordered pairs."""
    n = len(ref_values)
    if n < 2:
        return 0.0
    # sign matrices over all ordered pairs
    s0 = np.sign(ref_values[:, None] - ref_values[None, :])
    s1 = np.sign(new_values[:, None] - new_values[None, :])
    # cost per ordered pair
    same = s0 == s1  # includes both 0 (tied in both)
    reversed_ = (s0 == -s1) & (s0 != 0)  # fully reversed, both nonzero
    one_tied = (~same) & (~reversed_)  # exactly one side tied
    cost = np.where(reversed_, 1.0, np.where(one_tied, 0.5, 0.0))
    # sum over unordered pairs (i<j): the matrix is symmetric in cost, so
    # total over i!=j divided by 2 equals sum over i<j.
    iu = np.triu_indices(n, k=1)
    total = float(cost[iu].sum())
    return total / (n * (n - 1) / 2.0)


def _foot_distance(
    ref_ranks: np.ndarray, new_ranks: np.ndarray, n: int
) -> float:
    """D_foot = sum|r0_i - r_i| / floor(N^2 / 2)."""
    if n < 1:
        return 0.0
    denom = math.floor(n * n / 2.0)
    if denom == 0:
        return 0.0
    f = float(np.abs(ref_ranks - new_ranks).sum())
    return f / denom


def _top_distance(
    ref_ranking: list[str], new_ranking: list[str], n: int, k: int
) -> float:
    """D_top = 1 - |TopK(R0) ∩ TopK(R)| / K."""
    if k <= 0 or n == 0:
        return 0.0
    top0 = top_k(ref_ranking, k)
    top1 = top_k(new_ranking, k)
    inter = len(top0 & top1)
    return 1.0 - inter / k


def compute_topk(n: int, ratio: float = 0.10, minimum: int = 10) -> int:
    """K = max(minimum, ceil(ratio * N)) (spec 五.4)."""
    return max(minimum, math.ceil(ratio * n))


def rank_shift(
    reference: Mapping[str, Score],
    new_scores: Mapping[str, Score],
    weights: dict[str, float] | None = None,
    topk: dict[str, float] | None = None,
) -> ShiftBreakdown:
    """Compute the full RankShift between a reference and a new scoring.

    ``reference`` and ``new_scores`` must share the same set of sample_ids.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    tk = {**DEFAULT_TOPK, **(topk or {})}

    ids = list(reference.keys())
    n = len(ids)
    if n == 0:
        return ShiftBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    # Guard: new_scores must cover the same ids; missing -> treat as invalid
    # (caller should have substituted reference scores already).
    new_aligned = {sid: new_scores.get(sid, reference[sid]) for sid in ids}

    ref_vals = np.array([sort_value(reference[sid]) for sid in ids], dtype=np.int64)
    new_vals = np.array([sort_value(new_aligned[sid]) for sid in ids], dtype=np.int64)

    ref_ranks_map = average_ranks(reference)
    new_ranks_map = average_ranks(new_aligned)
    ref_ranks = np.array([ref_ranks_map[sid] for sid in ids], dtype=np.float64)
    new_ranks = np.array([new_ranks_map[sid] for sid in ids], dtype=np.float64)

    ref_ranking = build_ranking(reference)
    new_ranking = build_ranking(new_aligned)

    k = compute_topk(n, tk["top_k_ratio"], int(tk["top_k_min"]))

    d_pair = _pair_distance(ref_vals, new_vals)
    d_foot = _foot_distance(ref_ranks, new_ranks, n)
    d_top = _top_distance(ref_ranking, new_ranking, n, k)

    d = w["w_pair"] * d_pair + w["w_foot"] * d_foot + w["w_top"] * d_top
    return ShiftBreakdown(d_pair, d_foot, d_top, d, 100.0 * d, n, k)
