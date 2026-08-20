"""Private matching math: box geometry + assignment solvers.

Geometry here is the generic part only — algorithm-specific geometry
(buffered IoU, shape similarity) stays in the tracker module that defines
its semantics.

Both assignment solvers reject infeasible pairs and return the same
``(matches, unmatched_rows, unmatched_cols)`` shape:

* ``greedy_match_min`` — first-come-first-served over the cost-sorted pair
  list. The default (``assignment: greedy``); the shipped gates were tuned
  against it.
* ``hungarian_match`` — globally optimal (scipy); greedy can starve a row
  when one column is contested by two rows. Takes a feasibility MASK so any
  gating vocabulary maps onto it (here: ``cost <= threshold``).
"""

from __future__ import annotations

import numpy as np

# Sentinel cost for infeasible pairs in hungarian_match. Far above any real
# cost (fused IoU/appearance costs live in [0, 1]) so the solver only lands on
# one when a row/col has no feasible partner at all — those assignments are
# dropped afterwards.
INFEASIBLE_COST = 1e6

# (matches, unmatched_rows, unmatched_cols) — the shape all solvers return.
MatchResult = tuple[list[tuple[int, int]], list[int], list[int]]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy boxes. (M, 4), (N, 4) → (M, N).

    Divide-safe on degenerate (zero-area) boxes — returns 0.0 for them without
    emitting a RuntimeWarning.
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)
    ax1, ay1, ax2, ay2 = boxes_a.T
    bx1, by1, bx2, by2 = boxes_b.T
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def xyxy_to_cxcywh(bbox: "tuple[float, float, float, float] | np.ndarray") -> np.ndarray:
    """xyxy box (tuple or array-like) → (4,) ``[cx, cy, w, h]`` float64."""
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float64)


# ---------------------------------------------------------------------------
# Assignment solvers
# ---------------------------------------------------------------------------


def hungarian_match(cost: np.ndarray, feasible: np.ndarray) -> MatchResult:
    """Optimal assignment minimizing ``cost`` over ``feasible`` pairs.

    Infeasible pairs can never match — identical gating to the greedy matcher.
    scipy is imported lazily so this module stays cheap to import.
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0 or not feasible.any():
        return [], list(range(n_rows)), list(range(n_cols))

    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(np.where(feasible, cost, INFEASIBLE_COST))
    matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if feasible[r, c]]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    return (
        matches,
        [r for r in range(n_rows) if r not in matched_rows],
        [c for c in range(n_cols) if c not in matched_cols],
    )


def greedy_match_min(cost: np.ndarray, threshold: float) -> MatchResult:
    """Greedy min-cost assignment. Pairs with ``cost > threshold`` are rejected.

    First-come-first-served on the cost-sorted N² pair list.
    """
    n_rows, n_cols = cost.shape
    matches: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    if n_rows == 0 or n_cols == 0:
        return matches, list(range(n_rows)), list(range(n_cols))
    flat = [
        (cost[r, c], r, c) for r in range(n_rows) for c in range(n_cols) if cost[r, c] <= threshold
    ]
    flat.sort(key=lambda x: x[0])
    for _, r, c in flat:
        if r in used_rows or c in used_cols:
            continue
        matches.append((r, c))
        used_rows.add(r)
        used_cols.add(c)
    unmatched_rows = [r for r in range(n_rows) if r not in used_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in used_cols]
    return matches, unmatched_rows, unmatched_cols
