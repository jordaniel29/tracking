"""Tile several cameras' frames into one grid image.

One grid video per scenario is the view where cross-camera identity is legible
at a glance: the same person carries the same colour and ``G-<n>`` label in
every cell at the same instant.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

DEFAULT_GRID_WIDTH = 1920  # total width of the composed image


def grid_shape(n: int, *, cols: int | None = None) -> tuple[int, int]:
    """``(rows, cols)`` for ``n`` cells — as square as possible, never taller
    than wide (4 → 2x2, 5 → 2x3, 9 → 3x3)."""
    if n < 1:
        raise ValueError(f"need at least one cell, got {n}")
    if cols is not None:
        if cols < 1:
            raise ValueError(f"cols must be >= 1, got {cols}")
        return math.ceil(n / cols), cols
    c = math.ceil(math.sqrt(n))
    return math.ceil(n / c), c


def cell_size(frame_w: int, frame_h: int, cols: int, *, total_width: int = DEFAULT_GRID_WIDTH) -> tuple[int, int]:
    """Per-cell ``(w, h)`` that keeps the source aspect ratio and makes the grid
    ``total_width`` wide. Both are even, which mp4 encoders require."""
    w = max(2, (total_width // cols) & ~1)
    h = max(2, int(round(w * frame_h / frame_w)) & ~1)
    return w, h


def compose(cells: list[np.ndarray | None], *, cell: tuple[int, int], cols: int) -> np.ndarray:
    """Tile ``cells`` row-major into one image, each resized to ``cell``.

    ``None`` (a camera whose video has ended) and the unused slots of the last
    row become black, so the mosaic keeps a fixed size for the whole video.
    """
    w, h = cell
    rows = math.ceil(len(cells) / cols)
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    tiles = [blank if c is None else (c if (c.shape[1], c.shape[0]) == (w, h) else cv2.resize(c, (w, h))) for c in cells]
    tiles += [blank] * (rows * cols - len(tiles))
    return np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
