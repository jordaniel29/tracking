"""utils/mosaic.py — grid layout and composition."""

from __future__ import annotations

import numpy as np
import pytest

from pia_tracking.utils import cell_size, compose, grid_shape


def test_grid_shape_is_square_ish_and_never_taller_than_wide():
    assert grid_shape(1) == (1, 1)
    assert grid_shape(2) == (1, 2)
    assert grid_shape(4) == (2, 2)
    assert grid_shape(5) == (2, 3)
    assert grid_shape(9) == (3, 3)
    for n in range(1, 30):
        rows, cols = grid_shape(n)
        assert rows * cols >= n and rows <= cols


def test_grid_shape_explicit_cols_and_bad_input():
    assert grid_shape(5, cols=1) == (5, 1)
    assert grid_shape(5, cols=5) == (1, 5)
    with pytest.raises(ValueError):
        grid_shape(0)
    with pytest.raises(ValueError):
        grid_shape(4, cols=0)


def test_cell_size_keeps_aspect_and_is_even():
    w, h = cell_size(1920, 1080, cols=3, total_width=1920)
    assert (w, h) == (640, 360)
    for cols in range(1, 8):
        w, h = cell_size(1920, 1080, cols=cols)
        assert w % 2 == 0 and h % 2 == 0
        assert abs(w / h - 1920 / 1080) < 0.05


def test_compose_tiles_row_major_and_pads_with_black():
    cell = (4, 2)
    a = np.full((2, 4, 3), 10, np.uint8)
    b = np.full((2, 4, 3), 20, np.uint8)
    out = compose([a, b, None], cell=cell, cols=2)
    assert out.shape == (4, 8, 3)  # 2 rows x 2 cols of 4x2 cells
    assert out[0, 0, 0] == 10 and out[0, 4, 0] == 20  # row-major
    assert out[2:, :, :].max() == 0  # the None and the empty slot are black


def test_compose_resizes_mismatched_cells():
    out = compose([np.full((1080, 1920, 3), 7, np.uint8)], cell=(64, 36), cols=1)
    assert out.shape == (36, 64, 3) and out.min() == 7
