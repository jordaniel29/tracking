"""runners/render.py — draw stored predictions onto a video, no models."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pia_tracking.runners import RunOptions, run_render
from pia_tracking.runners.render import read_mot

N_FRAMES, W, H = 8, 96, 64


def _make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    assert writer.isOpened(), "mp4v codec unavailable"
    for i in range(N_FRAMES):
        writer.write(np.full((H, W, 3), i * 20, dtype=np.uint8))
    writer.release()


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


@pytest.fixture
def run_dir(tmp_path: Path) -> tuple[Path, Path]:
    src = tmp_path / "src"
    src.mkdir()
    _make_video(src / "camA.mp4")
    _make_video(src / "camB.mp4")
    out = tmp_path / "run"
    (out / "preds").mkdir(parents=True)
    # camA: track 1 on every frame, track 2 on the last two; camB: nothing predicted at all
    rows = [f"{f},1,10.00,10.00,20.00,30.00,0.9000,-1,-1,-1" for f in range(N_FRAMES)]
    rows += [f"{f},2,50.00,5.00,15.00,25.00,0.8000,-1,-1,-1" for f in (N_FRAMES - 2, N_FRAMES - 1)]
    (out / "preds" / "camA.txt").write_text("\n".join(rows) + "\n")
    return src, out


def test_read_mot_groups_by_frame(run_dir):
    _, out = run_dir
    rows = read_mot(out / "preds" / "camA.txt")
    assert len(rows) == N_FRAMES and len(rows[N_FRAMES - 1]) == 2
    assert rows[0][0] == (1, 10.0, 10.0, 20.0, 30.0, 0.9)


def test_render_single_camera_run(run_dir):
    src, out = run_dir
    rc = run_render([src / "camA.mp4"], out_dir=out, opts=RunOptions())
    assert rc == 0
    assert _frame_count(out / "camA.mp4") == N_FRAMES


def test_render_multi_camera_run_uses_global_ids_and_reports_missing_preds(run_dir):
    src, out = run_dir
    (out / "global_ids.json").write_text(json.dumps({"local_to_global": {"camA": {"1": 7}}}))  # track 2 unlabelled
    rc = run_render([src / "camA.mp4", src / "camB.mp4"], out_dir=out, opts=RunOptions(max_frames=5))
    assert rc == 1  # camB has no predictions → skipped, non-zero exit
    assert _frame_count(out / "camA.mp4") == 5
    assert not (out / "camB.mp4").exists()


def test_render_requires_a_run_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_render([], out_dir=tmp_path / "nope", opts=RunOptions())
