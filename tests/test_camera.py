"""camera/: video discovery, exclude patterns, the shared clock, round-robin sync."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from pia_tracking.camera import EPOCH, Exhausted, Frame, VideoSource, discover_videos, round_robin


def _touch(d: Path, *names: str) -> None:
    for n in names:
        (d / n).write_bytes(b"")


def test_discover_dir_sorted_and_grid_excluded_by_default(tmp_path):
    _touch(tmp_path, "cam9.mp4", "cam10.mp4", "grid_scenario1.mp4", "notes.txt")
    assert [p.name for p in discover_videos(videos_dir=tmp_path)] == ["cam10.mp4", "cam9.mp4"]


def test_discover_custom_exclude_and_explicit_files(tmp_path):
    _touch(tmp_path, "a.mp4", "b.mp4", "test_c.mp4")
    assert [p.name for p in discover_videos(videos_dir=tmp_path, exclude=("test_*",))] == ["a.mp4", "b.mp4"]
    assert [p.name for p in discover_videos(videos=[tmp_path / "test_c.mp4"], exclude=())] == ["test_c.mp4"]


def test_discover_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        discover_videos()
    with pytest.raises(NotADirectoryError):
        discover_videos(videos_dir=tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        discover_videos(videos=[tmp_path / "missing.mp4"])
    _touch(tmp_path, "grid_1.mp4")
    with pytest.raises(FileNotFoundError):  # everything excluded
        discover_videos(videos_dir=tmp_path)
    (tmp_path / "sub").mkdir()
    _touch(tmp_path / "sub", "cam1.mp4")
    _touch(tmp_path, "cam1.mp4")
    with pytest.raises(ValueError):  # duplicate stems → outputs would collide
        discover_videos(videos=[tmp_path / "cam1.mp4", tmp_path / "sub" / "cam1.mp4"])


class _FakeCap:
    """Stands in for cv2.VideoCapture: yields ``n`` frames, then ends."""

    def __init__(self, n: int) -> None:
        self.n, self.i, self.released = n, 0, False

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, self.i  # any object works as an "image" here

    def release(self) -> None:
        self.released = True


def _source(name: str, n: int, fps: float = 30.0) -> VideoSource:
    return VideoSource(path=Path(f"{name}.mp4"), camera_id=name, fps=fps, _cap=_FakeCap(n))


def test_source_clock_and_read_indexing():
    src = _source("cam1", 2, fps=10.0)
    assert src.read() == (0, 1) and src.read() == (1, 2) and src.read() is None
    assert src.ts(0) == EPOCH and src.ts(5) == EPOCH + timedelta(seconds=0.5)
    assert src.ts() == EPOCH + timedelta(seconds=0.2)  # after the last frame


def test_round_robin_interleaves_and_reports_exhaustion_once():
    a, b = _source("a", 3), _source("b", 1)
    events = list(round_robin([a, b]))
    order = [(e.source.camera_id, e.frame_idx) if isinstance(e, Frame) else (e.source.camera_id, "end") for e in events]
    assert order == [("a", 0), ("b", 0), ("a", 1), ("b", "end"), ("a", 2), ("a", "end")]
    assert a._cap.released and b._cap.released
    assert isinstance(events[3], Exhausted) and events[3].ts == b.ts(1)


def test_round_robin_max_frames_caps_every_source():
    a, b = _source("a", 10), _source("b", 10)
    frames = [e for e in round_robin([a, b], max_frames=2) if isinstance(e, Frame)]
    assert [(f.source.camera_id, f.frame_idx) for f in frames] == [("a", 0), ("b", 0), ("a", 1), ("b", 1)]
