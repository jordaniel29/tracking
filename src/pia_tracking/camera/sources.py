"""Video sources — which files are cameras, opening them, and their clock.

Offline files stand in for live feeds: one file per camera, all recorded
together, so frame k of every file is the same instant. Time is synthetic,
``EPOCH + frame_idx / fps``. Only differences matter downstream (the fusion
recency window, same-camera span overlap), so a fixed origin keeps timestamps
reproducible instead of depending on the wall clock.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger("pia_tracking.camera")

VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".m4v")
DEFAULT_EXCLUDE = ("grid_*",)  # composite grid views are not cameras
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def discover_videos(
    *,
    videos: Sequence[Path] | None = None,
    videos_dir: Path | None = None,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
) -> list[Path]:
    """The input videos: an explicit list, or every video in a directory, minus
    ``exclude`` glob patterns. Stems must be unique — every output file is
    named by stem, so a clash would silently overwrite."""
    if (videos is None) == (videos_dir is None):
        raise ValueError("pass exactly one of videos / videos_dir")
    if videos is not None:
        missing = [p for p in videos if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"video not found: {', '.join(map(str, missing))}")
        found = list(videos)
    else:
        assert videos_dir is not None
        if not videos_dir.is_dir():
            raise NotADirectoryError(str(videos_dir))
        found = sorted(p for p in videos_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
        if not found:
            raise FileNotFoundError(f"no videos in {videos_dir} matching {VIDEO_SUFFIXES}")

    kept = [v for v in found if not any(fnmatch.fnmatch(v.name, pat) for pat in exclude)]
    if len(kept) < len(found):
        logger.info("excluded %d file(s) matching %s", len(found) - len(kept), list(exclude))
    if not kept:
        raise FileNotFoundError("every input video was excluded")
    stems = [v.stem for v in kept]
    if len(set(stems)) != len(stems):
        raise ValueError(f"video file stems must be unique (outputs are named by stem): {stems}")
    return kept


@dataclass
class VideoSource:
    """One camera's frames, read in order, with the shared clock."""

    path: Path
    camera_id: str
    fps: float
    _cap: cv2.VideoCapture = field(repr=False)
    frames_read: int = 0

    def ts(self, frame_idx: int | None = None) -> datetime:
        """Time of ``frame_idx`` (default: the moment after the last frame read)."""
        idx = self.frames_read if frame_idx is None else frame_idx
        return EPOCH + timedelta(seconds=idx / self.fps)

    def read(self) -> tuple[int, np.ndarray] | None:
        """``(frame_idx, image)`` for the next frame, or None at end of file."""
        ok, image = self._cap.read()
        if not ok:
            return None
        idx = self.frames_read
        self.frames_read += 1
        return idx, image

    def close(self) -> None:
        self._cap.release()


def open_source(path: Path) -> VideoSource:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    return VideoSource(path=path, camera_id=path.stem, fps=cap.get(cv2.CAP_PROP_FPS) or 25.0, _cap=cap)
