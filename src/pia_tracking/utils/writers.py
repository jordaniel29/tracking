"""Output files: MOTChallenge text, annotated MP4, JSON.

The MOT file and the MP4 are deliberately separate outputs of the same rows:
the text is the machine-readable result (feed it to any MOT scorer), the video
is the same boxes drawn on the source frames for a human.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ..schemas import Detection, Track

logger = logging.getLogger("pia_tracking.writers")

MOTRow = tuple[int, float, float, float, float, float]  # track_id, x, y, w, h, conf


def detections_to_mot_rows(frame_idx: int, detections: Iterable[Detection]) -> list[str]:
    """Raw pre-tracking detections as MOT lines with id = -1."""
    return [
        f"{frame_idx},-1,{d.bbox[0]:.2f},{d.bbox[1]:.2f},{d.bbox[2] - d.bbox[0]:.2f},"
        f"{d.bbox[3] - d.bbox[1]:.2f},{d.confidence:.4f},-1,-1,-1\n"
        for d in detections
    ]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


class MOTWriter:
    """Buffers tracked rows and writes one MOTChallenge file
    (``frame,id,x,y,w,h,conf,-1,-1,-1``)."""

    def __init__(self) -> None:
        self._rows: list[tuple[int, int, float, float, float, float, float]] = []

    def add(self, frame_idx: int, tracks: Iterable[Track]) -> None:
        for t in tracks:
            x1, y1, x2, y2 = t.detection.bbox
            self._rows.append((frame_idx, t.track_id, x1, y1, x2 - x1, y2 - y1, t.detection.confidence))

    def rows_by_frame(self) -> dict[int, list[MOTRow]]:
        """``frame_idx → [(track_id, x, y, w, h, conf), ...]`` — for a second
        rendering pass once ids that were assigned late are known."""
        out: dict[int, list[MOTRow]] = {}
        for frame, tid, x, y, w, h, conf in self._rows:
            out.setdefault(frame, []).append((tid, x, y, w, h, conf))
        return out

    def write(self, path: Path, *, id_map: dict[int, int] | None = None, drop_unmapped: bool = False) -> Path:
        """Write the buffered rows.

        ``id_map`` (local id → another id, e.g. a global id assigned after the
        rows were buffered) remaps the id column. Rows whose id is not in the
        map keep it, or are skipped when ``drop_unmapped`` is set.
        """
        rows = self._rows if id_map is None else self._remapped(id_map, drop_unmapped)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for frame, tid, x, y, w, h, conf in sorted(rows, key=lambda r: (r[0], r[1])):
                fh.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")
        logger.info("mot_written path=%s rows=%d", path, len(rows))
        return path

    def _remapped(self, id_map: dict[int, int], drop_unmapped: bool) -> list[tuple]:
        rows: list[tuple] = []
        for frame, tid, *rest in self._rows:
            mapped = id_map.get(tid)
            if mapped is None:
                if drop_unmapped:
                    continue
                mapped = tid
            rows.append((frame, mapped, *rest))
        return rows


class VideoWriter:
    """Lazy cv2.VideoWriter — opened on the first frame so it gets the real size."""

    def __init__(self, path: Path, fps: float) -> None:
        self._path = path
        self._fps = fps if fps and fps > 0 else 25.0
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"cannot open video writer for {self._path} — is the mp4v codec available?")
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            logger.info("video_written path=%s", self._path)
            self._writer = None
