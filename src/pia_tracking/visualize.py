"""Annotated-video rendering and MOTChallenge output.

Two outputs per clip, deliberately separate:

* **MOT file** — ``frame,id,x,y,w,h,conf,-1,-1,-1``, one row per tracked box. The
  machine-readable result; feed it to any MOT scorer.
* **MP4** — the same boxes drawn on the source frames, colour-keyed by track id
  so an ID switch is visible as a colour change on one person.

Colours are derived from the track id by a hash, not assigned in sequence, so the
same person keeps the same colour across a re-render and two adjacent ids do not
get near-identical shades.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .schemas import Track

logger = logging.getLogger("pia_tracking.visualize")

_FONT = cv2.FONT_HERSHEY_SIMPLEX
# Distinct, mid-saturation BGR palette. Avoids pure red/green so the boxes stay
# legible on both the washed-out and the very dark footage in CCTV sets.
_PALETTE = [
    (66, 135, 245), (245, 176, 66), (66, 245, 152), (245, 66, 197),
    (197, 245, 66), (66, 245, 245), (150, 66, 245), (245, 66, 90),
    (110, 200, 120), (200, 120, 200), (120, 200, 200), (240, 140, 80),
]


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Stable BGR colour for a track id."""
    return _PALETTE[hash((track_id, 0x9E37)) % len(_PALETTE)]


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    *,
    show_conf: bool = False,
    label: str | None = None,
) -> np.ndarray:
    """Draw one frame's tracks. Returns a copy; the input is not mutated."""
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.detection.bbox)
        colour = color_for_id(t.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        text = f"id={t.track_id}"
        if show_conf:
            text += f" {t.detection.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, _FONT, 0.5, 1)
        # Keep the label inside the frame when the box touches the top edge,
        # otherwise it is drawn at negative y and silently disappears.
        ty = max(th + 2, y1)
        cv2.rectangle(out, (x1, ty - th - 2), (x1 + tw + 2, ty), colour, -1)
        cv2.putText(out, text, (x1 + 1, ty - 2), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if label:
        cv2.putText(out, label, (8, 22), _FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out


class MOTWriter:
    """Buffers tracked rows and writes one MOTChallenge file."""

    def __init__(self) -> None:
        self._rows: list[tuple[int, int, float, float, float, float, float]] = []

    def add(self, frame_idx: int, tracks: list[Track]) -> None:
        for t in tracks:
            x1, y1, x2, y2 = t.detection.bbox
            self._rows.append(
                (frame_idx, t.track_id, x1, y1, x2 - x1, y2 - y1, t.detection.confidence)
            )

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for frame, tid, x, y, w, h, conf in sorted(self._rows, key=lambda r: (r[0], r[1])):
                fh.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")
        logger.info("mot_written path=%s rows=%d", path, len(self._rows))
        return path


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
            self._writer = cv2.VideoWriter(
                str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h)
            )
            if not self._writer.isOpened():
                raise RuntimeError(
                    f"cannot open video writer for {self._path} — is the mp4v codec available?"
                )
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            logger.info("video_written path=%s", self._path)
            self._writer = None
