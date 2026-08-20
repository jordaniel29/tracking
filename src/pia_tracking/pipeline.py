"""Single-camera person tracking pipeline: detect → embed → associate.

Frames are processed one at a time, matching how a live per-camera process
consumes video. Each frame:

1. **Detect** — YOLO26 person detections (``piaspace_yolo26``).
2. **Embed** — CLIP-ReID appearance vector per detection crop
   (``piaspace_clip_reid``), batched in one call per frame.
3. **Associate** — BoostTrack++ fuses geometry and appearance into one cost and
   assigns stable per-camera track ids.

The tracker owns all cross-frame state; detector and ReID are stateless per
frame, which is why they are built once and reused across clips.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .schemas import Detection, Track, to_schema_detection

logger = logging.getLogger("pia_tracking.pipeline")

# Minimum crop side, in pixels, that will be handed to ReID. Below this the
# 256x128 CLIP-ReID input is mostly interpolation and the embedding is noise, so
# such detections are tracked on geometry alone rather than poisoning the
# appearance channel.
MIN_REID_CROP_PX = 8


@dataclass
class FrameResult:
    """Tracks for one frame, plus the raw detections they came from."""

    frame_idx: int
    tracks: list[Track]
    detections: list[Detection]


@dataclass
class RunStats:
    """Totals for one clip. ``frames`` counts frames PROCESSED, not frames with tracks."""

    frames: int = 0
    detections: int = 0
    track_rows: int = 0
    track_ids: set[int] = field(default_factory=set)

    @property
    def coverage(self) -> float:
        """Fraction of detections that became a tracked row.

        With no ground truth this is the only quantitative detection signal
        available: it separates detector recall from tracker spawn gating
        (``new_track_thresh`` / ``min_hits``). It is NOT accuracy — it cannot say
        whether an untracked detection was a real person or a false positive.
        """
        return self.track_rows / self.detections if self.detections else 0.0


def crop_bgr(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    """Clamped crop for one bbox, or None when it is degenerate.

    Not used by :class:`TrackingPipeline` — the tracker cuts its own crops for the
    appearance channel. Exposed for callers that need person crops for their own
    purposes (thumbnails, an external classifier).

    Detector boxes can run past the frame edge (a person half out of view), so
    clamping here rather than trusting the box avoids a zero-width slice that
    would make the whole ReID batch fail.
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 - x1 < MIN_REID_CROP_PX or y2 - y1 < MIN_REID_CROP_PX:
        return None
    return frame[y1:y2, x1:x2]


class TrackingPipeline:
    """Detector + ReID + tracker for ONE camera.

    Args:
        detector: object exposing ``detect(frame_bgr) -> list[Detection]``.
        tracker: object exposing ``update(detections, frame) -> list[Track]``.
        reid: optional embedder exposing ``embed(list[crop_bgr]) -> (N, D)``.
            When None the tracker associates on geometry alone.
        camera_id: stamped onto every emitted Track.
    """

    def __init__(
        self,
        *,
        detector: Any,
        tracker: Any,
        reid: Any | None = None,
        camera_id: str = "cam0",
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._reid = reid
        self._camera_id = camera_id
        self.stats = RunStats()

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        """Run one frame end to end and return its tracks."""
        # Detector adapters differ in field names (.conf vs .confidence); the
        # tracker reads .confidence. Normalising HERE, once, is why a detector
        # swap cannot break association with a mid-frame AttributeError.
        detections = [to_schema_detection(d) for d in self._detector.detect(frame)]
        self.stats.frames += 1
        self.stats.detections += len(detections)

        tracks = self._tracker.update(detections, frame)
        stamped = [
            Track(
                track_id=t.track_id,
                camera_id=self._camera_id,
                detection=t.detection,
                frame_idx=frame_idx,
                timestamp=t.timestamp if getattr(t, "timestamp", None) else _now(),
                state=t.state,
            )
            for t in tracks
        ]
        self.stats.track_rows += len(stamped)
        self.stats.track_ids.update(t.track_id for t in stamped)
        return FrameResult(frame_idx=frame_idx, tracks=stamped, detections=detections)


def _now() -> datetime:
    return datetime.now(timezone.utc)
