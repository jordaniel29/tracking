"""Detection and Track — the two data contracts the pipeline passes around —
plus normalization of raw detector output into them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    """A single object detection result.

    All detector implementations return ``list[Detection]``.
    """

    model_config = ConfigDict(frozen=True)

    bbox: tuple[float, float, float, float] = Field(
        ..., description="Bounding box (x1, y1, x2, y2) in pixel coordinates."
    )
    class_id: int = Field(..., description="Class ID (0 = person for COCO).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence.")
    class_name: str | None = Field(None, description="Optional human-readable class name.")


class Track(BaseModel):
    """Single-camera track. ``track_id`` is stable within ONE camera."""

    model_config = ConfigDict(frozen=True)

    track_id: int = Field(..., description="Tracker-assigned stable ID (per camera).")
    camera_id: str = Field(..., description="Camera identifier.")
    detection: Detection
    frame_idx: int = Field(..., ge=0)
    timestamp: datetime
    state: Literal["tentative", "active", "lost"] = Field(
        "active",
        description=(
            "tentative=just spawned; active=confirmed and matched this frame; "
            "lost=missed but within track buffer."
        ),
    )


def to_schema_detection(det: object) -> Detection:
    """Normalize any detector's output to ``Detection``.

    Detector wrappers name the same value differently (``piaspace_yolo26``
    emits ``.conf``; this schema uses ``.confidence``), while the tracker reads
    ``.confidence`` — so anything piped into a tracker must be normalized
    first. Already-schema Detection instances pass through unchanged.
    """
    if isinstance(det, Detection):
        return det
    confidence = getattr(det, "confidence", None)
    if confidence is None:
        confidence = getattr(det, "conf", 0.0)
    return Detection(
        bbox=det.bbox,
        class_id=det.class_id,
        confidence=float(confidence),
        class_name=getattr(det, "class_name", None),
    )
