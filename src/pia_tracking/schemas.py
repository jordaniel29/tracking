"""Detection and Track — the two data contracts the pipeline passes around.

Raw detector output is normalised into ``Detection`` by
``pia_tracking.detection.to_schema_detection``.
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
    """Single-camera track. ``track_id`` is stable within ONE camera;
    ``global_id``, when set, is stable across cameras."""

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
    global_id: int | None = Field(
        None,
        description=(
            "Cross-camera identity from pia_tracking.fusion. None in single-camera "
            "runs, and in multi-camera runs until the worker has enough evidence to "
            "name the person."
        ),
    )

