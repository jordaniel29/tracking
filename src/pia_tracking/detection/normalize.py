"""Normalise raw detector output into the ``Detection`` schema."""

from __future__ import annotations

from ..schemas import Detection


def to_schema_detection(det: object) -> Detection:
    """Detector wrappers name the same value differently (``piaspace_yolo26``
    emits ``.conf``; the schema uses ``.confidence``), while the tracker reads
    ``.confidence`` — so anything piped into a tracker is normalised first.
    Already-schema instances pass through unchanged."""
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
