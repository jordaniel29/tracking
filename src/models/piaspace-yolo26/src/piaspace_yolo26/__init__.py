"""piaspace-yolo26 — YOLO26 object-detection wrapper.

Image in, ``list[Detection]`` out. Configured via a flat dict;
``target_classes`` maps logical names (``"person"``, ``"vehicle"``) to lists
of COCO class ids so the downstream pipeline can route crops to the right
ReID model.

Public API:
    from piaspace_yolo26 import YOLO26Detector, Detection

    det = YOLO26Detector(cfg["detector"])
    detections = det.detect_image(frame_bgr)  # list[Detection]
"""

from __future__ import annotations

from .yolo26 import Detection, YOLO26Detector

__all__ = ["Detection", "YOLO26Detector"]
