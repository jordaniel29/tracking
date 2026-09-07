"""Config block → detector instance."""

from __future__ import annotations

from typing import Any


def build_detector(cfg: dict[str, Any]) -> Any:
    """The ``detector:`` block. Ships with YOLO26 (TensorRT engine or .pt) only;
    the object exposes ``detect(frame_bgr) -> list`` of boxes with ``.bbox``,
    ``.class_id`` and ``.conf``."""
    backend = cfg.get("backend", "yolo26")
    if backend != "yolo26":
        raise ValueError(f"unsupported detector backend '{backend}'; this package ships with 'yolo26' only")
    from piaspace_yolo26 import YOLO26Detector

    return YOLO26Detector(cfg)
