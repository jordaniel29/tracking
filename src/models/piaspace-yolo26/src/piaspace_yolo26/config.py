"""Constants for the YOLO26 wrapper.

Matches the `config.py` convention used by `piaspace_clip_reid.config` —
per-package constants live here, not scattered across the impl files.
"""

from __future__ import annotations

# Gated HF repo hosting the YOLO26 ONNX artifacts (same repo as CLIP-ReID;
# one HF_TOKEN with access covers both).
HF_REPO = "PIA-SPACE-LAB/SSAVE"

# Default inference settings; YAML config overrides these per call site.
DEFAULT_DEVICE = "cuda:0"
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.5
DEFAULT_IMGSZ = 640
