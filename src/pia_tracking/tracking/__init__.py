"""Single-camera tracking.

    base.py        ``Tracker`` contract + ``AppearanceTracker`` template (occlusion-gated
                   appearance updates, per-frame embedding export)
    boosttrack.py  BoostTrack++ — the shipped tracker
    matching.py    IoU / assignment math
    pipeline.py    ``TrackingPipeline`` — detect → embed → associate for ONE camera
    factory.py     ``build_tracker(cfg, reid)`` — config block → tracker
"""

from .base import AppearanceTracker, Tracker
from .boosttrack import BoostTrackTracker
from .factory import build_tracker
from .pipeline import FrameResult, RunStats, TrackingPipeline

__all__ = [
    "AppearanceTracker",
    "BoostTrackTracker",
    "FrameResult",
    "RunStats",
    "Tracker",
    "TrackingPipeline",
    "build_tracker",
]
