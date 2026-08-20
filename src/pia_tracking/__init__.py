"""Single-camera person detection + ReID + tracking.

Module map:
    schemas.py    ``Detection`` / ``Track`` models + detector-output normalization
    reid.py       ``ReIDBackend`` — the embedder interface the tracker consumes
    tracker/      ``Tracker`` contract + BoostTrack++ implementation
    pipeline.py   ``TrackingPipeline`` — detect → embed → associate, one camera
    visualize.py  annotated-video rendering + MOTChallenge output

Typical use:

    from pia_tracking import TrackingPipeline
    from pia_tracking.tracker import BoostTrackTracker
"""

from .pipeline import FrameResult, TrackingPipeline
from .reid import ReIDBackend
from .schemas import Detection, Track
from .tracker import BoostTrackTracker, Tracker

__all__ = [
    "BoostTrackTracker",
    "Detection",
    "FrameResult",
    "ReIDBackend",
    "Track",
    "Tracker",
    "TrackingPipeline",
]
__version__ = "0.1.0"
