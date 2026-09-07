"""Person detection + ReID + tracking — single camera, or several with shared ids.

Package map — one folder per stage of the pipeline:

    schemas.py    ``Detection`` / ``Track`` — the data contracts everything passes around
    config.py     ``load_config`` — config/tracking.yaml → dict
    camera/       video sources (discovery, opening, shared clock) + round-robin frame sync
    detection/    detector factory (YOLO26) + output normalisation
    reid/         ``ReIDBackend`` interface + embedder factory (CLIP-ReID)
    tracking/     single-camera: ``Tracker`` contracts, BoostTrack++, ``TrackingPipeline``
    fusion/       cross-camera: ``Tracklet`` → ``GlobalIDService`` → global ids, ``CameraWorker``
    utils/        crops, drawing, MOT / MP4 / JSON writers
    runners/      ``run_single`` / ``run_multi`` — the end-to-end runs behind infer.py

Data flow, per camera:  camera → detection → tracking (local ids) → fusion (global ids) → utils (outputs)

Typical use:

    from pia_tracking import TrackingPipeline, build_detector, build_reid, build_tracker
    from pia_tracking import CameraWorker, GlobalIDService          # several cameras
"""

from .config import load_config
from .detection import build_detector
from .fusion import CameraWorker, GlobalIDService, Identity, TrackAccumulator, Tracklet
from .reid import ReIDBackend, build_reid
from .schemas import Detection, Track
from .tracking import BoostTrackTracker, FrameResult, Tracker, TrackingPipeline, build_tracker

__all__ = [
    "BoostTrackTracker",
    "CameraWorker",
    "Detection",
    "FrameResult",
    "GlobalIDService",
    "Identity",
    "ReIDBackend",
    "Track",
    "TrackAccumulator",
    "Tracker",
    "TrackingPipeline",
    "Tracklet",
    "build_detector",
    "build_reid",
    "build_tracker",
    "load_config",
]
__version__ = "0.2.0"
