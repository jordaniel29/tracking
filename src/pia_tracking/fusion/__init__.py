"""Cross-camera identity (global ids).

    tracklet.py   ``Tracklet`` — one track's evidence segment (mean embedding, time
                  span, co-visible ids) — and ``TrackAccumulator``, which builds it
    global_id.py    ``GlobalIDService`` — Tracklet → global_id, shared by every camera
    percam_norm.py  ``PerCameraNormalizer`` — per-camera mean-centering of embeddings
                    before matching (removes shared scene bias)
    worker.py       ``CameraWorker`` — drives one camera's TrackingPipeline, accumulates
                    its tracks' embeddings, and asks the service at checkpoint / loss
"""

from .global_id import GlobalIDService, GlobalIDStats, Identity
from .percam_norm import PerCameraNormalizer
from .tracklet import TrackAccumulator, Tracklet, l2_normalize
from .worker import CameraWorker, CameraWorkerStats

__all__ = [
    "CameraWorker",
    "CameraWorkerStats",
    "GlobalIDService",
    "GlobalIDStats",
    "Identity",
    "PerCameraNormalizer",
    "TrackAccumulator",
    "Tracklet",
    "l2_normalize",
]
