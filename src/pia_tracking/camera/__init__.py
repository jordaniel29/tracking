"""Video sources and frame sync.

    sources.py    which files are cameras (discovery + exclude patterns), opening
                  them, and the shared synthetic clock
    scheduler.py  round-robin interleaving of several cameras — frame k of every
                  camera, then k+1 — so cross-camera decisions only see the past
"""

from .scheduler import Exhausted, Frame, round_robin
from .sources import DEFAULT_EXCLUDE, EPOCH, VIDEO_SUFFIXES, VideoSource, discover_videos, open_source

__all__ = [
    "DEFAULT_EXCLUDE",
    "EPOCH",
    "Exhausted",
    "Frame",
    "VIDEO_SUFFIXES",
    "VideoSource",
    "discover_videos",
    "open_source",
    "round_robin",
]
