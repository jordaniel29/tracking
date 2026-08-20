"""Single-camera trackers.

``base.py`` holds the contracts — the ``Tracker`` ABC the pipeline calls and
the ``AppearanceTracker`` template appearance-capable trackers build on.
``boosttrack.py`` is the shipped implementation (BoostTrack++).
``_matching.py`` is private geometry/assignment math.
"""

from .base import AppearanceTracker, Tracker
from .boosttrack import BoostTrackTracker

__all__ = ["AppearanceTracker", "BoostTrackTracker", "Tracker"]
