"""Config block → tracker instance."""

from __future__ import annotations

from typing import Any

from .base import Tracker


def build_tracker(cfg: dict[str, Any], reid: Any | None) -> Tracker:
    """The ``tracker:`` block. One instance per camera — the tracker owns all
    cross-frame state; detector and ReID are shared."""
    tracker_type = cfg.get("type")
    if tracker_type != "boost_track":
        raise ValueError(
            f"unsupported tracker '{tracker_type}'; this package ships with 'boost_track' (BoostTrack++) only"
        )
    from .boosttrack import BoostTrackTracker

    return BoostTrackTracker(reid=reid, **(cfg.get("params") or {}))
