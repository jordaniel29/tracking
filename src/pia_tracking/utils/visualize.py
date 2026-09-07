"""Drawing tracked boxes on frames, and the per-id colours.

Single-camera runs label ``id=<track_id>`` and colour by track id from a small
hashed palette, so an ID switch shows as a colour change on one person.

Multi-camera runs label ``G-<global_id>`` and colour by global id
(``global_id_label`` / ``global_id_color``): the same person is the same colour
on every camera, grey until the id is known.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ..schemas import Track

_FONT = cv2.FONT_HERSHEY_SIMPLEX
# Distinct, mid-saturation BGR palette. Avoids pure red/green so the boxes stay
# legible on both the washed-out and the very dark footage in CCTV sets.
_PALETTE = [
    (66, 135, 245), (245, 176, 66), (66, 245, 152), (245, 66, 197),
    (197, 245, 66), (66, 245, 245), (150, 66, 245), (245, 66, 90),
    (110, 200, 120), (200, 120, 200), (120, 200, 200), (240, 140, 80),
]
# A tracked box whose cross-camera identity is not known yet.
_UNASSIGNED_GREY = (160, 160, 160)


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Stable BGR colour for a local track id."""
    return _PALETTE[hash((track_id, 0x9E37)) % len(_PALETTE)]


def color_for_global_id(global_id: int) -> tuple[int, int, int]:
    """Vivid BGR colour per global id. Hue steps by 37 (coprime with 180), so
    consecutive ids land far apart on the hue wheel — sequential global ids get
    ~180 distinct colours, where the 12-entry palette would make two people
    share one and read as a single identity across cameras."""
    hue = (int(global_id) * 37) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 240]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def global_id_label(t: Track) -> str:
    """``G-<gid>`` once assigned, the local id until then."""
    return f"G-{t.global_id}" if t.global_id is not None else f"id={t.track_id}"


def global_id_color(t: Track) -> tuple[int, int, int]:
    return color_for_global_id(t.global_id) if t.global_id is not None else _UNASSIGNED_GREY


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    *,
    show_conf: bool = False,
    label: str | None = None,
    label_fn: Callable[[Track], str] | None = None,
    color_fn: Callable[[Track], tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw one frame's tracks. Returns a copy; the input is not mutated.

    ``label_fn`` / ``color_fn`` override the per-box text and colour (default:
    ``id=<track_id>`` coloured by track id). ``label`` is a frame-level tag
    (the camera name) drawn top-left.
    """
    out = frame.copy()
    for t in tracks:
        colour = color_fn(t) if color_fn is not None else color_for_id(t.track_id)
        text = label_fn(t) if label_fn is not None else f"id={t.track_id}"
        if show_conf:
            text += f" {t.detection.confidence:.2f}"
        _draw_box(out, t.detection.bbox, colour, text)
    if label:
        cv2.putText(out, label, (8, 22), _FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def _draw_box(out: np.ndarray, bbox: tuple[float, float, float, float], colour: tuple[int, int, int], text: str) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.5, 1)
    # Keep the label inside the frame when the box touches the top edge,
    # otherwise it is drawn at negative y and silently disappears.
    ty = max(th + 2, y1)
    cv2.rectangle(out, (x1, ty - th - 2), (x1 + tw + 2, ty), colour, -1)
    cv2.putText(out, text, (x1 + 1, ty - 2), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
