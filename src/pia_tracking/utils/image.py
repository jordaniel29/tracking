"""Image helpers."""

from __future__ import annotations

import numpy as np

# Minimum crop side, in pixels, worth handing to ReID. Below this the 256x128
# CLIP-ReID input is mostly interpolation and the embedding is noise, so such
# detections are tracked on geometry alone rather than poisoning appearance.
MIN_REID_CROP_PX = 8


def crop_bgr(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    """Clamped crop for one xyxy box, or None when it is degenerate.

    Detector boxes can run past the frame edge (a person half out of view), so
    clamping here rather than trusting the box avoids a zero-width slice that
    would make a whole ReID batch fail.
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 - x1 < MIN_REID_CROP_PX or y2 - y1 < MIN_REID_CROP_PX:
        return None
    return frame[y1:y2, x1:x2]
