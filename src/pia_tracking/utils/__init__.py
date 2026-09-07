"""Shared helpers: images, drawing, output files.

    image.py      ``crop_bgr`` — clamped person crops for ReID
    visualize.py  box + label drawing, per-id colours (local and global)
    writers.py    MOTChallenge and MP4 writers, JSON dump
"""

from .image import MIN_REID_CROP_PX, crop_bgr
from .visualize import color_for_global_id, color_for_id, draw_tracks, global_id_color, global_id_label
from .writers import MOTWriter, VideoWriter, detections_to_mot_rows, write_json

__all__ = [
    "MIN_REID_CROP_PX",
    "MOTWriter",
    "VideoWriter",
    "color_for_global_id",
    "color_for_id",
    "crop_bgr",
    "detections_to_mot_rows",
    "draw_tracks",
    "global_id_color",
    "global_id_label",
    "write_json",
]
