"""Shared helpers: images, drawing, output files.

    image.py      ``crop_bgr`` — clamped person crops for ReID
    visualize.py  box + label drawing, per-id colours (local and global)
    mosaic.py     tile several cameras' frames into one grid image
    writers.py    MOTChallenge and MP4 writers, JSON dump
"""

from .image import MIN_REID_CROP_PX, crop_bgr
from .mosaic import DEFAULT_GRID_WIDTH, cell_size, compose, grid_shape
from .visualize import color_for_global_id, color_for_id, draw_tracks, global_id_color, global_id_label
from .writers import MOTWriter, VideoWriter, detections_to_mot_rows, write_json

__all__ = [
    "DEFAULT_GRID_WIDTH",
    "MIN_REID_CROP_PX",
    "MOTWriter",
    "VideoWriter",
    "cell_size",
    "color_for_global_id",
    "color_for_id",
    "compose",
    "crop_bgr",
    "detections_to_mot_rows",
    "draw_tracks",
    "global_id_color",
    "global_id_label",
    "grid_shape",
    "write_json",
]
