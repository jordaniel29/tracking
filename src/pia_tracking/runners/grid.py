"""Grid mode: one mosaic video per run, all cameras tiled and time-aligned.

Reads an existing run's ``preds/`` (and ``global_ids.json`` for a multi-camera
run), draws the boxes on the source frames, and tiles every camera into one
``<out>/grid.mp4``. Cameras are read in lockstep on frame index — the
recordings are simultaneous, so column k of every cell is the same instant —
and a camera whose video ends early leaves a black cell.

No models: decode, draw, encode. Same drawing as ``--mode render``, so the
grid agrees cell-for-cell with the per-camera MP4s.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..camera import VideoSource, open_source
from ..utils import VideoWriter, cell_size, compose, draw_tracks, global_id_color, global_id_label, grid_shape
from .common import RunOptions
from .render import MOTRow, load_global_id_map, read_mot, track_from_row

logger = logging.getLogger("pia_tracking.runners.grid")


def run_grid(videos: list[Path], *, out_dir: Path, opts: RunOptions) -> int:
    """Write ``<out_dir>/grid.mp4`` from the run in ``out_dir``."""
    preds_dir = out_dir / "preds"
    if not preds_dir.is_dir():
        raise FileNotFoundError(f"{preds_dir} not found — --out must be an existing run directory")
    local_to_global = load_global_id_map(out_dir)

    usable = [v for v in videos if (preds_dir / f"{v.stem}.txt").is_file()]
    for v in videos:
        if v not in usable:
            logger.warning("no predictions for %s — left out of the grid", v.name)
    if not usable:
        raise FileNotFoundError(f"none of the {len(videos)} videos have predictions in {preds_dir}")

    sources = [open_source(v) for v in usable]
    try:
        return _write_grid(sources, out_dir, local_to_global, opts)
    finally:
        for s in sources:
            s.close()


def _write_grid(
    sources: list[VideoSource],
    out_dir: Path,
    local_to_global: dict[str, dict[int, int]] | None,
    opts: RunOptions,
) -> int:
    rows_per_cam = {s.camera_id: read_mot(out_dir / "preds" / f"{s.camera_id}.txt") for s in sources}
    n_frames = max(max(r, default=-1) for r in rows_per_cam.values()) + 1
    if opts.max_frames is not None:
        n_frames = min(n_frames, opts.max_frames)
    rows, cols = grid_shape(len(sources), cols=opts.grid_cols)
    cell = cell_size(*_probe_frame_size(sources[0]), cols, total_width=opts.grid_width)

    out_path = out_dir / "grid.mp4"
    writer = VideoWriter(out_path, sources[0].fps)
    logger.info(
        "grid %s cameras=%d layout=%dx%d cell=%dx%d frames=%d",
        out_path, len(sources), rows, cols, cell[0], cell[1], n_frames,
    )
    try:
        for _ in range(n_frames):
            writer.write(compose(
                [_cell(s, rows_per_cam[s.camera_id], local_to_global, opts, cell=cell) for s in sources],
                cell=cell, cols=cols,
            ))
    finally:
        writer.close()
    logger.info("done grid=%s cameras=%d frames=%d", out_path, len(sources), n_frames)
    return 0


def _cell(
    source: VideoSource,
    rows: dict[int, list[MOTRow]],
    local_to_global: dict[str, dict[int, int]] | None,
    opts: RunOptions,
    *,
    cell: tuple[int, int],
) -> np.ndarray | None:
    """This camera's annotated frame at cell size, or None once its video has ended.

    The frame is resized to the cell BEFORE drawing and the boxes are scaled to
    match, so label text and box strokes stay their normal size relative to the
    cell instead of being shrunk along with the image.
    """
    item = source.read()
    if item is None:
        return None
    idx, image = item
    h, w = image.shape[:2]
    cw, ch = cell
    if (w, h) != (cw, ch):
        image = cv2.resize(image, (cw, ch))
    sx, sy = cw / w, ch / h
    gid_map = None if local_to_global is None else local_to_global.get(source.camera_id, {})
    label_fn, color_fn = (global_id_label, global_id_color) if gid_map is not None else (None, None)
    ts = source.ts(idx)
    tracks = [
        track_from_row(source.camera_id, idx, ts, (tid, x * sx, y * sy, bw * sx, bh * sy, conf), gid_map or {})
        for tid, x, y, bw, bh, conf in rows.get(idx, [])
    ]
    return draw_tracks(
        image, tracks, show_conf=opts.show_conf, label=source.camera_id, label_fn=label_fn, color_fn=color_fn
    )


def _probe_frame_size(source: VideoSource) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(source.path))
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"cannot read frame size from {source.path}")
    return w, h
