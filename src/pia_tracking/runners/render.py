"""Render mode: draw an existing run's tracks onto the source videos — no models.

Reads ``<out>/preds/<stem>.txt`` (local ids) and, for a multi-camera run,
``<out>/global_ids.json`` (local → global map), and writes ``<out>/<stem>.mp4``
with exactly the labels and colours ``--final-labels`` produces. Use it to get
videos for a run made with ``--no-video``, or to re-render a run later.
``_render_final`` in multi mode goes through the same function.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from ..camera import VideoSource, open_source
from ..schemas import Detection, Track
from ..utils import VideoWriter, draw_tracks, global_id_color, global_id_label
from .common import RunOptions

logger = logging.getLogger("pia_tracking.runners.render")

MOTRow = tuple[int, float, float, float, float, float]  # track_id, x, y, w, h, conf


def load_global_id_map(out_dir: Path) -> dict[str, dict[int, int]] | None:
    """``{camera_id: {local id: global id}}`` from a multi-camera run's
    ``global_ids.json``, or None for a single-camera run (no such file)."""
    path = out_dir / "global_ids.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text()).get("local_to_global") or {}
    return {cam: {int(k): v for k, v in m.items()} for cam, m in raw.items()}


def read_mot(path: Path) -> dict[int, list[MOTRow]]:
    """``frame_idx → rows`` from a MOTChallenge file written by MOTWriter."""
    rows: dict[int, list[MOTRow]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        f, tid, x, y, w, h, conf, *_ = line.split(",")
        rows.setdefault(int(f), []).append((int(tid), float(x), float(y), float(w), float(h), float(conf)))
    return rows


def track_from_row(camera_id: str, frame_idx: int, ts: datetime, row: MOTRow, gid_map: dict[int, int]) -> Track:
    tid, x, y, w, h, conf = row
    return Track(
        track_id=tid,
        camera_id=camera_id,
        detection=Detection(bbox=(x, y, x + w, y + h), class_id=0, confidence=conf),
        frame_idx=frame_idx,
        timestamp=ts,
        global_id=gid_map.get(tid),
    )


def render_video(
    source: VideoSource,
    rows: dict[int, list[MOTRow]],
    gid_map: dict[int, int] | None,
    out_path: Path,
    *,
    show_conf: bool = False,
    max_frames: int | None = None,
) -> int:
    """Draw ``rows`` onto the source frames and write ``out_path``. Returns the
    frames written. ``gid_map`` None → single-camera labels (``id=<n>``, colour
    per local id); a dict → ``G-<gid>`` coloured by global id, grey when unmapped."""
    writer = VideoWriter(out_path, source.fps)
    label_fn, color_fn = (global_id_label, global_id_color) if gid_map is not None else (None, None)
    written = 0
    try:
        while (item := source.read()) is not None:
            frame_idx, image = item
            if max_frames is not None and frame_idx >= max_frames:
                break
            tracks = [
                track_from_row(source.camera_id, frame_idx, source.ts(frame_idx), r, gid_map or {})
                for r in rows.get(frame_idx, [])
            ]
            writer.write(
                draw_tracks(
                    image, tracks, show_conf=show_conf, label=source.camera_id, label_fn=label_fn, color_fn=color_fn
                )
            )
            written += 1
    finally:
        writer.close()
    return written


def run_render(videos: list[Path], *, out_dir: Path, opts: RunOptions) -> int:
    """Render every video whose predictions exist in ``out_dir/preds``."""
    preds_dir = out_dir / "preds"
    if not preds_dir.is_dir():
        raise FileNotFoundError(f"{preds_dir} not found — --out must be an existing run directory")
    local_to_global = load_global_id_map(out_dir)

    skipped = 0
    for video in videos:
        mot = preds_dir / f"{video.stem}.txt"
        if not mot.is_file():
            logger.warning("no predictions for %s (%s missing) — skipped", video.name, mot.name)
            skipped += 1
            continue
        gid_map = None if local_to_global is None else local_to_global.get(video.stem, {})
        source = open_source(video)
        try:
            n = render_video(
                source, read_mot(mot), gid_map, out_dir / f"{video.stem}.mp4",
                show_conf=opts.show_conf, max_frames=opts.max_frames,
            )
        finally:
            source.close()
        logger.info("rendered %s frames=%d labels=%s", video.stem, n, "global" if gid_map is not None else "local")

    logger.info("done rendered=%d skipped=%d out=%s", len(videos) - skipped, skipped, out_dir)
    return 0 if skipped == 0 else 1
