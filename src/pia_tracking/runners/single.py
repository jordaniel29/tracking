"""Single mode: each clip tracked on its own, per-clip local ids."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ..camera import VideoSource, open_source
from ..tracking import RunStats, TrackingPipeline, build_tracker
from ..utils import write_json
from .common import ClipSinks, RunOptions, summary_header

logger = logging.getLogger("pia_tracking.runners.single")


def run_single(
    videos: list[Path],
    *,
    config: dict[str, Any],
    config_path: Path,
    device: str | None,
    detector: Any,
    reid: Any | None,
    out_dir: Path,
    opts: RunOptions,
) -> int:
    """Track every clip; a failing clip is logged and skipped. Returns the exit code."""
    clips: list[dict[str, Any]] = []
    failed = 0
    for i, video in enumerate(videos, 1):
        logger.info("clip (%d/%d) %s", i, len(videos), video.name)
        try:
            clips.append(track_clip(video, config=config, detector=detector, reid=reid, out_dir=out_dir, opts=opts))
        except Exception:  # noqa: BLE001 — one bad clip must not stop the batch
            logger.exception("clip_failed %s", video.name)
            failed += 1

    total_frames = sum(c["frames"] for c in clips)
    total_sec = sum(c["elapsed_sec"] for c in clips)
    mean_fps = round(total_frames / total_sec, 2) if total_sec > 0 else None
    summary = {
        **summary_header(mode="single", config=config, config_path=config_path, device=device, reid=reid, opts=opts),
        "clips_ok": len(clips),
        "clips_failed": failed,
        "total_frames": total_frames,
        "mean_fps": mean_fps,
        "clips": clips,
    }
    write_json(out_dir / "run_summary.json", summary)
    logger.info("done ok=%d failed=%d frames=%d mean_fps=%s out=%s", len(clips), failed, total_frames, mean_fps, out_dir)
    return 0 if failed == 0 else 1


def track_clip(
    video: Path, *, config: dict[str, Any], detector: Any, reid: Any | None, out_dir: Path, opts: RunOptions
) -> dict[str, Any]:
    """One clip end to end: fresh tracker, shared models; returns its stats."""
    source = open_source(video)
    pipeline = TrackingPipeline(
        detector=detector, tracker=build_tracker(config["tracker"], reid), reid=reid, camera_id=source.camera_id
    )
    sinks = ClipSinks(source, out_dir, opts, render_video=opts.render_video)

    start = time.perf_counter()
    try:
        _track_frames(source, pipeline, sinks, opts)
    finally:
        source.close()
        sinks.close_video()
    elapsed = time.perf_counter() - start

    sinks.write_text(out_dir / "preds")
    return _clip_stats(source, pipeline.stats, elapsed)


def _track_frames(source: VideoSource, pipeline: TrackingPipeline, sinks: ClipSinks, opts: RunOptions) -> None:
    while (item := source.read()) is not None:
        frame_idx, image = item
        if opts.max_frames is not None and frame_idx >= opts.max_frames:
            break
        sinks.add(frame_idx, image, pipeline.process_frame(image, frame_idx))


def _clip_stats(source: VideoSource, stats: RunStats, elapsed: float) -> dict[str, Any]:
    fps = round(stats.frames / elapsed, 2) if elapsed > 0 else 0.0
    logger.info(
        "clip_done %s frames=%d dets=%d rows=%d ids=%d coverage=%.1f%% fps=%.1f",
        source.path.name, stats.frames, stats.detections, stats.track_rows, len(stats.track_ids),
        100 * stats.coverage, fps,
    )
    return {
        "video": source.path.name,
        "camera_id": source.camera_id,
        "frames": stats.frames,
        "detections": stats.detections,
        "track_rows": stats.track_rows,
        "track_ids": len(stats.track_ids),
        "detection_coverage": round(stats.coverage, 4),
        "elapsed_sec": round(elapsed, 2),
        "fps": fps,
    }
