"""Multi mode: all cameras at once, one Global ID service shared by all of them.

Frames are interleaved across cameras (``camera.round_robin``), each camera has
its own ``CameraWorker`` (tracker + accumulators), and a single
``GlobalIDService`` links the tracks: the same person carries the same G-<n>
on every camera.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..camera import Exhausted, VideoSource, open_source, round_robin
from ..fusion import CameraWorker, GlobalIDService
from ..fusion.worker import DEFAULT_CHECKPOINT_FRAMES, DEFAULT_MIN_FRAMES_BEFORE_ID_ASSIGN
from ..tracking import TrackingPipeline, build_tracker
from ..utils import global_id_color, global_id_label, write_json
from .common import ClipSinks, RunOptions, summary_header
from .render import render_video

logger = logging.getLogger("pia_tracking.runners.multi")

PROGRESS_EVERY = 300  # frames per camera between progress log lines


@dataclass
class _Camera:
    source: VideoSource
    worker: CameraWorker
    sinks: ClipSinks
    elapsed: float = 0.0  # seconds spent in process_frame


def run_multi(
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
    if reid is None:
        raise ValueError("multi mode needs the `reid:` config block — global ids link on appearance")
    gid_cfg = _global_id_config(config, config_path, opts)
    service = GlobalIDService.from_config(gid_cfg)
    cams = [_open_camera(v, config, gid_cfg, detector, reid, service, out_dir, opts) for v in videos]

    wall = _process(cams, opts)

    if opts.final_labels and opts.render_video:
        for cam in cams:
            logger.info("render_final %s", cam.source.camera_id)
            _render_final(cam, out_dir, opts)

    per_camera = _write_camera_outputs(cams, out_dir)
    gid_summary = service.summary()
    write_json(
        out_dir / "global_ids.json",
        {
            **gid_summary,
            "local_to_global": {c.source.camera_id: _str_keys(c.worker.gid_map) for c in cams},
            "unlabelled_tracks": {c.source.camera_id: c.worker.unassigned_track_ids for c in cams},
        },
    )
    summary = {
        **summary_header(mode="multi", config=config, config_path=config_path, device=device, reid=reid, opts=opts),
        **_run_summary(cams, gid_cfg, gid_summary, per_camera, wall, opts),
    }
    write_json(out_dir / "run_summary.json", summary)
    logger.info(
        "done cameras=%d frames=%d wall=%.0fs identities=%d multi_camera=%d minted=%d matched=%d revised=%d out=%s",
        len(cams), summary["total_frames"], wall, gid_summary["n_identities"], gid_summary["n_multi_camera"],
        gid_summary["minted"], gid_summary["matched"], gid_summary["revised"], out_dir,
    )
    return 0


# ── setup ───────────────────────────────────────────────────────────────────

def _global_id_config(config: dict[str, Any], config_path: Path, opts: RunOptions) -> dict[str, Any]:
    gid_cfg = dict(config.get("global_id") or {})
    if not gid_cfg:
        logger.warning("no `global_id:` block in %s — using defaults", config_path)
    gid_cfg.setdefault("checkpoint_frames", DEFAULT_CHECKPOINT_FRAMES)
    gid_cfg.setdefault("min_frames_before_id_assign", DEFAULT_MIN_FRAMES_BEFORE_ID_ASSIGN)
    if not opts.checkpoints:
        gid_cfg["checkpoint_frames"] = 0
    logger.info(
        "global_id similarity_threshold=%s reidentify_within_sec=%s checkpoint_frames=%s min_frames=%s",
        gid_cfg.get("similarity_threshold", "default"), gid_cfg.get("reidentify_within_sec", "default"),
        gid_cfg["checkpoint_frames"], gid_cfg["min_frames_before_id_assign"],
    )
    return gid_cfg


def _open_camera(
    video: Path,
    config: dict[str, Any],
    gid_cfg: dict[str, Any],
    detector: Any,
    reid: Any,
    service: GlobalIDService,
    out_dir: Path,
    opts: RunOptions,
) -> _Camera:
    source = open_source(video)
    # Fresh tracker per camera; detector and ReID are shared.
    pipeline = TrackingPipeline(
        detector=detector, tracker=build_tracker(config["tracker"], reid), reid=reid, camera_id=source.camera_id
    )
    worker = CameraWorker(
        camera_id=source.camera_id,
        pipeline=pipeline,
        reid=reid,
        global_id=service,
        checkpoint_frames=gid_cfg["checkpoint_frames"],
        min_frames_before_id_assign=gid_cfg["min_frames_before_id_assign"],
    )
    # With final_labels the MP4 is rendered once, after tracking (see _render_final).
    sinks = ClipSinks(source, out_dir, opts, render_video=opts.render_video and not opts.final_labels)
    logger.info("camera %s fps=%.2f", source.camera_id, source.fps)
    return _Camera(source=source, worker=worker, sinks=sinks)


# ── the run ─────────────────────────────────────────────────────────────────

def _process(cams: list[_Camera], opts: RunOptions) -> float:
    """Interleave the cameras frame by frame; returns wall seconds."""
    by_source = {id(c.source): c for c in cams}
    start = time.perf_counter()
    for event in round_robin([c.source for c in cams], max_frames=opts.max_frames):
        cam = by_source[id(event.source)]
        if isinstance(event, Exhausted):
            cam.worker.finish(event.ts)
            cam.sinks.close_video()
            logger.info("camera_done %s frames=%d", cam.source.camera_id, cam.source.frames_read)
            continue
        t0 = time.perf_counter()
        result = cam.worker.process_frame(event.image, event.frame_idx, event.ts)
        cam.elapsed += time.perf_counter() - t0
        cam.sinks.add(event.frame_idx, event.image, result, label_fn=global_id_label, color_fn=global_id_color)
        if (event.frame_idx + 1) % PROGRESS_EVERY == 0:
            logger.info(
                "progress %s frame=%d fps=%.1f", cam.source.camera_id, event.frame_idx + 1,
                (event.frame_idx + 1) / cam.elapsed if cam.elapsed else 0.0,
            )
    return time.perf_counter() - start


def _render_final(cam: _Camera, out_dir: Path, opts: RunOptions) -> None:
    """Second pass over the source video with the FINAL global ids (after every
    loss-time revision), so the MP4 agrees with preds/<cam>_global.txt.
    Decode + encode only — no models. Same drawing as ``--mode render``."""
    source = open_source(cam.source.path)
    try:
        render_video(
            source, cam.sinks.mot.rows_by_frame(), cam.worker.gid_map, out_dir / f"{source.camera_id}.mp4",
            show_conf=opts.show_conf, max_frames=cam.source.frames_read,
        )
    finally:
        source.close()


# ── outputs ─────────────────────────────────────────────────────────────────

def _write_camera_outputs(cams: list[_Camera], out_dir: Path) -> list[dict[str, Any]]:
    """MOT files (local + global ids, back-annotated) and per-camera stats."""
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for cam in cams:
        cam.sinks.write_text(preds_dir, global_id_map=cam.worker.gid_map)
        stats.append(_camera_stats(cam))
    return stats


def _camera_stats(cam: _Camera) -> dict[str, Any]:
    ps = cam.worker.pipeline_stats
    ws = cam.worker.stats
    fps = round(ps.frames / cam.elapsed, 2) if cam.elapsed > 0 else 0.0
    with_gid = len(cam.worker.gid_map)
    unlabelled = len(cam.worker.unassigned_track_ids)
    logger.info(
        "camera_summary %s frames=%d dets=%d rows=%d local_ids=%d with_gid=%d unlabelled=%d "
        "checkpoint=%d loss=%d too_short=%d fallback_embeds=%d fps=%.1f",
        cam.source.camera_id, ps.frames, ps.detections, ps.track_rows, len(ps.track_ids), with_gid, unlabelled,
        ws.checkpoint_assigns, ws.loss_assigns, ws.too_short, ws.fallback_embeds, fps,
    )
    return {
        "video": cam.source.path.name,
        "camera_id": cam.source.camera_id,
        "frames": ps.frames,
        "detections": ps.detections,
        "track_rows": ps.track_rows,
        "track_ids": len(ps.track_ids),
        "tracks_with_global_id": with_gid,
        "tracks_unlabelled": unlabelled,
        "detection_coverage": round(ps.coverage, 4),
        "checkpoint_assigns": ws.checkpoint_assigns,
        "loss_assigns": ws.loss_assigns,
        "too_short": ws.too_short,
        "fallback_embeds": ws.fallback_embeds,
        "elapsed_sec": round(cam.elapsed, 2),
        "fps": fps,
    }


def _run_summary(
    cams: list[_Camera],
    gid_cfg: dict[str, Any],
    gid_summary: dict[str, Any],
    per_camera: list[dict[str, Any]],
    wall: float,
    opts: RunOptions,
) -> dict[str, Any]:
    total_frames = sum(c["frames"] for c in per_camera)
    return {
        "video_labels": "final" if opts.final_labels else "live",
        "global_id": {
            **{k: gid_summary[k] for k in ("similarity_threshold", "reidentify_within_sec", "revise_at_loss")},
            "checkpoint_frames": gid_cfg["checkpoint_frames"],
            "min_frames_before_id_assign": gid_cfg["min_frames_before_id_assign"],
        },
        "cameras": len(cams),
        "total_frames": total_frames,
        "wall_sec": round(wall, 2),
        "mean_fps_all_cameras": round(total_frames / wall, 2) if wall > 0 else None,
        "identities": gid_summary["n_identities"],
        "identities_multi_camera": gid_summary["n_multi_camera"],
        **{k: gid_summary[k] for k in ("assign_calls", "minted", "matched", "accumulated", "revised")},
        "per_camera": per_camera,
    }


def _str_keys(d: dict[int, int]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(d.items())}
