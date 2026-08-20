"""Single-camera person tracking over a video file or a directory of them.

    python infer.py --video sample.mp4 --out runs/demo
    python infer.py --videos-dir videos/ --out runs/demo --config config/tracking.yaml

Writes per clip:
    <out>/<stem>.mp4              annotated video (box + id label, colour per id)
    <out>/preds/<stem>.txt        MOTChallenge rows: frame,id,x,y,w,h,conf,-1,-1,-1
    <out>/preds/<stem>_dets.txt   raw pre-tracking detections (with --show-all-dets)
    <out>/run_summary.json        config actually used + per-clip throughput

GPU selection: use `--device cuda:N`. Do NOT rely on a shell `CUDA_VISIBLE_DEVICES` —
Ultralytics rewrites that variable internally when it parses the device string.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure local 'src' is available if the package isn't installed natively.
try:
    import pia_tracking  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2
import yaml

from pia_tracking.pipeline import TrackingPipeline
from pia_tracking.visualize import MOTWriter, VideoWriter, draw_tracks

logger = logging.getLogger("infer")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".m4v")


# ── Configuration & Initialization ──────────────────────────────────────────

def load_config(path: Path, device: str | None) -> dict[str, Any]:
    """Loads the YAML config and overrides device allocations if requested."""
    config = yaml.safe_load(path.read_text())
    
    if "detector" not in config or "tracker" not in config:
        raise SystemExit(f"Error: Missing 'detector' or 'tracker' block in {path}")

    # Apply device overrides cleanly
    if device:
        config["detector"]["device"] = device
        
        reid_cfg = config.get("reid", {}).get("person", {})
        reid_backend = reid_cfg.get("backend")
        if reid_backend and reid_backend in reid_cfg:
            reid_cfg[reid_backend]["device"] = device

    return config


def build_detector(cfg: dict[str, Any]) -> Any:
    from piaspace_yolo26 import YOLO26Detector
    return YOLO26Detector(cfg)


def build_reid(reid_cfg: dict[str, Any]) -> Any | None:
    """Build the person ReID embedder, or return None if omitted."""
    person_cfg = (reid_cfg or {}).get("person")
    if not person_cfg:
        return None
        
    backend = person_cfg.get("backend", "clip_reid")
    if backend != "clip_reid":
        raise SystemExit(
            f"Unsupported ReID backend: '{backend}'. This package ships with "
            "'clip_reid' only. Remove the `reid:` block to track on geometry alone."
        )
        
    from piaspace_clip_reid import CLIPReIDEmbedder
    return CLIPReIDEmbedder(person_cfg["clip_reid"])


def build_tracker(cfg: dict[str, Any], reid: Any | None) -> Any:
    tracker_type = cfg.get("type")
    if tracker_type != "boost_track":
        raise SystemExit(
            f"Unsupported tracker: '{tracker_type}'. This package ships with "
            "'boost_track' (BoostTrack++) only."
        )
        
    from pia_tracking.tracker.boosttrack import BoostTrackTracker
    return BoostTrackTracker(reid=reid, **(cfg.get("params") or {}))


def discover_videos(args: argparse.Namespace) -> list[Path]:
    """Finds all valid video files based on CLI arguments."""
    if args.video:
        if not args.video.is_file():
            raise SystemExit(f"Video not found: {args.video}")
        return [args.video]
        
    videos = sorted(p for p in args.videos_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise SystemExit(f"No videos found in {args.videos_dir} matching {VIDEO_SUFFIXES}")
        
    return videos


# ── Core Inference Loop ─────────────────────────────────────────────────────

def run_clip(
    video: Path,
    *,
    detector: Any,
    reid: Any | None,
    tracker_cfg: dict[str, Any],
    out_dir: Path,
    max_frames: int | None,
    show_conf: bool,
    show_all_dets: bool,
    no_video: bool,
) -> dict[str, Any]:
    """Tracks a single clip, renders output files, and returns execution stats."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video stream: {video}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Fresh tracker state per clip. Models (Detector/ReID) are shared.
    pipeline = TrackingPipeline(
        detector=detector,
        tracker=build_tracker(tracker_cfg, reid),
        reid=reid,
        camera_id=video.stem,
    )
    
    mot_writer = MOTWriter()
    video_writer = None if no_video else VideoWriter(out_dir / f"{video.stem}.mp4", fps)
    raw_detections: list[str] = []

    start_time = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            success, frame = cap.read()
            if not success or (max_frames is not None and frame_idx >= max_frames):
                break
                
            result = pipeline.process_frame(frame, frame_idx)
            mot_writer.add(frame_idx, result.tracks)
            
            if show_all_dets:
                for d in result.detections:
                    x1, y1, x2, y2 = d.bbox
                    raw_detections.append(
                        f"{frame_idx},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                        f"{d.confidence:.4f},-1,-1,-1\n"
                    )
                    
            if video_writer is not None:
                annotated_frame = draw_tracks(frame, result.tracks, show_conf=show_conf, label=video.stem)
                video_writer.write(annotated_frame)
                
            frame_idx += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.close()

    elapsed = time.perf_counter() - start_time

    # Write textual outputs
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    mot_writer.write(preds_dir / f"{video.stem}.txt")
    
    if show_all_dets:
        (preds_dir / f"{video.stem}_dets.txt").write_text("".join(raw_detections))

    # Compile and log statistics
    stats = pipeline.stats
    fps_rate = round(stats.frames / elapsed, 2) if elapsed > 0 else 0.0
    
    logger.info(
        "clip_done %s frames=%d dets=%d rows=%d ids=%d coverage=%.1f%% fps=%.1f",
        video.name, stats.frames, stats.detections, stats.track_rows, 
        len(stats.track_ids), 100 * stats.coverage, fps_rate,
    )
    
    return {
        "video": video.name,
        "camera_id": video.stem,
        "frames": stats.frames,
        "detections": stats.detections,
        "track_rows": stats.track_rows,
        "track_ids": len(stats.track_ids),
        "detection_coverage": round(stats.coverage, 4),
        "elapsed_sec": round(elapsed, 2),
        "fps": fps_rate,
    }


# ── CLI & Main ──────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description="Single-camera person detection + ReID + tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Path to a single video file.")
    src.add_argument("--videos-dir", type=Path, help="Directory containing multiple videos.")
    
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--config", type=Path, default=Path("config/tracking.yaml"))
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0. Overrides config.")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames processed per clip.")
    parser.add_argument("--no-reid", action="store_true", help="Track on geometry alone (faster, more ID switches).")
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 rendering; write MOT files only.")
    parser.add_argument("--show-conf", action="store_true", help="Append confidence scores to ID labels.")
    parser.add_argument("--show-all-dets", action="store_true", help="Write raw pre-tracking detections to disk.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    config = load_config(args.config, args.device)
    videos = discover_videos(args)
    
    # Ensure root output directory exists
    args.out.mkdir(parents=True, exist_ok=True)

    logger.info("start videos=%d config=%s", len(videos), args.config)
    
    # Initialize models once to prevent reloading TRT engines per video
    detector = build_detector(config["detector"])
    reid = None if args.no_reid else build_reid(config.get("reid", {}))
    
    logger.info(
        "models_ready detector=%s reid=%s tracker=%s",
        config["detector"].get("model"), bool(reid), config["tracker"].get("type"),
    )

    clips_stats: list[dict[str, Any]] = []
    failed_clips = 0
    
    for i, video in enumerate(videos, 1):
        logger.info("clip (%d/%d) %s", i, len(videos), video.name)
        try:
            stats = run_clip(
                video,
                detector=detector,
                reid=reid,
                tracker_cfg=config["tracker"],
                out_dir=args.out,
                max_frames=args.max_frames,
                show_conf=args.show_conf,
                show_all_dets=args.show_all_dets,
                no_video=args.no_video,
            )
            clips_stats.append(stats)
        except Exception:
            logger.exception("clip_failed %s", video.name)
            failed_clips += 1

    # Generate Run Summary
    total_frames = sum(c["frames"] for c in clips_stats)
    total_sec = sum(c["elapsed_sec"] for c in clips_stats)
    mean_fps = round(total_frames / total_sec, 2) if total_sec > 0 else None
    
    summary = {
        "config_path": str(args.config),
        "detector": config["detector"].get("model"),
        "detector_conf": config["detector"].get("conf"),
        "detector_imgsz": config["detector"].get("imgsz"),
        "tracker": config["tracker"].get("type"),
        "tracker_params": config["tracker"].get("params", {}),
        "reid": (config.get("reid", {}).get("person") if reid else None),
        "device": args.device or config["detector"].get("device"),
        "rendered_video": not args.no_video,
        "clips_ok": len(clips_stats),
        "clips_failed": failed_clips,
        "total_frames": total_frames,
        "mean_fps": mean_fps,
        "clips": clips_stats,
    }
    
    (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    
    logger.info(
        "done ok=%d failed=%d frames=%d mean_fps=%s out=%s",
        len(clips_stats), failed_clips, total_frames, mean_fps, args.out,
    )
    
    return 0 if failed_clips == 0 else 1


if __name__ == "__main__":
    sys.exit(main())