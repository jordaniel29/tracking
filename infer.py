"""Person tracking over video files — several cameras with shared global ids
(the default), or each clip on its own.

    python infer.py --videos-dir assets/data/03_scenarios/scenario_01 --out runs/s01   # multi (default)
    python infer.py --video cam8.mp4 cam9.mp4 --out runs/pair --device cuda:0          # multi, explicit files
    python infer.py --mode single --videos-dir assets/data --out runs/demo             # each clip alone
    python infer.py --mode single --video clip.mp4 --out runs/demo --no-reid

--mode multi   Every video is one camera and the cameras were recorded together (frame k
               of each is the same instant). The per-camera pipeline runs on all of them
               at once and one GlobalIDService links the tracks: the same person carries
               the same G-<n> on every camera. Also valid for ONE video — re-entries on
               that camera are then linked.
--mode single  Each clip is tracked on its own with per-clip local ids. The `global_id:`
               config block is ignored.
--mode render  No models: draw an EXISTING run's predictions (--out is that run's directory,
               --videos-dir/--video its source videos) onto the frames and write the MP4s —
               e.g. for a run made with --no-video. Multi-camera runs get G-<gid> labels
               from global_ids.json, single-camera runs local ids.

    python infer.py --mode render --videos-dir assets/data/03_scenarios/scenario_01 --out runs/compare/trace_ft/scenario_01

Files matching --exclude (default "grid_*", a composite view) are skipped in every mode.

Writes:
    <out>/<stem>.mp4               annotated video. multi: G-<gid> label, one colour per global
                                   id on every camera, grey + local id until the id is known —
                                   labels are what was known AT that frame; --final-labels
                                   re-renders with the final ids. single: id=<n>, colour per id.
    <out>/preds/<stem>.txt         MOTChallenge rows (frame,id,x,y,w,h,conf,-1,-1,-1), local id
    <out>/preds/<stem>_global.txt  the same rows with the global id (multi; unlabelled tracks omitted)
    <out>/preds/<stem>_dets.txt    raw pre-tracking detections (--show-all-dets)
    <out>/global_ids.json          identities → (camera, local id) members; local→global map (multi)
    <out>/run_summary.json         config used + per-clip throughput (+ identity counts in multi)

GPU selection: use `--device cuda:N`. Do NOT rely on a shell `CUDA_VISIBLE_DEVICES` —
Ultralytics rewrites that variable internally when it parses the device string.

The work lives in the package: pia_tracking.runners (run_single / run_multi). This file
only parses arguments and builds the models.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure local 'src' is available if the package isn't installed natively.
try:
    import pia_tracking  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pia_tracking import build_detector, build_reid, load_config
from pia_tracking.camera import DEFAULT_EXCLUDE, discover_videos
from pia_tracking.runners import RunOptions, run_multi, run_render, run_single

logger = logging.getLogger("infer")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description="Person detection + ReID + tracking — multi-camera with shared global ids, or per clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["multi", "single", "render"], default="multi",
        help="multi: every video is a camera, tracks linked across cameras by global id. "
        "single: each clip on its own with local ids. "
        "render: no models — draw an existing run's predictions (in --out) onto the videos.",
    )
    _add_io_args(parser)
    _add_mode_args(parser)

    args = parser.parse_args(argv)
    if args.mode == "multi" and args.no_reid:
        parser.error("--no-reid is single-mode only: global ids link on appearance")
    if args.mode != "multi" and (args.no_checkpoints or args.final_labels):
        parser.error("--no-checkpoints / --final-labels apply to --mode multi only")
    if args.mode == "render" and (args.no_reid or args.no_video or args.show_all_dets):
        parser.error("--mode render only draws: --no-reid / --no-video / --show-all-dets do not apply")
    return args


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, nargs="+", help="Video file(s).")
    src.add_argument("--videos-dir", type=Path, help="Directory containing the videos.")
    parser.add_argument(
        "--exclude", nargs="*", default=list(DEFAULT_EXCLUDE), metavar="GLOB",
        help="Filename patterns to skip (e.g. a composite grid view).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--config", type=Path, default=Path("config/tracking.yaml"))
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0. Overrides config.")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames processed per video.")
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 rendering; write MOT files only.")
    parser.add_argument("--show-conf", action="store_true", help="Append confidence scores to ID labels.")
    parser.add_argument("--show-all-dets", action="store_true", help="Write raw pre-tracking detections to disk.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _add_mode_args(parser: argparse.ArgumentParser) -> None:
    single = parser.add_argument_group("single mode")
    single.add_argument(
        "--no-reid", action="store_true",
        help="Track on geometry alone (faster, more ID switches). single only.",
    )
    multi = parser.add_argument_group("multi mode")
    multi.add_argument(
        "--no-checkpoints", action="store_true",
        help="Assign ids only when a track ends (no mid-track checkpoint). multi only.",
    )
    multi.add_argument(
        "--final-labels", action="store_true",
        help="Render the MP4s after tracking with the final (revised) global ids, so they match "
        "preds/<cam>_global.txt. Default: labels as known at each frame, like a live view. multi only.",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        videos = discover_videos(videos=args.video, videos_dir=args.videos_dir, exclude=args.exclude)
        if args.mode == "render":
            logger.info("start mode=render videos=%d run=%s", len(videos), args.out)
            return run_render(
                videos, out_dir=args.out, opts=RunOptions(max_frames=args.max_frames, show_conf=args.show_conf)
            )

        config = load_config(args.config, args.device)
        args.out.mkdir(parents=True, exist_ok=True)
        logger.info("start mode=%s videos=%d config=%s", args.mode, len(videos), args.config)

        # Models are built once and shared by every clip / camera; only the
        # tracker is per camera (it owns the cross-frame state).
        detector = build_detector(config["detector"])
        reid = None if args.no_reid else build_reid(config.get("reid"))
        logger.info(
            "models_ready detector=%s reid=%s tracker=%s",
            config["detector"].get("model"), reid is not None, config["tracker"].get("type"),
        )

        opts = RunOptions(
            max_frames=args.max_frames,
            render_video=not args.no_video,
            show_conf=args.show_conf,
            dump_detections=args.show_all_dets,
            checkpoints=not args.no_checkpoints,
            final_labels=args.final_labels,
        )
        run = run_multi if args.mode == "multi" else run_single
        return run(
            videos, config=config, config_path=args.config, device=args.device,
            detector=detector, reid=reid, out_dir=args.out, opts=opts,
        )
    except (ValueError, FileNotFoundError, NotADirectoryError) as e:
        raise SystemExit(f"error: {e}") from e


if __name__ == "__main__":
    sys.exit(main())
