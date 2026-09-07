"""Shared by both runners: options, per-clip output sinks, the summary header."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..camera import VideoSource
from ..schemas import Track
from ..tracking import FrameResult
from ..utils import MOTWriter, VideoWriter, detections_to_mot_rows, draw_tracks


@dataclass(frozen=True)
class RunOptions:
    max_frames: int | None = None  # cap per video
    render_video: bool = True
    show_conf: bool = False  # confidence in the box labels
    dump_detections: bool = False  # also write raw pre-tracking detections
    # multi only
    checkpoints: bool = True  # mid-track id assignment after checkpoint_frames
    final_labels: bool = False  # re-render the MP4 with the final ids after tracking


LabelFn = Callable[[Track], str]
ColorFn = Callable[[Track], tuple[int, int, int]]


class ClipSinks:
    """Everything one clip writes: MOT rows, the annotated MP4, raw detections."""

    def __init__(self, source: VideoSource, out_dir: Path, opts: RunOptions, *, render_video: bool) -> None:
        self.stem = source.camera_id
        self.mot = MOTWriter()
        self.video = VideoWriter(out_dir / f"{self.stem}.mp4", source.fps) if render_video else None
        self._dump = opts.dump_detections
        self._show_conf = opts.show_conf
        self._raw_rows: list[str] = []

    def add(
        self,
        frame_idx: int,
        image: np.ndarray,
        result: FrameResult,
        *,
        label_fn: LabelFn | None = None,
        color_fn: ColorFn | None = None,
    ) -> None:
        self.mot.add(frame_idx, result.tracks)
        if self._dump:
            self._raw_rows += detections_to_mot_rows(frame_idx, result.detections)
        if self.video is not None:
            self.video.write(
                draw_tracks(
                    image, result.tracks, show_conf=self._show_conf, label=self.stem,
                    label_fn=label_fn, color_fn=color_fn,
                )
            )

    def close_video(self) -> None:
        if self.video is not None:
            self.video.close()

    def write_text(self, preds_dir: Path, *, global_id_map: dict[int, int] | None = None) -> None:
        """``<stem>.txt`` with local ids; with ``global_id_map`` also
        ``<stem>_global.txt`` (unlabelled tracks omitted); raw detections if asked."""
        self.mot.write(preds_dir / f"{self.stem}.txt")
        if global_id_map is not None:
            self.mot.write(preds_dir / f"{self.stem}_global.txt", id_map=global_id_map, drop_unmapped=True)
        if self._dump:
            preds_dir.mkdir(parents=True, exist_ok=True)
            (preds_dir / f"{self.stem}_dets.txt").write_text("".join(self._raw_rows))


def summary_header(
    *, mode: str, config: dict[str, Any], config_path: Path, device: str | None, reid: Any | None, opts: RunOptions
) -> dict[str, Any]:
    return {
        "mode": mode,
        "config_path": str(config_path),
        "detector": config["detector"].get("model"),
        "detector_conf": config["detector"].get("conf"),
        "detector_imgsz": config["detector"].get("imgsz"),
        "tracker": config["tracker"].get("type"),
        "tracker_params": config["tracker"].get("params", {}),
        "reid": config.get("reid", {}).get("person") if reid else None,
        "device": device or config["detector"].get("device"),
        "rendered_video": opts.render_video,
    }
