"""YOLO26 detector wrapper.

Runs person/vehicle detection per frame via ``ultralytics.YOLO`` with
class-group routing. Target classes are configured via the ``target_classes``
YAML key using COCO class ids — the dict maps logical names (``"person"``,
``"vehicle"``) to lists of COCO class ids so the downstream pipeline can route
crops to the right ReID model.

Usage:
    from piaspace_yolo26 import YOLO26Detector, Detection

    det = YOLO26Detector(cfg["detector"])
    detections = det.detect_image(frame_bgr)  # or det.detect(frame_bgr)

TRT engine auto-provisioning goes through ``piaspace_trt_runtime``; the
package is self-contained otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_name: str  # logical class: "person" | "vehicle"
    class_id: int  # raw COCO class id
    bbox: tuple[float, float, float, float]  # xyxy, absolute pixel coords
    conf: float


class YOLO26Detector:
    """YOLO26 detector with class-group filtering.

    The YAML ``target_classes`` maps logical names (person/vehicle) to lists of
    COCO class ids, so the pipeline can route crops to the right ReID model.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cuda:0")
        self.conf = float(config.get("conf", 0.35))
        self.iou = float(config.get("iou", 0.5))
        self.imgsz = int(config.get("imgsz", 640))
        self.min_box_size = int(config.get("min_box_size", 0))

        target_classes: dict = config.get("target_classes", {}) or {}
        # An EMPTY id list for a logical group ("person"/"vehicle") means
        # "use this backend's canonical COCO ids" — YOLO is 0-indexed (person=0),
        # so callers request a logical class without knowing the convention.
        # Explicit ids are still honored unchanged.
        default_group_ids = {"person": [0], "vehicle": [1, 2, 3, 5, 7]}
        # Flatten to {coco_id -> logical_name} for fast lookup, and keep raw set of ids for YOLO's classes arg.
        self._id_to_group: dict[int, str] = {}
        for group_name, ids in target_classes.items():
            resolved = list(ids) if ids else default_group_ids.get(group_name, [])
            for cid in resolved:
                self._id_to_group[int(cid)] = group_name
        self._target_ids: list[int] = sorted(self._id_to_group.keys())

        self._model = self._load_model()

    def _load_model(self):
        from ultralytics import YOLO

        model_name = self.config.get("model", "yolo26n.pt")
        weights_dir = Path(self.config.get("weights_dir", "./weights"))
        weights_dir.mkdir(parents=True, exist_ok=True)

        # For .engine targets, auto-provision via HF download + TRT build if
        # the engine isn't on disk. .pt falls back to Ultralytics' own
        # asset downloader.
        is_pt = Path(model_name).suffix.lower() in (".pt", ".pth")
        local_path = weights_dir / model_name
        if not is_pt and not local_path.exists():
            try:
                from .engine import ensure_engine_by_filename

                local_path = ensure_engine_by_filename(model_name, weights_dir=weights_dir)
            except Exception as e:
                logger.error("YOLO engine auto-provision failed: %s", e)
                # Fall through; Ultralytics will error with a clearer message.

        path_to_load = str(local_path) if local_path.exists() else model_name

        if is_pt:
            model = YOLO(path_to_load)
            model.to(self.device)
        else:
            model = YOLO(path_to_load, task="detect")
        logger.info(
            f"Loaded detector '{model_name}' on {self.device} "
            f"(format={'pytorch' if is_pt else Path(model_name).suffix.lstrip('.')}, "
            f"target coco ids={self._target_ids}, conf={self.conf}, imgsz={self.imgsz})"
        )
        return model

    @property
    def target_groups(self) -> list[str]:
        """Distinct logical class groups, e.g. ['person', 'vehicle']."""
        return sorted(set(self._id_to_group.values()))

    def detect_image(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Alias for :meth:`detect` — preferred name for new call sites."""
        return self.detect(frame_bgr)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame.

        Returns only detections whose class_id is in target_classes and whose
        box's min side is >= min_box_size.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        res = self._model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self._target_ids if self._target_ids else None,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        xyxy = res.boxes.xyxy.cpu().numpy()  # (N, 4)
        confs = res.boxes.conf.cpu().numpy()  # (N,)
        clses = res.boxes.cls.cpu().numpy().astype(int)  # (N,)

        for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, clses, strict=False):
            group = self._id_to_group.get(int(cid))
            if group is None:
                continue
            w, h = x2 - x1, y2 - y1
            if min(w, h) < self.min_box_size:
                continue
            detections.append(
                Detection(
                    class_name=group,
                    class_id=int(cid),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    conf=float(c),
                )
            )
        return detections

    def detect_batch(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        """Batched detection. Returns one detection list per frame."""
        if not frames_bgr:
            return []
        results = self._model.predict(
            source=frames_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self._target_ids if self._target_ids else None,
            device=self.device,
            verbose=False,
        )

        all_dets: list[list[Detection]] = []
        for res in results:
            dets: list[Detection] = []
            if res.boxes is None or len(res.boxes) == 0:
                all_dets.append(dets)
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clses = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, clses, strict=False):
                group = self._id_to_group.get(int(cid))
                if group is None:
                    continue
                w, h = x2 - x1, y2 - y1
                if min(w, h) < self.min_box_size:
                    continue
                dets.append(
                    Detection(
                        class_name=group,
                        class_id=int(cid),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        conf=float(c),
                    )
                )
            all_dets.append(dets)
        return all_dets
