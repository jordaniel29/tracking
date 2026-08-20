"""BoostTrack++ adapter — boosting the similarity measure and detection confidence.

Reference (papers only — see below): Stanojević & Todorović,
"BoostTrack: boosting the similarity measure and detection confidence for
improved multiple object tracking", Machine Vision and Applications, 2024; and
"BoostTrack++: using tracklet information to detect more objects in multiple
object tracking", arXiv:2408.13003.

Several exact formulas (Mahalanobis→similarity conversion, the shape-similarity
transform, the tracklet-confidence term) are delegated by the BoostTrack++ paper
to its reference [14]'s code, which we did not consult. Where under-specified we
use a standard public MOT formulation, documented at the relevant code below and
tunable via config.

Pipeline (BoostTrack++ "useS + useSB + useVT" configuration):

  1. Kalman-predict all tracks (active ∪ lost), 8-D CV filter on (cx, cy, w, h).
  2. Embed all detections via the pluggable ReIDBackend (optional).
  3. Build a fused similarity S = mean(SBIoU, S^MhD, S^shape), optionally fused
     with appearance cosine. Admissibility is a raw-IoU floor; Mahalanobis is a
     soft term (its hard veto is OFF by default — it fragments IDs under
     non-linear motion).
  4. Boost each detection's confidence: soft blend (Eq 11) + varying-threshold
     boost (Eq 12-13) — likely-occluded detections survive the spawn gate.
  5. Single-stage greedy association on cost = 1 - S, gated by sim_threshold.
  6. Update matched tracks (KF + appearance EMA); re-promote matched lost tracks.
  7. Spawn new tracks from unmatched detections (BOOSTED conf ≥ new_track_thresh),
     with Track-Aware Initialization suppressing duplicate/overlapping spawns.
  8. Retire tracks unobserved > max_age; emit active tracks (KF posterior bbox).

The Kalman filter and association are implemented in-tree (numpy only in the
greedy default; `assignment: hungarian` uses scipy). Generic geometry and
assignment helpers live in `_matching.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ._matching import greedy_match_min as _greedy_match
from ._matching import hungarian_match
from ._matching import iou_matrix as _iou_matrix
from ._matching import xyxy_to_cxcywh as _xyxy_to_cxcywh
from .base import (
    AppearanceTracker,
    FramePlan,
    MatchedPair,
    SpawnedTrack,
    track_aware_spawn_filter,
)

if TYPE_CHECKING:
    from datetime import datetime

    from ..reid import ReIDBackend
    from ..schemas import Detection, Track


@dataclass(kw_only=True)
class _BoostTrackFramePlan(FramePlan):
    """Typed per-frame state ``_associate`` hands to ``_finalize``."""

    # (candidate detection, BOOSTED confidence, appearance feature) for every
    # unmatched detection that cleared the spawn gate; TAI filters these.
    spawn_candidates: list[tuple["Detection", float, "np.ndarray | None"]]


# χ² 0.95 quantile, 4 dof — Mahalanobis gate on a 4-d measurement (cx, cy, w, h).
_CHI2_GATE_4DOF = 9.4877

# Kalman noise weights (size-scaled), the standard values used across public
# MOT trackers (SORT lineage).
_KF_STD_WEIGHT_POS = 1.0 / 20.0
_KF_STD_WEIGHT_VEL = 1.0 / 160.0


# ---------------------------------------------------------------------------
# Kalman filter (constant-velocity on cx, cy, w, h) with Mahalanobis gating
# ---------------------------------------------------------------------------
# State vector: [cx, cy, w, h, vx, vy, vw, vh]; observation [cx, cy, w, h].
# project()/gating_distance() exist so BoostTrack's Mahalanobis similarity
# term can be computed.


class _KalmanFilter:
    """Constant-velocity Kalman filter on (cx, cy, w, h) with gating support."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._F = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim, dtype=np.float64)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement.astype(np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])
        std = np.array(
            [
                2 * _KF_STD_WEIGHT_POS * measurement[2],
                2 * _KF_STD_WEIGHT_POS * measurement[3],
                2 * _KF_STD_WEIGHT_POS * measurement[2],
                2 * _KF_STD_WEIGHT_POS * measurement[3],
                10 * _KF_STD_WEIGHT_VEL * measurement[2],
                10 * _KF_STD_WEIGHT_VEL * measurement[3],
                10 * _KF_STD_WEIGHT_VEL * measurement[2],
                10 * _KF_STD_WEIGHT_VEL * measurement[3],
            ],
            dtype=np.float64,
        )
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std_pos = np.array(
            [
                _KF_STD_WEIGHT_POS * mean[2],
                _KF_STD_WEIGHT_POS * mean[3],
                _KF_STD_WEIGHT_POS * mean[2],
                _KF_STD_WEIGHT_POS * mean[3],
            ]
        )
        std_vel = np.array(
            [
                _KF_STD_WEIGHT_VEL * mean[2],
                _KF_STD_WEIGHT_VEL * mean[3],
                _KF_STD_WEIGHT_VEL * mean[2],
                _KF_STD_WEIGHT_VEL * mean[3],
            ]
        )
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))
        new_mean = self._F @ mean
        new_cov = self._F @ covariance @ self._F.T + motion_cov
        return new_mean, new_cov

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution into measurement space (cx, cy, w, h)."""
        std = np.array(
            [
                _KF_STD_WEIGHT_POS * mean[2],
                _KF_STD_WEIGHT_POS * mean[3],
                _KF_STD_WEIGHT_POS * mean[2],
                _KF_STD_WEIGHT_POS * mean[3],
            ]
        )
        innovation_cov = np.diag(np.square(std))
        projected_mean = self._H @ mean
        projected_cov = self._H @ covariance @ self._H.T + innovation_cov
        return projected_mean, projected_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        kalman_gain = np.linalg.solve(projected_cov.T, (self._H @ covariance.T)).T
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
    ) -> np.ndarray:
        """Squared Mahalanobis distance of each measurement to the projection.

        `measurements`: (N, 4) in (cx, cy, w, h). Returns (N,). Falls back to a
        large finite distance on a singular covariance (gate stays closed only
        where the math is well-defined; never raises).
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        if measurements.size == 0:
            return np.zeros((0,), dtype=np.float64)
        try:
            inv = np.linalg.inv(projected_cov)
        except np.linalg.LinAlgError:
            return np.full((measurements.shape[0],), np.inf, dtype=np.float64)
        d = measurements - projected_mean[None, :]
        return np.einsum("ni,ij,nj->n", d, inv, d)


# ---------------------------------------------------------------------------
# Internal track object
# ---------------------------------------------------------------------------


@dataclass
class _BTrack:
    """Internal per-camera BoostTrack track. The public schema is `schemas.Track`."""

    track_id: int
    mean: np.ndarray  # 8-d KF state
    covariance: np.ndarray  # 8x8 KF covariance
    feature: np.ndarray | None  # EMA appearance feature (L2-normalized) or None
    score: float  # last (boosted) detection confidence — drives SBIoU buffer
    class_id: int
    class_name: str | None = None
    state: str = "tentative"
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    last_detection: "Detection | None" = field(default=None, repr=False)

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        cx, cy, w, h = self.mean[:4]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ---------------------------------------------------------------------------
# Geometry helpers — generic pieces are imported from `_matching`; only
# BoostTrack-specific geometry (buffered IoU, shape similarity) lives here.
# ---------------------------------------------------------------------------


def _buffered_iou(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    track_conf: np.ndarray,
) -> np.ndarray:
    """Soft-Buffered IoU (BoostTrack++ Eq 6-7).

    Each track box is expanded by ``(1 - c_t)/2`` and each detection box by
    ``(1 - c_t)/4`` of its own (w, h), where ``c_t`` is the track's confidence
    (so the buffer shrinks to zero for confident tracks and SBIoU → IoU). The
    buffer scale is per-track, so this returns an (M_tracks, N_dets) matrix.
    """
    m, n = track_boxes.shape[0], det_boxes.shape[0]
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float64)
    out = np.zeros((m, n), dtype=np.float64)
    c = np.clip(track_conf, 0.0, 1.0)
    for i in range(m):
        s_t = (1.0 - c[i]) / 2.0
        s_d = (1.0 - c[i]) / 4.0
        tb = _expand_boxes(track_boxes[i : i + 1], s_t)
        db = _expand_boxes(det_boxes, s_d)
        out[i, :] = _iou_matrix(tb, db)[0]
    return out


def _expand_boxes(boxes: np.ndarray, scale: float) -> np.ndarray:
    """Expand xyxy boxes by `scale` of their (w, h) on each side."""
    if scale <= 0.0 or boxes.size == 0:
        return boxes
    x1, y1, x2, y2 = boxes.T
    w = x2 - x1
    h = y2 - y1
    return np.stack([x1 - scale * w, y1 - scale * h, x2 + scale * w, y2 + scale * h], axis=1)


def _shape_similarity(track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    """S^shape = exp(-ds), ds = |Δw|/max(w) + |Δh|/max(h). (M,4),(N,4) -> (M,N).

    Clean-room transform of the paper's shape distance (see module docstring).
    """
    m, n = track_boxes.shape[0], det_boxes.shape[0]
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float64)
    tw = (track_boxes[:, 2] - track_boxes[:, 0])[:, None]  # (M,1)
    th = (track_boxes[:, 3] - track_boxes[:, 1])[:, None]
    dw = (det_boxes[:, 2] - det_boxes[:, 0])[None, :]  # (1,N)
    dh = (det_boxes[:, 3] - det_boxes[:, 1])[None, :]
    eps = 1e-6
    denom_w = np.maximum(np.maximum(tw, dw), eps)
    denom_h = np.maximum(np.maximum(th, dh), eps)
    ds = np.abs(tw - dw) / denom_w + np.abs(th - dh) / denom_h
    return np.exp(-ds)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class BoostTrackTracker(AppearanceTracker):
    """BoostTrack++ — single-stage tracker with similarity + confidence boosting.

    Structured on :class:`.base.AppearanceTracker`: the base
    owns the per-frame loop and the appearance channel (occlusion-gated
    embedding updates + ``frame_embeddings`` export); this class implements the
    BoostTrack++-specific hooks (predict / associate / matched-update /
    finalize / emit).

    Args:
        reid: Optional ReID embedder (`reid.ReIDBackend`). When present,
            appearance cosine is fused into the similarity. None → geometric-only.
        det_thresh: Confidence (after boosting) a detection needs to spawn a track.
        new_track_thresh: Spawn gate; defaults to ``det_thresh`` when None.
        max_age: Frames a lost track survives without a match before deletion.
        min_hits: Hits before a track is promoted tentative → active.
        sim_threshold: Minimum fused similarity S for an admissible match (τ_S).
        use_sb: Use Soft-Buffered IoU instead of plain IoU (Eq 6-7).
        use_mhd: Include the Mahalanobis distance as a soft similarity term.
        use_mhd_gate: ALSO treat Mahalanobis as a HARD veto (forbid matches with
            d² > mhd_gate). Default False — the hard veto fragments IDs under
            non-linear motion, so the admissibility gate is the raw-IoU floor
            below instead.
        iou_floor: Raw-IoU floor below which a (track, detection) pair cannot be
            associated — prevents far / appearance-only matches without vetoing a
            track's own next detection under motion (cf. TrackTrack's IoU gate).
        use_shape: Include the shape-similarity term.
        use_tai: Track-Aware Initialization — suppress spawning a new track from a
            detection that overlaps (IoU > tai_thr) an active track or a more
            confident new-track candidate (anti-duplicate; ported from TrackTrack).
        tai_thr: IoU overlap threshold for the TAI init guard.
        use_vt: Apply the varying-threshold confidence boost (Eq 12-13).
        soft_alpha: α in the soft confidence boost (Eq 11).
        soft_q: q exponent in the soft confidence boost (Eq 11).
        vt_beta_high: β_high — similarity threshold for freshly-updated tracks.
        vt_beta_low: β_low — threshold floor for stale tracks.
        vt_gamma: γ — per-frame decay of the varying threshold.
        mhd_gate: χ² gate on squared Mahalanobis distance (also the S^MhD scale).
        appearance_weight: Fusion weight w_app for appearance vs geometry
            (used only when `reid` is provided).
        ema_alpha: Appearance-feature EMA decay (weight on the existing feature).
        occlusion_gate_iou: Occlusion-gated embedding updates — enforced by the
            base; see :class:`.base.AppearanceTracker`.
            ``None`` (default) disables the gate.
        emit_tentative: emission policy (base knob) — True (default) emits
            tentative tracks from their first frame; False emits confirmed
            (``min_hits`` reached) tracks only.
        assignment: ``"greedy"`` (default — the eval-validated matcher the
            shipped gates were tuned with) or ``"hungarian"`` (globally-optimal
            solver, same threshold gate; flip only after a dedicated A/B).
        camera_id: Stable identifier emitted in produced Track records.
    """

    def __init__(
        self,
        reid: "ReIDBackend | None" = None,
        det_thresh: float = 0.6,
        new_track_thresh: float | None = None,
        max_age: int = 30,
        min_hits: int = 3,
        sim_threshold: float = 0.3,
        use_sb: bool = True,
        use_mhd: bool = True,
        use_mhd_gate: bool = False,
        iou_floor: float = 0.10,
        use_shape: bool = True,
        use_tai: bool = True,
        tai_thr: float = 0.55,
        use_vt: bool = True,
        soft_alpha: float = 0.65,
        soft_q: float = 1.5,
        vt_beta_high: float = 0.95,
        vt_beta_low: float = 0.8,
        vt_gamma: float = 0.0075,
        mhd_gate: float = _CHI2_GATE_4DOF,
        appearance_weight: float = 0.5,
        ema_alpha: float = 0.9,
        occlusion_gate_iou: float | None = None,
        emit_tentative: bool = True,
        assignment: str = "greedy",
        camera_id: str = "cam0",
    ) -> None:
        if assignment not in ("greedy", "hungarian"):
            raise ValueError(f"Unknown assignment '{assignment}'. Supported: greedy, hungarian.")
        super().__init__(
            reid=reid,
            occlusion_gate_iou=occlusion_gate_iou,
            emit_tentative=emit_tentative,
            camera_id=camera_id,
        )
        self._det_thresh = det_thresh
        self._new_track_thresh = det_thresh if new_track_thresh is None else new_track_thresh
        self._max_age = max_age
        self._min_hits = min_hits
        self._sim_threshold = sim_threshold
        self._use_sb = use_sb
        self._use_mhd = use_mhd
        self._use_mhd_gate = use_mhd_gate
        self._iou_floor = iou_floor
        self._use_shape = use_shape
        self._use_tai = use_tai
        self._tai_thr = tai_thr
        self._use_vt = use_vt
        self._soft_alpha = soft_alpha
        self._soft_q = soft_q
        self._vt_beta_high = vt_beta_high
        self._vt_beta_low = vt_beta_low
        self._vt_gamma = vt_gamma
        self._mhd_gate = mhd_gate
        self._appearance_weight = appearance_weight if reid is not None else 0.0
        self._ema_alpha = ema_alpha
        self._assignment = assignment

        self._kf = _KalmanFilter()
        self._tracks: list[_BTrack] = []
        self._lost: list[_BTrack] = []
        self._next_id = 1

    @property
    def tracker_name(self) -> str:
        return "boost-track"

    # ------------------------------------------------------------------
    # AppearanceTracker hooks (per-frame flow — see AppearanceTracker.update)
    # ------------------------------------------------------------------

    def _predict(self, frame: "np.ndarray | None") -> None:
        # Predict all tracks (active ∪ lost).
        for trk in self._tracks + self._lost:
            trk.mean, trk.covariance = self._kf.predict(trk.mean, trk.covariance)
            trk.age += 1
            trk.time_since_update += 1

    def _solve(
        self, cost: np.ndarray, threshold: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Assignment minimizing ``cost`` with pairs above ``threshold``
        infeasible — identical gating for both solvers."""
        if self._assignment == "hungarian":
            return hungarian_match(cost, cost <= threshold)
        return _greedy_match(cost, threshold)

    def _associate(self, detections: list["Detection"], frame: "np.ndarray | None") -> FramePlan:
        pool = self._tracks + self._lost

        # 1. Appearance embeddings for all detections (if a ReID backend exists).
        det_feats = self._encode_crops(detections, frame)

        # 2. Fused similarity matrix S over pool × detections, and the per-pair
        #    Mahalanobis gate mask.
        sim, gate_ok = self._similarity(pool, detections, det_feats)

        # 3. Boost detection confidence using the best similarity to any track.
        boosted_conf = self._boost_confidence(detections, pool, sim, gate_ok)

        # 4. Single-stage association: greedy on cost = 1 - S, gated by sim_threshold.
        cost = 1.0 - sim
        # Disallow gated-out pairs and pairs below the similarity floor.
        cost = np.where(gate_ok & (sim > self._sim_threshold), cost, np.inf)
        matches, _unmatched_trk_idx, unmatched_det_idx = self._solve(
            cost, threshold=1.0 - self._sim_threshold
        )

        pairs = [
            MatchedPair(
                track=pool[trk_i],
                det=detections[det_i],
                feat=det_feats[det_i] if det_feats is not None else None,
                payload=float(boosted_conf[det_i]),  # drives SBIoU next frame
            )
            for trk_i, det_i in matches
        ]

        # Occluders for the base's embedding gate: ACTIVE tracks left
        # unmatched this frame (their object is present, its detection was
        # suppressed). Unmatched LOST tracks are excluded — their predictions
        # drift over the max_age window and would flag clean crops.
        matched_trk = {trk_i for trk_i, _ in matches}
        unmatched_active = [
            pool[i].bbox_xyxy for i in range(len(self._tracks)) if i not in matched_trk
        ]
        occluders = np.array(unmatched_active, dtype=np.float64) if unmatched_active else None

        # Spawn candidates: unmatched detections whose BOOSTED confidence
        # clears the gate (TAI itself runs in _finalize — it needs the
        # post-transition active list).
        spawn_candidates = [
            (detections[i], float(boosted_conf[i]), det_feats[i] if det_feats is not None else None)
            for i in unmatched_det_idx
            if boosted_conf[i] >= self._new_track_thresh
        ]

        return _BoostTrackFramePlan(
            matches=pairs,
            occluder_boxes=occluders,
            spawn_candidates=spawn_candidates,
        )

    def _update_matched(
        self,
        match: MatchedPair,
        feat: np.ndarray | None,
        plan: FramePlan,
    ) -> None:
        self._update_track(match.track, match.det, match.payload, feat)

    def _finalize(
        self, detections: list["Detection"], plan: FramePlan, now: "datetime"
    ) -> list[SpawnedTrack]:
        assert isinstance(plan, _BoostTrackFramePlan)  # narrowed: our _associate made it
        # 1. Sort the pool back into active / lost, re-promoting matched lost
        #    tracks. Key on `track_id` (a stable, unique-per-track domain id) — not
        #    Python `id()`, which is not serialization-safe for Phase-2 IPC workers.
        matched_ids = {m.track.track_id for m in plan.matches}
        new_active: list[_BTrack] = []
        new_lost: list[_BTrack] = []
        for trk in self._tracks:
            if trk.track_id in matched_ids:
                new_active.append(trk)
            else:
                trk.state = "lost"
                new_lost.append(trk)
        for trk in self._lost:
            if trk.track_id in matched_ids:
                trk.state = "active"
                new_active.append(trk)
            elif trk.time_since_update <= self._max_age:
                new_lost.append(trk)
            # else drop (older than max_age)

        # 2. Spawn new tracks from the gated candidates, with Track-Aware
        #    Initialization suppressing duplicates.
        allow = self._tai_filter(plan.spawn_candidates, new_active)
        spawned_out: list[SpawnedTrack] = []
        for keep, (det, conf, feat) in zip(allow, plan.spawn_candidates, strict=True):
            if not keep:
                continue
            spawned = self._spawn(det, conf, feat)
            new_active.append(spawned)
            spawned_out.append(SpawnedTrack(spawned.track_id, det, feat))

        self._tracks = new_active
        self._lost = new_lost
        return spawned_out

    def _emit_active(self, now: "datetime") -> list["Track"]:
        # Emit active tracks (KF posterior bbox).
        return [self._to_track(trk, now) for trk in self._tracks if trk.last_detection is not None]

    def _reset_state(self) -> None:
        self._tracks = []
        self._lost = []
        self._next_id = 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _encode_crops(
        self, detections: list["Detection"], frame: "np.ndarray | None"
    ) -> np.ndarray | None:
        if self._reid is None or not detections or frame is None:
            return None
        h, w = frame.shape[:2]
        crops_bgr: list[np.ndarray] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            xi1 = max(0, int(round(x1)))
            yi1 = max(0, int(round(y1)))
            xi2 = min(w, int(round(x2)))
            yi2 = min(h, int(round(y2)))
            if xi2 <= xi1 or yi2 <= yi1:
                crops_bgr.append(np.zeros((8, 8, 3), dtype=np.uint8))
            else:
                crops_bgr.append(frame[yi1:yi2, xi1:xi2])
        feats = self._reid.embed(crops_bgr)
        return np.asarray(feats, dtype=np.float32)

    def _similarity(
        self,
        pool: list[_BTrack],
        detections: list["Detection"],
        det_feats: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (similarity (M,N), gate_ok mask (M,N) bool)."""
        m, n = len(pool), len(detections)
        if m == 0 or n == 0:
            return np.zeros((m, n)), np.ones((m, n), dtype=bool)

        track_boxes = np.array([t.bbox_xyxy for t in pool], dtype=np.float64)
        det_boxes = np.array([d.bbox for d in detections], dtype=np.float64)
        det_meas = np.array([_xyxy_to_cxcywh(d.bbox) for d in detections], dtype=np.float64)

        # Raw IoU is both an optional cost term and the admissibility floor gate.
        iou_raw = _iou_matrix(track_boxes, det_boxes)

        # --- geometric terms ---
        terms: list[np.ndarray] = []
        if self._use_sb:
            track_conf = np.array([t.score for t in pool], dtype=np.float64)
            terms.append(_buffered_iou(track_boxes, det_boxes, track_conf))
        else:
            terms.append(iou_raw)

        # Admissibility gate: raw-IoU floor (prevents far / appearance-only
        # matches without vetoing a track's own next detection under motion).
        gate_ok = iou_raw > self._iou_floor
        if self._use_mhd:
            s_mhd = np.zeros((m, n), dtype=np.float64)
            for i, trk in enumerate(pool):
                d2 = self._kf.gating_distance(trk.mean, trk.covariance, det_meas)  # (N,)
                s_mhd[i] = np.maximum(0.0, 1.0 - d2 / self._mhd_gate)
                # Mahalanobis is a SOFT term by default; only veto when explicitly enabled.
                if self._use_mhd_gate:
                    gate_ok[i] &= d2 <= self._mhd_gate
            terms.append(s_mhd)

        if self._use_shape:
            terms.append(_shape_similarity(track_boxes, det_boxes))

        s_geom = np.mean(np.stack(terms, axis=0), axis=0)

        # --- optional appearance fusion ---
        if det_feats is not None and self._appearance_weight > 0.0:
            track_feats = np.stack(
                [
                    (
                        t.feature
                        if t.feature is not None
                        else np.zeros(det_feats.shape[1], dtype=np.float32)
                    )
                    for t in pool
                ]
            )
            s_app = np.clip(track_feats @ det_feats.T, 0.0, 1.0)  # cosine, both L2-normed
            sim = (1.0 - self._appearance_weight) * s_geom + self._appearance_weight * s_app
        else:
            sim = s_geom

        # Gated-out pairs cannot match regardless of appearance.
        sim = np.where(gate_ok, sim, 0.0)
        return sim, gate_ok

    def _boost_confidence(
        self,
        detections: list["Detection"],
        pool: list[_BTrack],
        sim: np.ndarray,
        gate_ok: np.ndarray,
    ) -> np.ndarray:
        """BoostTrack++ detection-confidence boosting (Eq 11 soft + Eq 12-13 VT)."""
        raw = np.array([d.confidence for d in detections], dtype=np.float64)
        if len(pool) == 0 or sim.size == 0:
            return raw

        gated_sim = np.where(gate_ok, sim, 0.0)
        max_sim = gated_sim.max(axis=0)  # best similarity to any track, per detection (N,)

        # Soft boost (Eq 11): blend raw conf with a similarity-driven boost.
        soft = self._soft_alpha * raw + (1.0 - self._soft_alpha) * np.power(max_sim, self._soft_q)
        boosted = np.maximum(raw, soft)

        # Varying-threshold boost (Eq 12-13): if any track within its age-decayed
        # threshold β_j matches this detection, lift confidence to at least det_thresh.
        if self._use_vt:
            tsu = np.array([t.time_since_update for t in pool], dtype=np.float64)  # (M,)
            beta = np.maximum(self._vt_beta_low, self._vt_beta_high - self._vt_gamma * (tsu - 1.0))
            vt_hit = np.any(gated_sim >= beta[:, None], axis=0)  # (N,)
            boosted = np.where(vt_hit, np.maximum(boosted, self._det_thresh), boosted)

        return boosted

    def _tai_filter(
        self,
        candidates: list[tuple["Detection", float, "np.ndarray | None"]],
        active_tracks: list[_BTrack],
    ) -> np.ndarray:
        """Track-Aware Initialization: suppress a new-track candidate that overlaps
        an active track or a more-confident candidate (anti-duplicate). Returns a
        boolean keep-mask aligned with ``candidates``."""
        n = len(candidates)
        if not self._use_tai or n == 0:
            return np.ones(n, dtype=bool)
        cand_boxes = np.array([det.bbox for det, _, _ in candidates], dtype=np.float64)
        scores = np.array([conf for _, conf, _ in candidates], dtype=np.float64)
        iou_ct = (
            _iou_matrix(
                cand_boxes, np.array([t.bbox_xyxy for t in active_tracks], dtype=np.float64)
            )
            if active_tracks
            else np.zeros((n, 0))
        )
        iou_cc = _iou_matrix(cand_boxes, cand_boxes)
        # TAI core, ported from the TrackTrack paper (the BoostTrack++ paper
        # has no init suppression). Candidates were pre-filtered by
        # new_track_thresh, so no min_score here.
        return track_aware_spawn_filter(iou_ct, iou_cc, scores, self._tai_thr)

    def _update_track(
        self,
        trk: _BTrack,
        det: "Detection",
        boosted_conf: float,
        det_feat: np.ndarray | None,
    ) -> None:
        measurement = _xyxy_to_cxcywh(det.bbox)
        trk.mean, trk.covariance = self._kf.update(trk.mean, trk.covariance, measurement)
        # Store the BOOSTED confidence — it drives the SBIoU buffer next frame.
        trk.score = float(boosted_conf)
        trk.class_id = det.class_id
        trk.class_name = det.class_name
        trk.hits += 1
        trk.time_since_update = 0
        trk.state = "active" if trk.hits >= self._min_hits else "tentative"
        trk.last_detection = det
        if det_feat is not None:
            if trk.feature is None:
                trk.feature = det_feat
            else:
                blended = self._ema_alpha * trk.feature + (1.0 - self._ema_alpha) * det_feat
                trk.feature = blended / max(np.linalg.norm(blended), 1e-12)

    def _spawn(self, det: "Detection", boosted_conf: float, feat: np.ndarray | None) -> _BTrack:
        mean, cov = self._kf.initiate(_xyxy_to_cxcywh(det.bbox))
        trk = _BTrack(
            track_id=self._next_id,
            mean=mean,
            covariance=cov,
            feature=feat,
            score=float(boosted_conf),
            class_id=det.class_id,
            class_name=det.class_name,
            state="tentative",
            hits=1,
            age=1,
            time_since_update=0,
            last_detection=det,
        )
        self._next_id += 1
        return trk

    def _to_track(self, trk: _BTrack, now: datetime) -> "Track":
        from ..schemas import Detection, Track

        bbox = trk.bbox_xyxy
        det = trk.last_detection
        assert det is not None  # guarded by caller
        smoothed = Detection(
            bbox=bbox,
            class_id=trk.class_id,
            confidence=trk.score,
            class_name=trk.class_name,
        )
        return Track(
            track_id=trk.track_id,
            camera_id=self._camera_id,
            detection=smoothed,
            frame_idx=self._frame_idx,
            timestamp=now,
            state=trk.state,
        )
