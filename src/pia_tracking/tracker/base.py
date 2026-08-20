"""Tracker contracts: the public ABC and the shared appearance template.

Two layers, both in this module:

* ``Tracker`` — the *public contract*, what the pipeline calls. Every tracker
  implementation goes behind it; this package ships one, BoostTrack++
  (``boosttrack.py``).
* ``AppearanceTracker`` — the *shared implementation frame* for
  appearance-capable trackers: it owns the per-frame update loop and, with it,
  the whole appearance channel that every tracker must handle identically:

  - the per-frame ``frame_embeddings()`` export bookkeeping (what callers can
    reuse — e.g. for cross-camera matching — instead of re-embedding crops);
  - **occlusion-gated embedding updates**: a detection whose box overlaps
    another current-frame box is a contaminated crop — its embedding must
    neither blend into the track's appearance EMA nor be exported. The
    base computes the occlusion flags and withholds the feature *before*
    the adapter's update hook ever sees it, so a new tracker cannot forget
    (or re-implement) the gate.

Per-frame flow (``update`` is the template — adapters do NOT override it):

    update(detections, frame):
        1. _predict(frame)                # KF predict / motion compensation
        2. plan = _associate(dets, frame) # all matching DECISIONS, no
                                          #   appearance mutation (contract!)
        3. occlusion flags                # base-owned, from plan
        4. for each plan.match:           # gated update loop
               feat = match.feat unless match.det occluded → None
               _update_matched(match, feat, plan)   # KF update + EMA(feat)
               export feat (or _export_embedding override)
        5. spawned = _finalize(dets, plan, now)  # transitions, retirement,
                                          #   spawns — RETURNS SpawnedTrack
                                          #   records; the base gates and
                                          #   exports their embeddings
        6. return _emit_active(now)       # emission policy applied by base

Hook contract for a new tracker:
  - ``_associate`` may mutate motion/bookkeeping state but MUST NOT touch any
    track's appearance feature and MUST NOT write ``_frame_embeddings`` — the
    only appearance write paths are the feature handed to ``_update_matched``
    and the ``SpawnedTrack`` records returned by ``_finalize``, both gated by
    the base.
  - ``MatchedPair.det`` / ``SpawnedTrack.det`` carry the ORIGINAL
    ``Detection`` objects from this frame's list — the base resolves them by
    identity and raises immediately on a foreign object, so index-mapping
    bugs cannot exist (adapters that split/reorder detections just keep the
    object references).
  - Per-frame adapter state rides on a TYPED ``FramePlan`` subclass (see
    ``boosttrack.py``), not on an untyped dict.
  - Internal track objects must expose ``.track_id``.
  - A tracker with no appearance model at all (e.g. a geometry-only OC-SORT)
    would stay on the plain ``Tracker`` ABC — this base buys it nothing.

Config: every subclass accepts ``occlusion_gate_iou`` (float | None). ``None``
(the default) disables the gate — behaviour is then byte-identical to a
tracker without it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from ._matching import iou_matrix as _iou_matrix

if TYPE_CHECKING:
    from ..reid import ReIDBackend
    from ..schemas import Detection, Track


class Tracker(ABC):
    """Abstract base class for single-camera trackers.

    A Tracker maintains state across frames and assigns persistent IDs to
    objects within ONE camera. Cross-camera ID assignment is out of scope for
    this package.
    """

    @abstractmethod
    def update(
        self,
        detections: list["Detection"],
        frame: "np.ndarray",
    ) -> list["Track"]:
        """Update tracker with new frame's detections.

        Args:
            detections: Detections from the current frame.
            frame: Current frame (HWC uint8 BGR) — used for appearance features
                   in trackers that compute embeddings.

        Returns:
            Active tracks as of this frame. Each Track has a stable track_id.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all state (e.g., on stream restart)."""

    @property
    @abstractmethod
    def tracker_name(self) -> str:
        """Stable tracker identifier for logging."""

    def frame_embeddings(self) -> "dict[int, np.ndarray] | None":
        """Raw per-frame appearance embeddings from the most recent ``update()``.

        Keyed by ``track_id``, one entry per active track for which the tracker
        computed an appearance embedding this frame (during association). Lets
        callers reuse them — e.g. for cross-camera global-ID accumulation —
        instead of re-embedding the identical crops a second time.

        Returns ``None`` (the default) for trackers with no appearance model;
        callers must fall back to embedding crops themselves in that case.
        """
        return None


@dataclass
class MatchedPair:
    """One (existing track, current detection) association decided by
    ``_associate``.

    Attributes:
        track: the adapter's internal track object (must expose ``.track_id``).
        det: the matched ``Detection`` — the ORIGINAL object from this frame's
            detections list (resolved by identity in the base; a copy or a
            foreign object raises immediately).
        feat: appearance embedding of this detection, or None for a purely
            geometric match (e.g. a low-confidence second-stage match). The
            base may still withhold it from the update hook when the
            detection is occlusion-flagged.
        payload: adapter-private per-match data (e.g. BoostTrack's boosted
            confidence).
    """

    track: Any
    det: "Detection"
    feat: np.ndarray | None = None
    payload: Any = None


@dataclass(kw_only=True)
class FramePlan:
    """Everything ``_associate`` decided about the current frame.

    Adapters carry their per-frame state on a TYPED subclass (e.g. spawn
    candidates, per-frame score arrays) instead of an untyped dict — the same
    adapter produces and consumes the plan, so the subclass can be precise.

    Attributes:
        occluder_boxes: REQUIRED (pass ``None`` explicitly for "no occluders"
            — the field has no default so the decision cannot be forgotten).
            (M, 4) xyxy boxes of tracks left unmatched this frame that should
            count as occluders for the embedding gate — typically the
            recently-seen (active/Tracked) unmatched tracks, whose object is
            present but whose detection was suppressed. Long-lost tracks must
            be excluded (drifted predictions would flag clean crops).
        matches: association pairs, in the order their updates must be applied.
    """

    occluder_boxes: np.ndarray | None
    matches: list[MatchedPair] = field(default_factory=list)


@dataclass
class SpawnedTrack:
    """A track spawned in ``_finalize`` — returned so the BASE can gate and
    export its embedding (the spawn-side counterpart of the match loop).

    Attributes:
        track_id: id of the freshly spawned track.
        det: the spawning ``Detection`` (original object, identity-resolved).
        feat: its appearance embedding, or None. Exported only if the
            detection was not occlusion-flagged; the spawned track keeps the
            feature for association either way (a track needs some appearance
            to match at all).
    """

    track_id: int
    det: "Detection"
    feat: np.ndarray | None


class AppearanceTracker(Tracker):
    """Shared update-loop frame for appearance-capable trackers.

    Args (common to all subclasses; forwarded via ``super().__init__``):
        reid: optional ReID embedder. None → geometric-only tracking and
            ``frame_embeddings()`` returns None.
        occlusion_gate_iou: occlusion-gated embedding updates. When set, a
            matched track whose detection box overlaps another current-frame
            box (another detection, or a track in ``plan.occluder_boxes``)
            with IoU strictly above this value keeps its last clean embedding:
            the base passes ``feat=None`` to ``_update_matched`` and
            suppresses the export. ``None`` disables the gate entirely.
        emit_tentative: emission policy. True (default) emits tentative
            (not-yet-confirmed) tracks alongside confirmed ones — every
            adapter then reports from an object's first frame, so tracker A/Bs
            compare on an identical emission policy.
            False emits confirmed (``state == "active"``) tracks only.
        camera_id: stable identifier emitted in produced Track records.
    """

    def __init__(
        self,
        *,
        reid: "ReIDBackend | None" = None,
        occlusion_gate_iou: float | None = None,
        emit_tentative: bool = True,
        camera_id: str = "cam0",
    ) -> None:
        self._reid = reid
        self._occlusion_gate_iou = occlusion_gate_iou
        self._emit_tentative = emit_tentative
        self._camera_id = camera_id
        self._frame_idx = 0
        self._frame_embeddings: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Template — final; adapters implement the hooks below instead
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list["Detection"],
        frame: "np.ndarray | None" = None,
    ) -> list["Track"]:
        if frame is None and self._frame_required():
            # Fail at the contract boundary, not deep inside GMC/crop code —
            # pre-template, frame-requiring adapters made this a TypeError.
            raise ValueError(
                f"{type(self).__name__} requires a frame for this configuration "
                "(motion compensation and ReID crops need pixels); pass the BGR "
                "image to update()"
            )
        self._frame_idx += 1
        now = datetime.now(timezone.utc)
        self._frame_embeddings = {}
        # Identity map: MatchedPair.det / SpawnedTrack.det must be THE objects
        # from this list — resolves each back to its index (occlusion flags are
        # positional) and turns any mapping bug into an immediate error.
        det_index = {id(d): i for i, d in enumerate(detections)}

        self._predict(frame)
        plan = self._associate(detections, frame)
        occluded = self._occlusion_flags(detections, plan.occluder_boxes)

        for match in plan.matches:
            # Resolved unconditionally (not just when a feature exists) so an
            # identity/mapping bug surfaces on the first frame it occurs.
            det_i = self._resolve(det_index, match.det, "MatchedPair")
            feat = match.feat
            if feat is not None and occluded[det_i]:
                # Occlusion-gated embedding update: the crop overlaps another
                # object — the track keeps its last clean embedding.
                feat = None
            self._update_matched(match, feat, plan)
            if feat is not None:
                export = self._export_embedding(match.track, feat)
                if export is not None:
                    self._frame_embeddings[match.track.track_id] = export

        for spawn in self._finalize(detections, plan, now) or ():
            # Spawn-side gate: a contaminated spawn crop is withheld from the
            # global-ID accumulator (the track keeps the feature internally).
            det_i = self._resolve(det_index, spawn.det, "SpawnedTrack")
            if spawn.feat is not None and not occluded[det_i]:
                self._frame_embeddings[spawn.track_id] = spawn.feat

        tracks = self._emit_active(now)
        if self._emit_tentative:
            return tracks
        # Emission policy: confirmed tracks only.
        return [t for t in tracks if t.state == "active"]

    @staticmethod
    def _resolve(det_index: dict[int, int], det: "Detection", owner: str) -> int:
        idx = det_index.get(id(det))
        if idx is None:
            raise ValueError(
                f"{owner}.det is not one of this frame's detections — adapters "
                "must pass the ORIGINAL Detection objects, not copies"
            )
        return idx

    def reset(self) -> None:
        self._frame_idx = 0
        self._frame_embeddings = {}
        self._reset_state()

    def frame_embeddings(self) -> "dict[int, np.ndarray] | None":
        return self._frame_embeddings if self._reid is not None else None

    # ------------------------------------------------------------------
    # Base-owned appearance hygiene
    # ------------------------------------------------------------------

    def _occlusion_flags(
        self,
        detections: list["Detection"],
        occluder_boxes: np.ndarray | None,
    ) -> np.ndarray:
        """Per-detection contamination flags (all-False when the gate is off)."""
        if self._occlusion_gate_iou is None or not detections:
            return np.zeros(len(detections), dtype=bool)
        det_boxes = np.array([d.bbox for d in detections], dtype=np.float64)
        return _occlusion_mask(det_boxes, occluder_boxes, self._occlusion_gate_iou)

    # ------------------------------------------------------------------
    # Hooks — the algorithm-specific parts a concrete tracker implements
    # ------------------------------------------------------------------

    @abstractmethod
    def _predict(self, frame: "np.ndarray | None") -> None:
        """Advance motion state for all tracks (Kalman predict, aging)."""

    @abstractmethod
    def _associate(self, detections: list["Detection"], frame: "np.ndarray | None") -> FramePlan:
        """Decide this frame's matches / spawn candidates / occluders.

        May encode crops internally and mutate motion/bookkeeping state, but
        must not touch appearance features or exports (see module docstring).
        """

    @abstractmethod
    def _update_matched(
        self,
        match: MatchedPair,
        feat: np.ndarray | None,
        plan: FramePlan,
    ) -> None:
        """Fold one matched detection (``match.det``) into its track (KF
        update, state, appearance EMA). ``feat`` is None for geometric matches
        AND for occlusion-gated frames — blend appearance only from ``feat``."""

    @abstractmethod
    def _finalize(
        self, detections: list["Detection"], plan: FramePlan, now: datetime
    ) -> "list[SpawnedTrack] | None":
        """State transitions, retirement, and spawns.

        Return a ``SpawnedTrack`` per spawn whose embedding should be exported
        (the base applies the occlusion gate); return None/[] when the adapter
        exports nothing for spawns (e.g. one whose export is the EMA feature,
        which a first-frame track does not have yet)."""

    @abstractmethod
    def _emit_active(self, now: datetime) -> list["Track"]:
        """Produce the public Track records for this frame."""

    @abstractmethod
    def _reset_state(self) -> None:
        """Clear adapter-owned state (tracks, id counter, caches)."""

    def _export_embedding(self, track: Any, feat: np.ndarray) -> np.ndarray | None:
        """What to export for a matched track this frame. Default: the raw
        detection embedding. Override to export e.g. the track's EMA feature.
        Return None to suppress the export."""
        return feat

    def _frame_required(self) -> bool:
        """Whether ``update()`` must reject ``frame=None`` up front.

        Default False: adapters that degrade gracefully without pixels
        (geometric-only association) accept a missing frame. Adapters whose
        current configuration genuinely needs pixels (camera-motion
        compensation, ReID crops) override this so the crash happens at the
        contract boundary with a clear message, not deep inside cv2/crop
        code."""
        return False


# ---------------------------------------------------------------------------
# Spawn suppression
# ---------------------------------------------------------------------------
# Track-Aware Initialization (TAI) originates in the TrackTrack paper; the
# BoostTrack++ paper itself has no init suppression, so its use here (the
# tracker's `use_tai` knob) is an in-house addition ported from that paper.


def track_aware_spawn_filter(
    iou_cand_track: np.ndarray,
    iou_cand_cand: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    min_score: float | None = None,
) -> np.ndarray:
    """Track-Aware Initialization keep-mask over new-track candidates.

    A candidate is suppressed when it overlaps an existing active track
    (``iou_cand_track`` row max > ``iou_threshold``), when a MORE confident
    surviving candidate overlaps it (> ``iou_threshold``), or — if
    ``min_score`` is given — when its own score does not exceed it.

    Args:
        iou_cand_track: (N, M) IoU of candidates vs active tracks (M may be 0).
        iou_cand_cand: (N, N) IoU of candidates vs each other.
        scores: (N,) candidate confidences.
        iou_threshold: overlap above which suppression applies (strict ``>``).
        min_score: optional hard floor on candidate score (strict ``>``).

    Returns:
        (N,) bool keep-mask.
    """
    n = len(scores)
    allow = np.ones(n, dtype=bool) if min_score is None else scores > min_score
    for idx in range(n):
        if not allow[idx]:
            continue
        if iou_cand_track.shape[1] > 0 and np.max(iou_cand_track[idx]) > iou_threshold:
            allow[idx] = False
            continue
        for jdx in range(n):
            if (
                idx != jdx
                and allow[jdx]
                and scores[idx] > scores[jdx]
                and iou_cand_cand[idx, jdx] > iou_threshold
            ):
                allow[jdx] = False
    return allow


def _occlusion_mask(
    det_boxes: np.ndarray,
    other_boxes: np.ndarray | None,
    iou_threshold: float,
) -> np.ndarray:
    """Boolean mask of detections whose box overlaps another current-frame box.

    A detection's crop is contaminated when it contains parts of another
    object, so its embedding must not update track appearance state. Overlap
    is checked against BOTH the other detections of this frame and
    ``other_boxes`` (tracks left unmatched this frame — an occluder whose own
    detection was suppressed). The threshold is a strict ``>``.
    """
    n = det_boxes.shape[0]
    occluded = np.zeros(n, dtype=bool)
    if n == 0:
        return occluded
    if n > 1:
        iou_dd = _iou_matrix(det_boxes, det_boxes)
        np.fill_diagonal(iou_dd, 0.0)
        occluded |= iou_dd.max(axis=1) > iou_threshold
    if other_boxes is not None and other_boxes.shape[0] > 0:
        occluded |= _iou_matrix(det_boxes, other_boxes).max(axis=1) > iou_threshold
    return occluded
