"""``CameraWorker`` — one camera's tracks, fed to the shared ``GlobalIDService``.

TRACE's ``PipelineWorker`` reduced to what offline video needs. Per frame:

    TrackingPipeline.process_frame        detect → embed → associate (local ids)
    tracker.frame_embeddings()            the ReID vectors the tracker already
                                          computed — same crops, no second pass
    TrackAccumulator per local track      running mean of those vectors
    checkpoint                            ``checkpoint_frames`` embedded frames
                                          and no id yet → GlobalIDService.assign
    loss                                  track absent from this frame → assign
                                          its remaining frames (a short track
                                          that never checkpointed needs
                                          ``min_frames_before_id_assign``)
    end of source                         ``finish`` flushes every open track

The worker owns the local → global map. Every ``Track`` it emits carries
``global_id`` — None until assigned, so a label can fall back to the local id.

TRACE runs one asyncio task per camera against the shared service. Here the
caller interleaves cameras frame by frame (``camera.round_robin``): the same
causal order — an id decision only ever sees identities from its past — in one
process with one copy of each model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from ..schemas import Track
from ..tracking.pipeline import FrameResult, TrackingPipeline
from ..utils.image import crop_bgr
from .global_id import GlobalIDService
from .tracklet import TrackAccumulator

logger = logging.getLogger("pia_tracking.fusion.worker")

DEFAULT_CHECKPOINT_FRAMES = 40
DEFAULT_MIN_FRAMES_BEFORE_ID_ASSIGN = 10


@dataclass
class CameraWorkerStats:
    checkpoint_assigns: int = 0
    loss_assigns: int = 0
    too_short: int = 0  # tracks that ended without an id (fewer than min frames)
    fallback_embeds: int = 0  # crops embedded here because the tracker had no vector


class CameraWorker:
    """Per-camera driver: pipeline + accumulators + local→global map.

    Args:
        camera_id: stamped on every Tracklet and Track.
        pipeline: this camera's ``TrackingPipeline`` (its own tracker state).
        reid: the embedder, for tracks the tracker did not embed this frame.
            Required — without appearance there is nothing to link on.
        global_id: the service shared by every camera.
        checkpoint_frames: embedded frames before a still-visible track asks
            for an id (0 disables checkpoints; loss-time assign only).
        min_frames_before_id_assign: a track lost before its checkpoint asks
            only with at least this many frames; shorter ones end unlabelled.
    """

    def __init__(
        self,
        *,
        camera_id: str,
        pipeline: TrackingPipeline,
        reid: Any,
        global_id: GlobalIDService,
        checkpoint_frames: int = DEFAULT_CHECKPOINT_FRAMES,
        min_frames_before_id_assign: int = DEFAULT_MIN_FRAMES_BEFORE_ID_ASSIGN,
    ) -> None:
        if reid is None:
            raise ValueError("CameraWorker needs a ReID embedder — cross-camera ids link on appearance")
        self._camera_id = camera_id
        self._pipeline = pipeline
        self._tracker = pipeline.tracker
        self._reid = reid
        self._service = global_id
        self._checkpoint_frames = int(checkpoint_frames)
        self._min_frames = max(0, int(min_frames_before_id_assign))
        if self._checkpoint_frames > 0 and self._min_frames >= self._checkpoint_frames:
            # Every track that ends before its first checkpoint is shorter than
            # checkpoint_frames, so a floor at or above it denies all of them.
            logger.warning(
                "camera=%s min_frames_before_id_assign=%d >= checkpoint_frames=%d — "
                "no track that ends before its first checkpoint can get an id",
                camera_id, self._min_frames, self._checkpoint_frames,
            )
        self._active: dict[int, TrackAccumulator] = {}
        self._gids: dict[int, int] = {}
        self._unassigned: set[int] = set()
        self.stats = CameraWorkerStats()

    # ── read side ────────────────────────────────────────────────────────

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def pipeline_stats(self):
        return self._pipeline.stats

    @property
    def gid_map(self) -> dict[int, int]:
        """local track_id → global_id for every track that got one."""
        return dict(self._gids)

    @property
    def unassigned_track_ids(self) -> list[int]:
        """Local tracks that ended without a global id."""
        return sorted(self._unassigned)

    def global_id_of(self, track_id: int) -> int | None:
        return self._gids.get(track_id)

    # ── per frame ────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_idx: int, ts: datetime) -> FrameResult:
        """One frame: track, accumulate, checkpoint, finalize losses. The
        returned tracks carry ``ts`` and their ``global_id`` (None if unknown)."""
        result = self._pipeline.process_frame(frame, frame_idx)
        tracks = result.tracks
        current = {t.track_id for t in tracks}

        self._open_accumulators(current)
        if tracks:
            self._accumulate(frame, tracks, ts, frame_idx)
        self._checkpoint_unlabelled(current)
        for tid in set(self._active) - current:
            self._finalize_track(tid, reason="lost")

        return FrameResult(frame_idx=frame_idx, tracks=self._stamp(tracks, ts), detections=result.detections)

    def finish(self, ts: datetime) -> None:
        """Source exhausted: assign every track still open."""
        for tid in list(self._active):
            self._finalize_track(tid, reason="source-exhausted")

    # ── steps ────────────────────────────────────────────────────────────

    def _open_accumulators(self, current: set[int]) -> None:
        for tid in current:
            if tid not in self._active:
                self._active[tid] = TrackAccumulator(tid)
        if len(current) > 1:
            # Two boxes in one frame are two people; the service must never merge them.
            for tid in current:
                self._active[tid].note_concurrent(current)

    def _accumulate(self, frame: np.ndarray, tracks: list[Track], ts: datetime, frame_idx: int) -> None:
        """Add this frame's embedding to each track — the tracker's own vector
        where it has one, else a fresh embedding of the crop (batched)."""
        reused = self._tracker.frame_embeddings() or {}
        pending: list[tuple[int, np.ndarray]] = []
        for t in tracks:
            emb = reused.get(t.track_id)
            if emb is not None:
                self._active[t.track_id].add(emb, ts, frame_idx)
                continue
            # The tracker embeds only the crops it associated on (and withholds
            # occluded ones); anything else is embedded here.
            crop = crop_bgr(frame, t.detection.bbox)
            if crop is not None:
                pending.append((t.track_id, crop))
        if pending:
            self._embed_fallback(pending, ts, frame_idx)

    def _embed_fallback(self, pending: list[tuple[int, np.ndarray]], ts: datetime, frame_idx: int) -> None:
        try:
            embeddings = self._reid.embed([c for _, c in pending])
        except Exception as e:  # noqa: BLE001 — a bad batch must not stop the camera
            logger.warning("reid_failed camera=%s frame=%d crops=%d err=%s", self._camera_id, frame_idx, len(pending), e)
            return
        for (tid, _), emb in zip(pending, embeddings, strict=True):
            self._active[tid].add(emb, ts, frame_idx)
        self.stats.fallback_embeds += len(pending)

    def _checkpoint_unlabelled(self, current: set[int]) -> None:
        """Mid-track id assignment for on-screen tracks that have none yet."""
        if self._checkpoint_frames <= 0 or current <= self._gids.keys():
            return
        for tid in current:
            if tid in self._gids:
                continue
            acc = self._active[tid]
            if acc.frame_count == 0 or acc.frame_count % self._checkpoint_frames:
                continue
            self._checkpoint(tid, acc)

    def _checkpoint(self, track_id: int, acc: TrackAccumulator) -> None:
        tracklet = acc.finalize(self._camera_id)
        segment = acc.split_segment()  # the frames after this stay disjoint
        try:
            gid = self._service.assign(tracklet)
        except Exception as e:  # noqa: BLE001 — keep the evidence for the loss-time assign
            acc.merge_segment(*segment)
            logger.warning("checkpoint_assign_failed camera=%s track=%d err=%s", self._camera_id, track_id, e)
            return
        self._note_gid(track_id, gid)
        self.stats.checkpoint_assigns += 1

    def _finalize_track(self, track_id: int, *, reason: str) -> None:
        """A track ended (or the source did): assign whatever it accumulated
        since its last checkpoint."""
        acc = self._active.pop(track_id, None)
        if acc is None:
            return
        if acc.frame_count == 0:
            # Nothing new since the checkpoint (or never embedded at all).
            if track_id not in self._gids:
                self._unassigned.add(track_id)
            return
        tracklet = acc.finalize(self._camera_id)
        if track_id not in self._gids and tracklet.frame_count < self._min_frames:
            self._unassigned.add(track_id)
            self.stats.too_short += 1
            logger.debug(
                "assign_skipped camera=%s track=%d frames=%d < min=%d reason=%s",
                self._camera_id, track_id, tracklet.frame_count, self._min_frames, reason,
            )
            return
        try:
            gid = self._service.assign(tracklet)
        except Exception as e:  # noqa: BLE001
            logger.warning("assign_failed camera=%s track=%d reason=%s err=%s", self._camera_id, track_id, reason, e)
            if track_id not in self._gids:
                self._unassigned.add(track_id)
            return
        self._note_gid(track_id, gid)
        self.stats.loss_assigns += 1

    def _note_gid(self, track_id: int, gid: int) -> None:
        self._gids[track_id] = gid
        # A local id can end too short, re-activate (BoostTrack keeps it alive
        # for max_age frames), run long and earn an id — it is not unlabelled
        # any more.
        self._unassigned.discard(track_id)

    def _stamp(self, tracks: list[Track], ts: datetime) -> list[Track]:
        return [t.model_copy(update={"timestamp": ts, "global_id": self._gids.get(t.track_id)}) for t in tracks]
