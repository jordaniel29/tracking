"""CameraWorker policy (checkpoint / loss / flush) against a scripted fake pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from pia_tracking.fusion import CameraWorker, GlobalIDService, TrackAccumulator
from pia_tracking.schemas import Detection, Track
from pia_tracking.tracking import FrameResult, RunStats

T0 = datetime(2000, 1, 1, tzinfo=timezone.utc)
FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


def unit(seed: int, dim: int = 16) -> np.ndarray:
    """Deterministic unit vector; distinct seeds (mod dim) are orthogonal."""
    v = np.zeros(dim, dtype=np.float32)
    v[seed % dim] = 1.0
    return v


def ts(frame_idx: int) -> datetime:
    return T0 + timedelta(seconds=frame_idx / 30.0)


class _FakeTracker:
    def __init__(self) -> None:
        self.embs: dict[int, np.ndarray] = {}

    def frame_embeddings(self):
        return self.embs


class _FakePipeline:
    """``script[frame_idx] = [(track_id, embedding | None), ...]``. None means the
    tracker did not embed that track this frame (worker must fall back)."""

    def __init__(self, camera_id: str, script: dict[int, list]) -> None:
        self.tracker = _FakeTracker()
        self.stats = RunStats()
        self._cam = camera_id
        self._script = script

    def process_frame(self, frame, frame_idx):
        entries = self._script.get(frame_idx, [])
        self.tracker.embs = {tid: e for tid, e in entries if e is not None}
        tracks = [
            Track(
                track_id=tid,
                camera_id=self._cam,
                detection=Detection(bbox=(0, 0, 20, 40), class_id=0, confidence=0.9),
                frame_idx=frame_idx,
                timestamp=T0,
            )
            for tid, _ in entries
        ]
        self.stats.frames += 1
        return FrameResult(frame_idx=frame_idx, tracks=tracks, detections=[])


class _FakeReID:
    def __init__(self, vec: np.ndarray) -> None:
        self.vec = vec
        self.calls = 0

    def embed(self, crops):
        self.calls += 1
        return np.stack([self.vec] * len(crops))


def make_worker(camera_id, script, service, reid=None, **kw):
    return CameraWorker(
        camera_id=camera_id,
        pipeline=_FakePipeline(camera_id, script),  # type: ignore[arg-type]
        reid=reid or _FakeReID(unit(99)),
        global_id=service,
        **kw,
    )


def test_checkpoint_assigns_after_40_frames_and_labels_from_then_on():
    a = unit(1)
    svc = GlobalIDService()
    w = make_worker("cam8", {i: [(1, a)] for i in range(60)}, svc, checkpoint_frames=40)
    seen = {}
    for i in range(60):
        seen[i] = w.process_frame(FRAME, i, ts(i)).tracks[0].global_id
    assert seen[38] is None and seen[39] == 1 and seen[59] == 1  # 40th embedded frame is idx 39
    assert w.stats.checkpoint_assigns == 1 and svc.stats.minted == 1


def test_short_track_lost_before_min_frames_gets_no_id():
    svc = GlobalIDService()
    w = make_worker("cam8", {i: [(1, unit(1))] for i in range(5)}, svc, min_frames_before_id_assign=10)
    for i in range(6):  # frame 5 has no track → loss
        w.process_frame(FRAME, i, ts(i))
    assert w.gid_map == {} and w.unassigned_track_ids == [1]
    assert w.stats.too_short == 1 and svc.stats.assign_calls == 0


def test_track_lost_before_checkpoint_but_long_enough_assigns_at_loss():
    svc = GlobalIDService()
    w = make_worker("cam8", {i: [(1, unit(1))] for i in range(15)}, svc)
    for i in range(16):
        w.process_frame(FRAME, i, ts(i))
    assert w.gid_map == {1: 1} and w.stats.loss_assigns == 1


def test_two_cameras_same_person_share_global_id():
    a = unit(1)
    svc = GlobalIDService()
    w8 = make_worker("cam8", {i: [(1, a)] for i in range(50)}, svc)
    w9 = make_worker("cam9", {i: [(3, a)] for i in range(50)}, svc)
    for i in range(50):  # interleaved, as infer.py --mode multi does
        w8.process_frame(FRAME, i, ts(i))
        w9.process_frame(FRAME, i, ts(i))
    assert w8.global_id_of(1) == w9.global_id_of(3) == 1
    assert svc.summary()["n_multi_camera"] == 1


def test_finish_flushes_open_tracks_and_reactivation_accumulates():
    a = unit(1)
    svc = GlobalIDService(revise_at_loss=False)
    script = {i: [(1, a)] for i in range(45)}  # checkpoint at 40, lost at 45
    script.update({i: [(1, a)] for i in range(50, 60)})  # same local id re-activates
    w = make_worker("cam8", script, svc)
    for i in range(60):
        w.process_frame(FRAME, i, ts(i))
    assert svc.stats.accumulated == 1  # frames 40-44 folded in at the loss
    w.finish(ts(60))
    assert svc.stats.accumulated == 2  # frames 50-59 folded in at flush
    # 40 (checkpoint) + 5 (loss) + 10 (flush) embedded frames, one identity.
    assert svc.identities[1].weight == 55 and w.gid_map == {1: 1}


def test_too_short_then_reactivated_track_is_not_left_unlabelled():
    a = unit(1)
    svc = GlobalIDService()
    script = {i: [(1, a)] for i in range(4)}  # 4 frames, lost → too short, no id
    script.update({i: [(1, a)] for i in range(10, 30)})  # same local id back for 20 frames
    w = make_worker("cam8", script, svc, min_frames_before_id_assign=10)
    for i in range(31):
        w.process_frame(FRAME, i, ts(i))
    assert w.stats.too_short == 1
    assert w.gid_map == {1: 1} and w.unassigned_track_ids == []


def test_fallback_embeds_tracks_the_tracker_did_not():
    svc = GlobalIDService()
    reid = _FakeReID(unit(5))
    w = make_worker("cam8", {i: [(1, None)] for i in range(12)}, svc, reid=reid)
    for i in range(13):
        w.process_frame(FRAME, i, ts(i))
    assert reid.calls == 12 and w.stats.fallback_embeds == 12
    assert w.gid_map == {1: 1}


def test_accumulator_split_and_merge_partition_evidence():
    acc = TrackAccumulator(7)
    a = unit(1)
    for i in range(3):
        acc.add(a, ts(i), i)
    seg = acc.split_segment()
    assert acc.frame_count == 0
    acc.add(a, ts(3), 3)
    acc.merge_segment(*seg)
    t = acc.finalize("cam")
    assert t.frame_count == 4 and t.first_seen == ts(0) and t.last_seen == ts(3)
    assert np.allclose(t.embedding, a)
