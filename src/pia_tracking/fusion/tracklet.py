"""The unit of evidence the Global ID service reasons about.

A ``Tracklet`` is one segment of one single-camera track: the L2-normalised mean
of its per-frame ReID embeddings, when it was on screen, and which other local
tracks shared frames with it. ``TrackAccumulator`` builds one incrementally from
the embeddings the tracker already computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

_EPS = 1e-12


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), _EPS)


@dataclass(frozen=True)
class Tracklet:
    """One evidence segment of a single-camera track, ready for assignment.

    ``concurrent_track_ids`` are the local ids that shared a frame with this
    track on this camera — two boxes at once are two people, so the service
    must never merge this track into an identity holding one of them.
    """

    camera_id: str
    track_id: int
    embedding: np.ndarray
    first_seen: datetime
    last_seen: datetime
    frame_count: int
    concurrent_track_ids: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        where = f"cam={self.camera_id} track={self.track_id}"
        if self.frame_count < 1:
            raise ValueError(f"{where}: frame_count must be >= 1")
        if self.first_seen > self.last_seen:
            raise ValueError(
                f"{where}: first_seen {self.first_seen.isoformat()} is after last_seen "
                f"{self.last_seen.isoformat()}"
            )
        if self.first_seen.tzinfo is None or self.last_seen.tzinfo is None:
            # Mixing naive and aware datetimes raises deep inside a span
            # comparison; fail here, naming the producer.
            raise ValueError(f"{where}: timestamps must be timezone-aware")


class TrackAccumulator:
    """Running mean of one local track's ReID embeddings — the current evidence
    segment. Vectors are unit-norm already, so summing and normalising once at
    ``finalize`` equals averaging them."""

    __slots__ = ("track_id", "_sum", "_count", "_first_seen", "_last_seen", "_last_frame_idx", "_concurrent")

    def __init__(self, track_id: int) -> None:
        self.track_id = track_id
        self._sum: np.ndarray | None = None
        self._count = 0
        self._first_seen: datetime | None = None
        self._last_seen: datetime | None = None
        self._last_frame_idx = -1
        self._concurrent: set[int] = set()

    def add(self, embedding: np.ndarray, ts: datetime, frame_idx: int) -> None:
        e = np.asarray(embedding, dtype=np.float32)
        if self._sum is None:
            self._sum = e.copy()
            self._first_seen = ts
        else:
            self._sum += e
        self._count += 1
        self._last_seen = ts
        self._last_frame_idx = frame_idx

    def note_concurrent(self, frame_track_ids: set[int]) -> None:
        """Record the local ids on screen in the same frame (self excluded)."""
        self._concurrent |= frame_track_ids
        self._concurrent.discard(self.track_id)

    @property
    def frame_count(self) -> int:
        return self._count

    @property
    def last_frame_idx(self) -> int:
        return self._last_frame_idx

    def split_segment(self) -> tuple[np.ndarray, int, datetime]:
        """Take the current segment and start a fresh one, so a checkpoint and
        the frames after it cover disjoint spans — nothing is counted twice."""
        if self._sum is None or self._first_seen is None:
            raise ValueError(f"track_id={self.track_id}: cannot split an empty segment")
        segment = (self._sum, self._count, self._first_seen)
        self._sum = None
        self._count = 0
        self._first_seen = None
        return segment

    def merge_segment(self, embedding_sum: np.ndarray, count: int, first_seen: datetime) -> None:
        """Undo ``split_segment`` when the checkpoint assign failed."""
        self._sum = embedding_sum if self._sum is None else self._sum + embedding_sum
        self._count += count
        self._first_seen = first_seen if self._first_seen is None else min(self._first_seen, first_seen)

    def finalize(self, camera_id: str) -> Tracklet:
        if self._sum is None or self._first_seen is None or self._last_seen is None:
            raise ValueError(f"track_id={self.track_id}: cannot finalize an empty accumulator")
        return Tracklet(
            camera_id=camera_id,
            track_id=self.track_id,
            embedding=l2_normalize(self._sum),
            first_seen=self._first_seen,
            last_seen=self._last_seen,
            frame_count=self._count,
            concurrent_track_ids=frozenset(self._concurrent),
        )
