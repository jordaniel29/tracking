"""Cross-camera identity: ``Tracklet`` → ``global_id``.

In-memory, single-process reduction of TRACE's Progressive-HAC Global ID
service to the decision rule in the design diagram:

    tracklet arrives from any camera
      ├─ this (camera, track) already has a global_id
      │     → fold the new frames into that identity, then re-check that it
      │       is still the best match (``revise_at_loss``)
      └─ otherwise
            → candidates = identities seen within ``reidentify_within_sec`` on
              ANY camera, minus those that are provably someone else (a member
              on the SAME camera at the SAME time — two boxes at once are two
              people)
            → best cosine ≥ ``similarity_threshold`` ? reuse that id : mint one

Each identity keeps its centroid as the frame-count-weighted SUM of the
tracklet embeddings and normalises on read, so the result does not depend on
the order the cameras deliver in.

Deliberately not carried over from TRACE: Milvus persistence, Redis events,
per-camera mean-centering, exemplar set-matching, camera-topology transit
gating, rehydrate. Cameras are treated as overlapping — a person may be seen on
two cameras at the same instant — so appearance is the only cross-camera
discriminator and the recency window is the only temporal one.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any

import numpy as np

from .tracklet import Tracklet, l2_normalize

logger = logging.getLogger("pia_tracking.fusion.global_id")

DEFAULT_SIMILARITY_THRESHOLD = 0.45
DEFAULT_REIDENTIFY_WITHIN_SEC = 600.0
DEFAULT_REVISE_MARGIN = 0.05


@dataclass
class Identity:
    """One global identity: weighted centroid + the (camera, track) members.

    ``spans`` records, per camera, when each member was on screen — the
    same-camera co-occurrence exclusion reads it.
    """

    global_id: int
    vector_sum: np.ndarray  # Σ embᵢ · frame_countᵢ, unnormalised
    weight: int  # Σ frame_countᵢ
    members: list[tuple[str, int]]
    last_camera: str
    last_updated: datetime
    spans: dict[str, list[tuple[int, datetime, datetime]]] = field(default_factory=dict)

    @property
    def centroid(self) -> np.ndarray:
        return l2_normalize(self.vector_sum)

    @property
    def cameras(self) -> list[str]:
        return sorted({cam for cam, _ in self.members})

    def overlaps_on_camera(self, camera_id: str, first_seen: datetime, last_seen: datetime) -> bool:
        """True if a member on ``camera_id`` was on screen during [first_seen, last_seen]."""
        return any(
            first <= last_seen and first_seen <= last
            for _, first, last in self.spans.get(camera_id, ())
        )

    def contains_concurrent(self, camera_id: str, track_ids: frozenset[int]) -> bool:
        if not track_ids:
            return False
        mine = {tid for tid, _, _ in self.spans.get(camera_id, ())}
        return not mine.isdisjoint(track_ids)

    def add_member(self, tracklet: Tracklet, emb: np.ndarray) -> None:
        self.vector_sum = self.vector_sum + emb * tracklet.frame_count
        self.weight += tracklet.frame_count
        self.members.append((tracklet.camera_id, tracklet.track_id))
        self.spans.setdefault(tracklet.camera_id, []).append(
            (tracklet.track_id, tracklet.first_seen, tracklet.last_seen)
        )
        self._touch(tracklet)

    def add_segment(self, tracklet: Tracklet, emb: np.ndarray) -> None:
        """More frames for an EXISTING member: grow the centroid, widen its span."""
        self.vector_sum = self.vector_sum + emb * tracklet.frame_count
        self.weight += tracklet.frame_count
        self.extend_span(tracklet)
        self._touch(tracklet)

    def extend_span(self, tracklet: Tracklet) -> None:
        spans = self.spans.setdefault(tracklet.camera_id, [])
        for i, (tid, first, last) in enumerate(spans):
            if tid == tracklet.track_id:
                spans[i] = (tid, min(first, tracklet.first_seen), max(last, tracklet.last_seen))
                return
        spans.append((tracklet.track_id, tracklet.first_seen, tracklet.last_seen))

    def _touch(self, tracklet: Tracklet) -> None:
        # Recency only advances: a late-arriving segment must not rewind the
        # window every future candidate is measured against.
        if tracklet.last_seen >= self.last_updated:
            self.last_camera = tracklet.camera_id
            self.last_updated = tracklet.last_seen

    def detach_member(
        self, camera_id: str, track_id: int, vector_sum_part: np.ndarray, weight_part: int
    ) -> dict[str, Any]:
        """Take one member's evidence out (for re-scoring it against every
        identity, this one included, without it competing against itself).
        Returns a snapshot ``restore`` puts back verbatim."""
        snap = {
            "vector_sum": self.vector_sum.copy(),
            "weight": self.weight,
            "members": list(self.members),
            "spans": {c: list(v) for c, v in self.spans.items()},
            "last_camera": self.last_camera,
            "last_updated": self.last_updated,
        }
        self.vector_sum = self.vector_sum - vector_sum_part
        self.weight = max(0, self.weight - int(weight_part))
        self.members = [m for m in self.members if m != (camera_id, track_id)]
        if camera_id in self.spans:
            self.spans[camera_id] = [s for s in self.spans[camera_id] if s[0] != track_id]
        return snap

    def restore(self, snap: dict[str, Any]) -> None:
        self.vector_sum = snap["vector_sum"]
        self.weight = snap["weight"]
        self.members = snap["members"]
        self.spans = snap["spans"]
        self.last_camera = snap["last_camera"]
        self.last_updated = snap["last_updated"]


@dataclass
class GlobalIDStats:
    assign_calls: int = 0
    minted: int = 0
    matched: int = 0
    accumulated: int = 0
    revised: int = 0
    redelivered: int = 0


class GlobalIDService:
    """Shared by every camera worker; ``assign`` is the only entry point.

    Args:
        similarity_threshold: cosine a tracklet must reach to join an identity.
        reidentify_within_sec: an identity last seen longer ago than this is
            not a candidate — the person is treated as new.
        revise_at_loss: when a track that already has an id delivers more
            frames, re-score its FULL evidence against every identity and move
            it if another wins by ``revise_margin``.
        revise_margin: see above.
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        reidentify_within_sec: float = DEFAULT_REIDENTIFY_WITHIN_SEC,
        revise_at_loss: bool = True,
        revise_margin: float = DEFAULT_REVISE_MARGIN,
    ) -> None:
        self._similarity_threshold = float(similarity_threshold)
        self._reidentify_within_sec = float(reidentify_within_sec)
        self._revise_at_loss = bool(revise_at_loss)
        self._revise_margin = float(revise_margin)
        self._identities: dict[int, Identity] = {}
        self._next_id = 1
        # (camera_id, track_id) → global_id, and the bookkeeping that lets a
        # later segment of the same track be recognised and re-scored.
        self._assigned: dict[tuple[str, int], int] = {}
        self._last_segment: dict[tuple[str, int], tuple[datetime, datetime, int]] = {}
        self._evidence: dict[tuple[str, int], tuple[np.ndarray, int, datetime]] = {}
        self.stats = GlobalIDStats()

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "GlobalIDService":
        """Build from the ``global_id:`` block of config/tracking.yaml."""
        cfg = cfg or {}
        revise = cfg.get("revise_at_loss") or {}
        return cls(
            similarity_threshold=cfg.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD),
            reidentify_within_sec=cfg.get("reidentify_within_sec", DEFAULT_REIDENTIFY_WITHIN_SEC),
            revise_at_loss=revise.get("enabled", True),
            revise_margin=revise.get("margin", DEFAULT_REVISE_MARGIN),
        )

    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    @property
    def identities(self) -> dict[int, Identity]:
        return self._identities

    def global_id_of(self, camera_id: str, track_id: int) -> int | None:
        return self._assigned.get((camera_id, track_id))

    # ------------------------------------------------------------------

    def assign(self, tracklet: Tracklet) -> int:
        """Return the global id for ``tracklet``, minting one if nothing matches."""
        emb = l2_normalize(tracklet.embedding)
        key = (tracklet.camera_id, tracklet.track_id)
        sig = (tracklet.first_seen, tracklet.last_seen, tracklet.frame_count)
        self.stats.assign_calls += 1

        if key in self._assigned:
            return self._assign_existing(key, tracklet, emb, sig)

        matched = self._best_match(tracklet, emb)
        if matched is not None:
            ident, sim = matched
            ident.add_member(tracklet, emb)
            self.stats.matched += 1
            logger.info(
                "global_id matched gid=%d cam=%s track=%d sim=%.3f cameras=%s",
                ident.global_id, tracklet.camera_id, tracklet.track_id, sim, ident.cameras,
            )
        else:
            ident = self._mint(tracklet, emb)
            self.stats.minted += 1
            logger.info(
                "global_id new gid=%d cam=%s track=%d frames=%d",
                ident.global_id, tracklet.camera_id, tracklet.track_id, tracklet.frame_count,
            )
        self._assigned[key] = ident.global_id
        self._last_segment[key] = sig
        self._evidence[key] = (emb * tracklet.frame_count, tracklet.frame_count, tracklet.first_seen)
        return ident.global_id

    def _assign_existing(
        self,
        key: tuple[str, int],
        tracklet: Tracklet,
        emb: np.ndarray,
        sig: tuple[datetime, datetime, int],
    ) -> int:
        gid = self._assigned[key]
        ident = self._identities[gid]
        prior = self._last_segment[key]
        # Only a segment that starts strictly after the previous one ended is
        # new evidence. The same segment again, or one overlapping what was
        # already counted, must not be folded in twice.
        if sig == prior or tracklet.first_seen <= prior[1]:
            ident.extend_span(tracklet)
            self.stats.redelivered += 1
            logger.debug(
                "global_id re-delivery gid=%d cam=%s track=%d (no new evidence)",
                gid, tracklet.camera_id, tracklet.track_id,
            )
            return gid

        if self._revise_at_loss:
            moved = self._maybe_revise(key, gid, ident, tracklet, emb, sig)
            if moved is not None:
                return moved

        ident.add_segment(tracklet, emb)
        self._last_segment[key] = sig
        ev_sum, ev_w, ev_first = self._evidence[key]
        self._evidence[key] = (ev_sum + emb * tracklet.frame_count, ev_w + tracklet.frame_count, ev_first)
        self.stats.accumulated += 1
        logger.info(
            "global_id accumulated gid=%d cam=%s track=%d +%d frames (weight=%d)",
            gid, tracklet.camera_id, tracklet.track_id, tracklet.frame_count, ident.weight,
        )
        return gid

    def _maybe_revise(
        self,
        key: tuple[str, int],
        gid: int,
        ident: Identity,
        tracklet: Tracklet,
        emb: np.ndarray,
        sig: tuple[datetime, datetime, int],
    ) -> int | None:
        """Re-score the track's FULL evidence with the track detached from its
        identity. Returns the new gid if another identity wins by the margin,
        else None (caller accumulates as usual)."""
        ev_sum, ev_w, ev_first = self._evidence[key]
        full_sum = ev_sum + emb * tracklet.frame_count
        full_w = ev_w + tracklet.frame_count
        full_emb = l2_normalize(full_sum)
        full_view = replace(tracklet, first_seen=ev_first, frame_count=full_w)

        snap = ident.detach_member(tracklet.camera_id, tracklet.track_id, ev_sum, ev_w)
        current_sim = float(np.dot(ident.centroid, full_emb)) if ident.weight > 0 else None
        matched = self._best_match(full_view, full_emb)
        if (
            matched is None
            or matched[0].global_id == gid
            or (current_sim is not None and matched[1] < current_sim + self._revise_margin)
        ):
            ident.restore(snap)
            return None

        target, target_sim = matched
        target.add_member(full_view, full_emb)
        if ident.weight <= 0 or not ident.members:
            del self._identities[gid]
        self._assigned[key] = target.global_id
        self._last_segment[key] = sig
        self._evidence[key] = (full_sum, full_w, ev_first)
        self.stats.revised += 1
        logger.info(
            "global_id revised cam=%s track=%d gid %d -> %d (sim %.3f vs %s, frames=%d)",
            tracklet.camera_id, tracklet.track_id, gid, target.global_id, target_sim,
            f"{current_sim:.3f}" if current_sim is not None else "n/a", full_w,
        )
        return target.global_id

    def _best_match(self, tracklet: Tracklet, emb: np.ndarray) -> tuple[Identity, float] | None:
        best: tuple[Identity, float] | None = None
        for ident in self._identities.values():
            if ident.weight <= 0:
                continue
            dt = (tracklet.first_seen - ident.last_updated).total_seconds()
            if dt > self._reidentify_within_sec:
                continue
            if ident.contains_concurrent(tracklet.camera_id, tracklet.concurrent_track_ids):
                continue
            if ident.overlaps_on_camera(tracklet.camera_id, tracklet.first_seen, tracklet.last_seen):
                continue
            sim = float(np.dot(ident.centroid, emb))
            if sim < self._similarity_threshold:
                continue
            if best is None or sim > best[1] or (sim == best[1] and ident.global_id < best[0].global_id):
                best = (ident, sim)
        return best

    def _mint(self, tracklet: Tracklet, emb: np.ndarray) -> Identity:
        ident = Identity(
            global_id=self._next_id,
            vector_sum=emb * tracklet.frame_count,
            weight=tracklet.frame_count,
            members=[(tracklet.camera_id, tracklet.track_id)],
            last_camera=tracklet.camera_id,
            last_updated=tracklet.last_seen,
            spans={tracklet.camera_id: [(tracklet.track_id, tracklet.first_seen, tracklet.last_seen)]},
        )
        self._identities[ident.global_id] = ident
        self._next_id += 1
        return ident

    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """JSON-ready view: parameters, counters, and every identity's members."""
        identities = []
        for ident in sorted(self._identities.values(), key=lambda i: i.global_id):
            firsts = [first for sp in ident.spans.values() for _, first, _ in sp]
            lasts = [last for sp in ident.spans.values() for _, _, last in sp]
            identities.append(
                {
                    "global_id": ident.global_id,
                    "cameras": ident.cameras,
                    "members": [{"camera_id": c, "track_id": t} for c, t in ident.members],
                    "frames": ident.weight,
                    "first_seen": min(firsts).isoformat() if firsts else None,
                    "last_seen": max(lasts).isoformat() if lasts else None,
                }
            )
        return {
            "similarity_threshold": self._similarity_threshold,
            "reidentify_within_sec": self._reidentify_within_sec,
            "revise_at_loss": {"enabled": self._revise_at_loss, "margin": self._revise_margin},
            "n_identities": len(identities),
            "n_multi_camera": sum(len(i["cameras"]) > 1 for i in identities),
            **asdict(self.stats),
            "identities": identities,
        }
