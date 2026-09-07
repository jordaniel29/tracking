"""GlobalIDService decision rules, on synthetic embeddings (no models, no GPU)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from pia_tracking.fusion import GlobalIDService, Tracklet, l2_normalize

T0 = datetime(2000, 1, 1, tzinfo=timezone.utc)


def unit(seed: int, dim: int = 16) -> np.ndarray:
    """Deterministic unit vector. Distinct seeds (mod dim) are orthogonal, so two
    'different people' never land above the 0.45 gate by chance — random 16-d
    vectors do (cosine std ≈ 0.25)."""
    v = np.zeros(dim, dtype=np.float32)
    v[seed % dim] = 1.0
    return v


def tl(cam: str, tid: int, emb: np.ndarray, t0: float, t1: float, n: int = 40, conc=()) -> Tracklet:
    return Tracklet(
        camera_id=cam,
        track_id=tid,
        embedding=emb,
        first_seen=T0 + timedelta(seconds=t0),
        last_seen=T0 + timedelta(seconds=t1),
        frame_count=n,
        concurrent_track_ids=frozenset(conc),
    )


def test_first_tracklet_mints_and_second_camera_matches():
    svc = GlobalIDService()
    a = unit(1)
    g1 = svc.assign(tl("cam8", 1, a, 0, 2))
    g2 = svc.assign(tl("cam9", 7, a, 1, 3))  # overlapping in time, other camera
    assert g1 == g2 == 1
    assert svc.identities[1].cameras == ["cam8", "cam9"]
    assert svc.stats.minted == 1 and svc.stats.matched == 1


def test_dissimilar_appearance_mints_new_identity():
    svc = GlobalIDService()
    g1 = svc.assign(tl("cam8", 1, unit(1), 0, 2))
    g2 = svc.assign(tl("cam9", 2, unit(2), 0, 2))
    assert g1 != g2
    assert svc.summary()["n_identities"] == 2


def test_recency_window_expires_identity():
    svc = GlobalIDService(reidentify_within_sec=600)
    a = unit(1)
    g1 = svc.assign(tl("cam8", 1, a, 0, 2))
    g2 = svc.assign(tl("cam9", 2, a, 500, 502))  # within window → same person
    g3 = svc.assign(tl("cam9", 3, a, 1200, 1202))  # 698 s after last sighting → new person
    assert g1 == g2
    assert g3 != g1


def test_same_camera_same_time_never_merges():
    svc = GlobalIDService()
    a = unit(1)
    g1 = svc.assign(tl("cam8", 1, a, 0, 5))
    g2 = svc.assign(tl("cam8", 2, a, 3, 8))  # identical look, on screen together → two people
    assert g1 != g2
    g3 = svc.assign(tl("cam8", 3, a, 10, 12))  # after both left → re-entry, may match
    assert g3 == g1  # ties break to the lowest id


def test_concurrent_ids_block_merge_even_with_stale_spans():
    svc = GlobalIDService()
    a = unit(1)
    g1 = svc.assign(tl("cam8", 1, a, 0, 2))
    # Spans do not overlap, but the worker saw track 1 on screen with track 2.
    g2 = svc.assign(tl("cam8", 2, a, 3, 5, conc=(1,)))
    assert g1 != g2


def test_redelivery_is_noop_and_disjoint_segment_accumulates():
    svc = GlobalIDService(revise_at_loss=False)
    a = unit(1)
    first = tl("cam8", 1, a, 0, 2, n=40)
    g1 = svc.assign(first)
    assert svc.assign(first) == g1  # exact re-delivery
    assert svc.stats.redelivered == 1 and svc.identities[g1].weight == 40
    assert svc.assign(tl("cam8", 1, a, 1, 4, n=10)) == g1  # overlaps the counted span
    assert svc.stats.redelivered == 2 and svc.identities[g1].weight == 40
    assert svc.assign(tl("cam8", 1, a, 3, 6, n=10)) == g1  # strictly after → new evidence
    assert svc.stats.accumulated == 1 and svc.identities[g1].weight == 50


def test_revise_at_loss_moves_track_to_better_identity():
    svc = GlobalIDService(revise_at_loss=True, revise_margin=0.05)
    a, b = unit(1), unit(2)
    ga = svc.assign(tl("cam1", 1, a, 0, 2, n=40))
    gb = svc.assign(tl("cam1", 2, b, 0, 2, n=40))
    assert ga != gb
    # Checkpoint on cam2: a short, a-leaning view → provisionally identity A.
    early = l2_normalize(a + 0.3 * b)
    assert svc.assign(tl("cam2", 5, early, 1, 3, n=40)) == ga
    # Loss segment: many more frames that clearly look like B → track moves.
    assert svc.assign(tl("cam2", 5, b, 4, 20, n=400)) == gb
    assert svc.stats.revised == 1
    assert ("cam2", 5) not in svc.identities[ga].members
    assert ("cam2", 5) in svc.identities[gb].members
    assert svc.global_id_of("cam2", 5) == gb


def test_revise_stays_put_without_margin():
    svc = GlobalIDService(revise_at_loss=True, revise_margin=0.05)
    a = unit(1)
    ga = svc.assign(tl("cam1", 1, a, 0, 2))
    gb = svc.assign(tl("cam1", 2, unit(2), 0, 2))
    assert svc.assign(tl("cam2", 5, a, 1, 3)) == ga
    assert svc.assign(tl("cam2", 5, a, 4, 6)) == ga  # more of the same → accumulate, no move
    assert svc.stats.revised == 0 and svc.stats.accumulated == 1
    assert gb in svc.identities


def test_summary_counts_multi_camera_identities():
    svc = GlobalIDService()
    a, b = unit(1), unit(2)
    svc.assign(tl("cam8", 1, a, 0, 2))
    svc.assign(tl("cam9", 1, a, 0, 2))
    svc.assign(tl("cam8", 2, b, 0, 2))
    s = svc.summary()
    assert s["n_identities"] == 2 and s["n_multi_camera"] == 1
    assert s["identities"][0]["members"] == [
        {"camera_id": "cam8", "track_id": 1},
        {"camera_id": "cam9", "track_id": 1},
    ]


def test_tracklet_rejects_naive_or_reversed_timestamps():
    a = unit(1)
    try:
        Tracklet("c", 1, a, T0 + timedelta(1), T0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("reversed timestamps accepted")
    try:
        Tracklet("c", 1, a, datetime(2000, 1, 1), datetime(2000, 1, 1), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("naive timestamps accepted")
