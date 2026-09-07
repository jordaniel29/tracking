"""fusion/percam_norm.py — per-camera mean-centering, and its wiring into GlobalIDService."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from pia_tracking.fusion import GlobalIDService, PerCameraNormalizer, Tracklet, l2_normalize
from pia_tracking.fusion.percam_norm import center

T0 = datetime(2000, 1, 1, tzinfo=timezone.utc)


def unit(i: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


def tl(cam: str, tid: int, emb: np.ndarray, t0: float, t1: float, n: int = 40) -> Tracklet:
    return Tracklet(cam, tid, emb, T0 + timedelta(seconds=t0), T0 + timedelta(seconds=t1), n)


# ── the normalizer itself ────────────────────────────────────────────────────

def test_first_observation_passes_through_almost_unchanged():
    norm = PerCameraNormalizer(prior_weight=200.0)
    e = l2_normalize(unit(0) + 0.3 * unit(1))
    out = norm.prepare("cam1", e, weight=10).centered
    # shrink = 10/(10+200) ≈ 0.048: barely centred, still unit norm
    assert float(out @ e) > 0.99 and abs(float(np.linalg.norm(out)) - 1.0) < 1e-5


def test_centering_strengthens_with_evidence_and_removes_shared_component():
    norm = PerCameraNormalizer(prior_weight=10.0)
    bias = unit(0)
    # everyone on cam1 shares direction 0; people differ in direction 1 vs 2
    a = l2_normalize(bias + 0.4 * unit(1))
    b = l2_normalize(bias + 0.4 * unit(2))
    raw_sim = float(a @ b)
    for _ in range(20):  # a lot of evidence → shrink → ~1
        norm.observe_and_normalize("cam1", a, weight=50)
        norm.observe_and_normalize("cam1", b, weight=50)
    ca, cb = norm.prepare("cam1", a, 50).centered, norm.prepare("cam1", b, 50).centered
    assert float(ca @ cb) < raw_sim - 0.5  # the shared bias is gone; they now look different
    assert norm.prepare("cam1", a, 50).shrink > 0.99


def test_prepare_is_pure_and_commit_is_once():
    norm = PerCameraNormalizer(prior_weight=5.0)
    e = unit(3)
    p1 = norm.prepare("cam1", e, 10)
    p2 = norm.prepare("cam1", e, 10)
    assert np.allclose(p1.centered, p2.centered) and norm.evidence("cam1") == 0.0
    p1.commit()
    p1.commit()  # idempotent
    assert norm.evidence("cam1") == 10.0
    assert norm.cameras() == ["cam1"]


def test_pending_center_matches_prepare_transform():
    norm = PerCameraNormalizer(prior_weight=5.0)
    norm.observe_and_normalize("cam1", unit(0), 30)
    p = norm.prepare("cam1", unit(1), 10)
    other = l2_normalize(unit(1) + 0.2 * unit(0))
    # same (shrink, mu) applied to another vector of the same tracklet
    assert np.allclose(p.center(other), center(other, p._mu, p.shrink))


def test_degenerate_vector_backs_off_instead_of_emitting_zero():
    mu = unit(0)
    out = center(unit(0), mu, shrink=1.0)  # emb == mu → emb - mu == 0
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5 and float(out @ unit(0)) > 0.99


def test_prior_weight_must_be_positive():
    with pytest.raises(ValueError):
        PerCameraNormalizer(prior_weight=0.0)


# ── inside the service ───────────────────────────────────────────────────────

def test_service_from_config_and_summary():
    svc = GlobalIDService.from_config({"percam_norm": {"enabled": True, "prior_weight": 50}})
    assert svc.percam_norm is not None and svc.percam_norm.prior_weight == 50.0
    assert GlobalIDService.from_config({}).percam_norm is None
    with pytest.raises(ValueError):
        GlobalIDService.from_config({"percam_norm": True})
    svc.assign(tl("cam1", 1, unit(0), 0, 2))
    s = svc.summary()["percam_norm"]
    assert s["enabled"] and s["frames_per_camera"] == {"cam1": 40.0}


def test_service_centers_before_matching_and_commits_after():
    svc = GlobalIDService(percam_norm=True, percam_prior_weight=10.0)
    a = unit(0)
    g1 = svc.assign(tl("cam1", 1, a, 0, 2))
    assert svc.percam_norm.evidence("cam1") == 40.0  # committed after mint
    g2 = svc.assign(tl("cam2", 7, a, 1, 3))  # same person, other camera — cold start on cam2
    assert g2 == g1 and svc.percam_norm.evidence("cam2") == 40.0
    # a re-delivery of the same segment folds nothing into mu_cam
    svc.assign(tl("cam2", 7, a, 1, 3))
    assert svc.percam_norm.evidence("cam2") == 40.0
    # a disjoint segment does
    svc.assign(tl("cam2", 7, a, 4, 6, n=10))
    assert svc.percam_norm.evidence("cam2") == 50.0


def test_centering_separates_people_who_share_a_camera_bias():
    """Two DIFFERENT people on cam1 whose raw embeddings share a strong camera
    component: raw cosine clears 0.45, centred cosine does not — once cam1 has
    enough evidence for the mean to be trusted."""
    bias = unit(0)
    p1 = l2_normalize(bias + 0.35 * unit(1))
    p2 = l2_normalize(bias + 0.35 * unit(2))
    assert float(p1 @ p2) > 0.45  # raw: would merge

    raw = GlobalIDService(similarity_threshold=0.45)
    a = raw.assign(tl("cam1", 1, p1, 0, 2))
    b = raw.assign(tl("cam1", 2, p2, 10, 12))  # later, no overlap → allowed to match
    assert a == b  # raw service merges them

    centred = GlobalIDService(similarity_threshold=0.45, percam_norm=True, percam_prior_weight=5.0)
    # warm cam1's mean with earlier people carrying the same bias
    for i in range(3, 9):
        centred.assign(tl("cam1", i, l2_normalize(bias + 0.35 * unit(3 + i % 4)), 100 * i, 100 * i + 2, n=100))
    a = centred.assign(tl("cam1", 1, p1, 2000, 2002))
    b = centred.assign(tl("cam1", 2, p2, 2010, 2012))
    assert a != b  # centred service keeps them apart
