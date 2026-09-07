"""Per-camera embedding mean-centering, applied before cross-camera matching.

Cross-camera similarity between DIFFERENT people is dominated by shared
camera/scene bias — colour balance, exposure, background bleed, everyone on a
camera wearing dark clothes — rather than by appearance. Subtracting each
camera's running mean embedding removes most of that shared component, so the
cosine gate separates people instead of cameras.

Online and label-free: keep a frame-count-weighted running mean ``mu_cam`` of
the tracklet embeddings seen on each camera and subtract it before matching.
Early estimates are noisy (cold start), so the subtraction is shrunk toward
zero by ``w / (w + prior_weight)`` where ``w`` is the evidence (frames) seen so
far — the first tracklets pass through nearly unchanged and centering reaches
full strength as evidence accumulates. That also avoids the degenerate
first-tracklet case (``emb - mu == 0`` when ``mu`` IS that tracklet).

Two-phase handle: :meth:`PerCameraNormalizer.prepare` centres a tracklet and
returns a :class:`PendingObservation` whose ``.centered`` vector is used for
matching; the caller invokes ``.commit()`` once the assignment is final, folding
the RAW embedding into ``mu_cam``. The handle holds the raw vector itself, so
the centred one can never be folded back into the mean by mistake.

Centering shifts the similarity scale up, so its gate moves with it: pair
centering ON with ``similarity_threshold`` ~0.45 and OFF with ~0.40.
Known approximation: ``mu_cam`` drifts as evidence accumulates, so early and
late tracklets live in slightly different spaces; centroids average over it.
"""

from __future__ import annotations

import numpy as np

DEFAULT_PRIOR_WEIGHT = 200.0  # frames of evidence before centering is ~half strength
_EPS = 1e-6
_BACKOFF_STEPS = 24  # halvings tried before the last-resort raw fallback


def center(emb: np.ndarray, mu: np.ndarray, shrink: float) -> np.ndarray:
    """Unit-norm ``emb - shrink·mu``, backing the shrink off when degenerate.

    Degenerate means ``emb ≈ shrink·mu`` — the vector looks like the camera
    mean. Emitting a near-zero vector is meaningless and returning the RAW one
    would mix an un-centred vector into the centred space, so the shrink is
    halved until the result is well-defined: a partially-centred vector in the
    same space. It converges to the raw embedding only if ``mu ≈ emb`` at every
    shrink — a camera that has genuinely seen one direction, for which raw IS
    the right cold-start answer.
    """
    s = shrink
    centered = emb - s * mu
    norm = float(np.linalg.norm(centered))
    steps = 0
    while norm < _EPS and steps < _BACKOFF_STEPS:
        s *= 0.5
        centered = emb - s * mu
        norm = float(np.linalg.norm(centered))
        steps += 1
    if norm < _EPS:
        centered, norm = emb, float(np.linalg.norm(emb)) or 1.0
    return (centered / norm).astype(np.float32, copy=False)


class PendingObservation:
    """Result of :meth:`PerCameraNormalizer.prepare`: the centred vector plus a
    one-shot ``commit`` that folds the RAW embedding into ``mu_cam``.

    ``center(vec)`` applies the SAME ``(shrink, mu)`` to further vectors of the
    same tracklet, so all of one tracklet's evidence lands in one space.
    """

    __slots__ = ("centered", "_normalizer", "_camera_id", "_raw", "_weight", "_committed", "_mu", "_shrink")

    def __init__(
        self,
        centered: np.ndarray,
        normalizer: "PerCameraNormalizer",
        camera_id: str,
        raw: np.ndarray,
        weight: float,
        mu: np.ndarray,
        shrink: float,
    ) -> None:
        self.centered = centered
        self._normalizer = normalizer
        self._camera_id = camera_id
        self._raw = raw
        self._weight = weight
        self._committed = False
        self._mu = mu
        self._shrink = shrink

    @property
    def shrink(self) -> float:
        return self._shrink

    def center(self, vec: np.ndarray) -> np.ndarray:
        """Centre another vector of the SAME tracklet with the captured
        ``(shrink, mu)`` — pure, unit-norm output."""
        return center(vec, self._mu, self._shrink)

    def commit(self) -> None:
        """Fold the raw evidence into ``mu_cam`` (idempotent per handle)."""
        if self._committed:
            return
        self._committed = True
        self._normalizer._fold(self._camera_id, self._raw, self._weight)


class PerCameraNormalizer:
    """Online per-camera mean-centering of L2-normalised embeddings.

    Not thread-safe by itself; the service calls it serially. Two-phase use:
    :meth:`prepare` (pure — safe to repeat) for matching, then
    ``pending.commit()`` exactly once when the assignment is final.
    :meth:`observe_and_normalize` collapses both for callers with nothing in
    between (tests, offline scoring).
    """

    def __init__(self, prior_weight: float = DEFAULT_PRIOR_WEIGHT) -> None:
        if prior_weight <= 0.0:
            raise ValueError(f"prior_weight must be > 0, got {prior_weight}")
        self._prior = float(prior_weight)
        self._sums: dict[str, np.ndarray] = {}
        self._weights: dict[str, float] = {}

    @property
    def prior_weight(self) -> float:
        return self._prior

    def evidence(self, camera_id: str) -> float:
        """Frames folded into this camera's mean so far."""
        return self._weights.get(camera_id, 0.0)

    def cameras(self) -> list[str]:
        return sorted(self._weights)

    def _tentative(self, camera_id: str, emb: np.ndarray, w: float) -> tuple[np.ndarray, float]:
        """Sum/weight that WOULD result from folding ``(emb, w)`` — no mutation.
        Shared by prepare and _fold so the vector centred at prepare time is
        centred against exactly the mean commit stores."""
        prev = self._sums.get(camera_id)
        if prev is None:
            return emb.astype(np.float32) * w, w
        return prev + emb * w, self._weights[camera_id] + w

    def _fold(self, camera_id: str, raw: np.ndarray, weight: float) -> None:
        w = max(float(weight), 1.0)
        self._sums[camera_id], self._weights[camera_id] = self._tentative(camera_id, raw, w)

    def prepare(self, camera_id: str, emb: np.ndarray, weight: float) -> PendingObservation:
        """Centre ``emb`` against the camera mean WITHOUT mutating state.

        The mean includes this observation's own tentative contribution (the
        mean over all crops seen, as the offline prototype computed it); the
        shrink keeps that from degenerating on a camera with little history.
        """
        w = max(float(weight), 1.0)
        tentative_sum, tentative_total = self._tentative(camera_id, emb, w)
        mu = tentative_sum / tentative_total
        shrink = tentative_total / (tentative_total + self._prior)
        return PendingObservation(center(emb, mu, shrink), self, camera_id, emb, w, mu, shrink)

    def observe_and_normalize(self, camera_id: str, emb: np.ndarray, weight: float) -> np.ndarray:
        """:meth:`prepare` then commit in one call."""
        pending = self.prepare(camera_id, emb, weight)
        pending.commit()
        return pending.centered
