"""SAM 3 familiarity proxy — hermetic tests (synthetic embeddings; no transformers, no GPU, no frames).

The metric maths and the ``_scores`` dispatch are exercised directly on planted, L2-normalised embedding sets
with analytically-known answers, so the feature is pinned independently of the (GPU-only) SAM 3 embedding pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config
from src.features.familiarity import FamiliarityProxy, mahalanobis, silhouette

# orthonormal basis vectors → three maximally-separated species prototypes.
A = np.array([1.0, 0.0, 0.0])
B = np.array([0.0, 1.0, 0.0])
C = np.array([0.0, 0.0, 1.0])


def _n(v: np.ndarray) -> np.ndarray:
    """L2-normalise (the embedders always return unit vectors)."""
    return v / np.linalg.norm(v)


def _cfg(metric: str) -> Config:
    cfg = Config()
    cfg.features.familiarity_metric = metric
    return cfg


def test_silhouette_max_for_tight_isolated_cluster() -> None:
    """A tight own cluster (intra≈0) orthogonal to the nearest other species (inter=1) → silhouette≈1."""
    tight = [A, _n(A + 1e-6 * B)]  # ~identical → intra≈0
    s = silhouette(tight, {"a": A, "b": B}, "a", exclude="a")
    assert s == pytest.approx(1.0, abs=1e-3)


def test_silhouette_lower_for_diffuse_cluster() -> None:
    """A diffuse cluster (large intra) scores strictly below a tight one at the same inter distance."""
    tight = silhouette([A, A], {"a": A, "b": B}, "a", exclude="a")
    spread = [_n(A + 0.9 * B), _n(A - 0.9 * B)]  # wide own cluster around A
    diffuse = silhouette(spread, {"a": _n(np.mean(spread, axis=0)), "b": B}, "a", exclude="a")
    assert diffuse < tight
    assert -1.0 <= diffuse <= 1.0


def test_silhouette_nan_guards() -> None:
    """Fewer than two vectors, or no other species to compare against, → NaN."""
    assert np.isnan(silhouette([A], {"a": A, "b": B}, "a", exclude="a"))  # singleton
    assert np.isnan(silhouette([A, A], {"a": A}, "a", exclude="a"))  # only self


def test_mahalanobis_finite_and_grows_with_distance() -> None:
    """Mahalanobis typicality is finite for a full pool and larger for a more atypical prototype."""
    rng = np.random.default_rng(0)
    pool = [_n(A + 0.1 * rng.normal(size=3)) for _ in range(30)]  # a cloud around A
    near = mahalanobis(A, pool)
    far = mahalanobis(C, pool)  # orthogonal to the cloud's mean
    assert np.isfinite(near) and np.isfinite(far)
    assert far > near
    assert np.isnan(mahalanobis(A, [A]))  # degenerate pool


def test_scores_nearest_prototype_is_cosine_gap_with_loso() -> None:
    """nearest_prototype = 1 - max cosine to the nearest OTHER species; leave-species-out excludes self."""
    fp = FamiliarityProxy(_cfg("nearest_prototype"))
    vecs = {"a": [A, A], "b": [B, B], "c": [_n(A + 0.2 * B), _n(A + 0.2 * B)]}  # c is near a
    s = fp._scores(vecs, vecs, ["a", "b", "c"], loso=True)
    # a's nearest other is c (cos≈0.98) not itself → small distance; never 0 (self excluded).
    assert s["a"] > 0.0
    assert s["a"] == pytest.approx(1.0 - float(A @ _n(A + 0.2 * B)), abs=1e-6)


def test_scores_dense_and_nan_contract() -> None:
    """The series is dense over probe species; missing / singleton species → NaN (silhouette)."""
    fp = FamiliarityProxy(_cfg("silhouette"))
    vecs = {"a": [A, A], "b": [B]}  # b is a singleton
    s = fp._scores(vecs, vecs, ["a", "b", "missing"], loso=True)
    assert list(s.index) == ["a", "b", "missing"]  # dense over probe species
    assert np.isfinite(s["a"])
    assert np.isnan(s["b"]) and np.isnan(s["missing"])
    assert s.name == "familiarity_proxy" and s.dtype == np.float64


def test_scores_unknown_metric_raises() -> None:
    """An unrecognised metric is a loud error, not a silent NaN column."""
    fp = FamiliarityProxy(_cfg("bogus"))
    with pytest.raises(ValueError, match="unknown familiarity_metric"):
        fp._scores({"a": [A, A]}, {"a": [A, A], "b": [B, B]}, ["a"], loso=True)
