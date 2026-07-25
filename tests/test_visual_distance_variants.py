"""Distributional visual-distance variants (Fréchet / MMD) — hermetic numeric checks.

No gated data, no encoders — the distances are exercised directly on synthetic embedding sets with
analytically-known answers, so the maths is pinned independently of the (GPU-only) embedding pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.pipeline import (
    distributional_distance,
    frechet_distance,
    gaussian_stats,
    mmd_rbf,
)


def test_frechet_is_squared_mean_gap_when_covariances_match() -> None:
    """Equal covariances → the trace term vanishes, leaving ||mu1 - mu2||^2."""
    cov = np.eye(2)
    d = frechet_distance(np.array([0.0, 0.0]), cov, np.array([3.0, 4.0]), cov)
    assert d == pytest.approx(25.0, abs=1e-3)


def test_frechet_trace_term_for_scaled_covariance() -> None:
    """Tr(I) + Tr(4I) - 2 Tr(sqrt(4I)) = 2 + 8 - 2*4 = 2, with means equal."""
    mu = np.zeros(2)
    assert frechet_distance(mu, np.eye(2), mu, 4.0 * np.eye(2)) == pytest.approx(2.0, abs=1e-3)


def test_frechet_zero_for_identical_distributions() -> None:
    """A distribution against itself is ~0 (up to the stabilising eps)."""
    x = np.random.default_rng(0).normal(size=(200, 8))
    mu, cov = gaussian_stats(x)
    assert frechet_distance(mu, cov, mu, cov) == pytest.approx(0.0, abs=1e-3)


def test_mmd_zero_for_identical_sets_and_positive_when_separated() -> None:
    """Biased RBF-MMD is exactly 0 for an identical set and clearly positive for a shifted one."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 8))
    assert mmd_rbf(x, x, seed=0) == pytest.approx(0.0, abs=1e-9)
    y = rng.normal(loc=10.0, size=(100, 8))
    assert mmd_rbf(x, y, seed=0) > 0.1


def test_distributional_distance_dispatch_and_guards() -> None:
    """Both variants are positive for shifted sets; too-few-vectors → NaN; unknown variant → error."""
    rng = np.random.default_rng(1)
    target = rng.normal(size=(50, 8))
    reference = rng.normal(loc=1.0, size=(300, 8))
    assert distributional_distance(target, reference, "frechet", seed=0) > 0
    assert distributional_distance(target, reference, "mmd", seed=0) > 0
    assert np.isnan(distributional_distance(target[:1], reference, "frechet"))
    with pytest.raises(ValueError):
        distributional_distance(target, reference, "cosine")


def test_reference_subsampling_is_deterministic() -> None:
    """The ref_cap subsample is seeded, so repeated calls agree exactly (reproducible feature)."""
    rng = np.random.default_rng(2)
    target = rng.normal(size=(40, 8))
    reference = rng.normal(loc=0.5, size=(9000, 8))
    a = distributional_distance(target, reference, "frechet", seed=7, ref_cap=1000)
    b = distributional_distance(target, reference, "frechet", seed=7, ref_cap=1000)
    assert a == b
