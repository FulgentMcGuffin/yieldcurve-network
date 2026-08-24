"""Tests for the decay/lambda boundary and the AFNS yield-adjustment term."""

from __future__ import annotations

import numpy as np
import pytest

from ycn.analysis.af_loadings import (
    afns_loadings,
    afns_yield_adjustment,
    decay_to_lambda,
    lambda_to_decay,
)
from ycn.analysis.yield_curve_factors import ns_basis


@pytest.mark.parametrize("decay", [0.05, 0.5, 1.0, 1.8, 2.0, 7.5])
def test_convention_round_trip(decay):
    """decay -> lambda -> decay is the identity."""
    assert lambda_to_decay(decay_to_lambda(decay)) == pytest.approx(decay, rel=1e-15)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_conversions_reject_non_positive(bad):
    """A non-positive decay or lambda is a hard error, not a NaN downstream."""
    with pytest.raises(ValueError):
        decay_to_lambda(bad)
    with pytest.raises(ValueError):
        lambda_to_decay(bad)


@pytest.mark.parametrize("decay", [0.05, 0.5, 1.0, 1.8, 2.0])
def test_afns_loadings_match_ns_basis_exactly(maturities, decay):
    """AFNS loadings ARE the Nelson-Siegel loadings, bit for bit.

    Exactness matters: it is what lets a zero-Sigma AFNS fit reproduce legacy NS
    residuals precisely rather than approximately.
    """
    expected = np.column_stack(ns_basis(maturities, decay))
    actual = afns_loadings(maturities, decay)
    assert np.array_equal(actual, expected)


def test_loadings_take_scale_not_rate(maturities):
    """`afns_loadings` must NOT be handed a rate by mistake.

    The reciprocal round-trip is inexact (1 / (1 / 1.8) != 1.8), so routing the
    decay through the rate convention and back would perturb the loadings in
    their last bits. Passing a rate where a scale belongs must therefore produce
    visibly different loadings, not silently equal ones.
    """
    decay = 1.8
    as_scale = afns_loadings(maturities, decay)
    as_rate = afns_loadings(maturities, decay_to_lambda(decay))
    assert not np.allclose(as_scale, as_rate)


def test_zero_sigma_gives_zero_adjustment(maturities):
    """No volatility means no convexity correction, so AFNS collapses to NS."""
    adj = afns_yield_adjustment(maturities, 1.0, np.zeros((3, 3)))
    assert np.array_equal(adj, np.zeros_like(maturities))


def test_diagonal_sigma_accepts_vector_or_matrix(maturities):
    """A length-3 vector is read as the diagonal of Sigma."""
    vec = np.array([0.01, 0.02, 0.03])
    assert np.allclose(
        afns_yield_adjustment(maturities, 1.0, vec),
        afns_yield_adjustment(maturities, 1.0, np.diag(vec)),
    )


def test_adjustment_is_non_positive_and_grows_with_maturity(maturities):
    """Convexity pushes arbitrage-free yields below their NS counterparts.

    The leading tau^2/6 term dominates at the long end, so the correction must
    be non-positive and monotonically larger in magnitude across the grid.
    """
    adj = afns_yield_adjustment(maturities, 1.0, np.diag([0.01, 0.02, 0.03]))
    assert np.all(adj <= 0.0)
    magnitude = np.abs(adj)
    assert np.all(np.diff(magnitude) > 0.0)


def test_adjustment_vanishes_at_short_maturity():
    """A(tau)/tau -> 0 as tau -> 0: nothing to correct over no time."""
    tiny = np.array([1e-6, 1e-4, 1e-2])
    adj = afns_yield_adjustment(tiny, 1.0, np.diag([0.01, 0.02, 0.03]))
    assert np.all(np.abs(adj) < 1e-6)
    assert abs(adj[0]) < abs(adj[-1])


def test_adjustment_scales_quadratically_with_sigma(maturities):
    """Doubling Sigma quadruples the adjustment -- it is a variance term.

    This is the guard behind the yield-unit hazard: percent-vs-decimal input is
    a factor of 100 in Sigma and therefore 10,000 in the correction.
    """
    base = np.diag([0.01, 0.02, 0.03])
    single = afns_yield_adjustment(maturities, 1.0, base)
    double = afns_yield_adjustment(maturities, 1.0, 2.0 * base)
    assert np.allclose(double, 4.0 * single, rtol=1e-12)


def test_correlated_sigma_differs_from_diagonal(maturities):
    """Off-diagonal Sigma entries reach the adjustment via the D/E/F terms."""
    diagonal = np.diag([0.01, 0.02, 0.03])
    correlated = diagonal.copy()
    correlated[1, 0] = 0.005
    correlated[2, 0] = 0.004

    assert not np.allclose(
        afns_yield_adjustment(maturities, 1.0, diagonal),
        afns_yield_adjustment(maturities, 1.0, correlated),
    )


def test_adjustment_rejects_non_positive_maturities():
    """Maturity zero would divide by zero inside the closed form."""
    with pytest.raises(ValueError, match="strictly positive"):
        afns_yield_adjustment(np.array([0.0, 1.0]), 1.0, np.eye(3) * 0.01)


def test_adjustment_rejects_bad_sigma_shape(maturities):
    """Sigma must be (3, 3) or length 3."""
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        afns_yield_adjustment(maturities, 1.0, np.eye(2))


def test_adjustment_is_finite_across_lambda_range(maturities):
    """No overflow or cancellation blow-up over the plausible lambda range."""
    for lam in (0.05, 0.2, 0.5556, 1.0, 2.0, 20.0):
        adj = afns_yield_adjustment(maturities, lam, np.diag([0.01, 0.02, 0.03]))
        assert np.all(np.isfinite(adj))
