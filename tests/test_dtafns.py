"""Tests for the discrete-time AFNS model.

The load-bearing test here is convergence: as the observation period shrinks,
both the DTAFNS loadings and its yield adjustment must approach their
continuous-time AFNS counterparts. Since the two are implemented independently
-- AFNS from a published closed form, DTAFNS from the discrete affine recursion
-- agreement in the limit is a genuine cross-validation of both rather than a
restatement of one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ycn.analysis.af_loadings import (
    afns_loadings,
    afns_yield_adjustment,
    decay_to_lambda,
    dtafns_loadings,
    dtafns_period_decay,
    dtafns_yield_adjustment,
)
from ycn.analysis.af_models import ModelSpec, ResidualModel, fit_panel

SIGMA = np.diag([0.007, 0.011, 0.025])
STEPS = (1.0 / 252.0, 1.0 / 2520.0, 1.0 / 25200.0)


def test_period_decay_is_in_the_unit_interval():
    """DTAFNS needs a per-period decay strictly inside (0, 1)."""
    for decay in (0.05, 0.5, 1.0, 1.8, 2.0):
        for dt in STEPS:
            lam = dtafns_period_decay(decay, dt)
            assert 0.0 < lam < 1.0


def test_period_decay_shrinks_with_the_period():
    """A shorter period means less decay per period."""
    values = [dtafns_period_decay(1.0, dt) for dt in STEPS]
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize("decay", [0.5, 1.0, 1.8])
def test_loadings_converge_to_afns(maturities, decay):
    """DTAFNS loadings approach the continuous ones as the period shrinks."""
    target = afns_loadings(maturities, decay)
    errors = [
        float(np.max(np.abs(dtafns_loadings(maturities, decay, dt) - target)))
        for dt in STEPS
    ]

    assert errors == sorted(errors, reverse=True), f"not monotone: {errors}"
    assert errors[-1] < 1e-4, f"finest step still off by {errors[-1]:.2e}"


@pytest.mark.parametrize("decay", [0.5, 1.0])
def test_adjustment_converges_to_afns(maturities, decay):
    """The discrete sum approaches the continuous integral.

    This is the cross-validation: the AFNS adjustment comes from CDR's published
    closed form, the DTAFNS one from summing the discrete affine recursion. They
    share no code, so convergence means both are right.
    """
    target = afns_yield_adjustment(maturities, decay_to_lambda(decay), SIGMA)
    errors = [
        float(
            np.max(
                np.abs(dtafns_yield_adjustment(maturities, decay, SIGMA, dt) - target)
            )
        )
        for dt in STEPS
    ]

    assert errors == sorted(errors, reverse=True), f"not monotone: {errors}"
    # At the finest step the residual difference must be small next to the
    # correction itself, which reaches roughly 77bp at thirty years.
    assert errors[-1] < 0.01 * float(np.max(np.abs(target)))


def test_adjustment_is_non_positive_and_grows_with_maturity(maturities):
    """Same qualitative shape as AFNS: a convexity discount that deepens."""
    adj = dtafns_yield_adjustment(maturities, 1.0, SIGMA, 1.0 / 252.0)
    assert np.all(adj <= 0.0)
    assert np.all(np.diff(np.abs(adj)) > 0.0)


def test_zero_volatility_gives_zero_adjustment(maturities):
    """No volatility, no correction -- so DTAFNS collapses to its loadings."""
    adj = dtafns_yield_adjustment(maturities, 1.0, np.zeros((3, 3)), 1.0 / 252.0)
    assert np.array_equal(adj, np.zeros_like(maturities))


def test_adjustment_scales_quadratically_with_sigma(maturities):
    """Doubling Sigma quadruples the correction, as for AFNS."""
    dt = 1.0 / 252.0
    single = dtafns_yield_adjustment(maturities, 1.0, SIGMA, dt)
    double = dtafns_yield_adjustment(maturities, 1.0, 2.0 * SIGMA, dt)
    assert np.allclose(double, 4.0 * single, rtol=1e-12)


def test_loadings_reject_maturities_below_one_period(maturities):
    """A bond shorter than one observation period is not representable."""
    with pytest.raises(ValueError, match="at least one observation period"):
        dtafns_loadings(maturities, 1.0, dt=1.0)


def test_level_loading_is_one(maturities):
    """The first factor is a level factor in discrete time too."""
    design = dtafns_loadings(maturities, 1.0, 1.0 / 252.0)
    assert np.allclose(design[:, 0], 1.0)


def test_dtafns_fits_through_the_registry(maturities):
    """DTAFNS runs end to end and produces a real correction."""
    rng = np.random.default_rng(41)
    n_dates = 400
    design = afns_loadings(maturities, 1.0)
    factors = np.array([0.022, -0.012, 0.010]) + np.cumsum(
        rng.normal(0.0, 3e-4, size=(n_dates, 3)), axis=0
    )
    yields = factors @ design.T + rng.normal(
        0.0, 1.2e-4, size=(n_dates, len(maturities))
    )

    fit = fit_panel(maturities, yields, ModelSpec(ResidualModel.DTAFNS, n_refit=1))

    assert fit.residuals.shape == yields.shape
    assert np.all(np.isfinite(fit.residuals))
    assert np.any(fit.adjustment != 0.0)
    assert np.all(fit.adjustment <= 0.0)
    assert fit.diagnostics["model"] == "dtafns"


def test_dtafns_and_afns_agree_on_the_same_data(maturities):
    """At a daily period the two models should reach very similar residuals.

    They are different models, so this is not exact equality -- but a daily step
    is fine enough that a large divergence would signal a bug in one of them.
    """
    rng = np.random.default_rng(43)
    n_dates = 400
    design = afns_loadings(maturities, 1.0)
    factors = np.array([0.022, -0.012, 0.010]) + np.cumsum(
        rng.normal(0.0, 3e-4, size=(n_dates, 3)), axis=0
    )
    yields = factors @ design.T + rng.normal(
        0.0, 1.2e-4, size=(n_dates, len(maturities))
    )

    afns = fit_panel(maturities, yields, ModelSpec(ResidualModel.AFNS, n_refit=1))
    dtafns = fit_panel(maturities, yields, ModelSpec(ResidualModel.DTAFNS, n_refit=1))

    scale = float(np.max(np.abs(afns.adjustment)))
    assert np.max(np.abs(afns.adjustment - dtafns.adjustment)) < 0.02 * scale
