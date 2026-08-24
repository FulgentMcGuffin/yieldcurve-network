"""The anchor tests binding the arbitrage-free framework to the legacy NS path.

Two claims are pinned here, and between them they are what stop the two
frameworks drifting apart:

1. Routing Nelson-Siegel through the model registry reproduces the legacy
   per-date code path exactly.
2. AFNS with zero volatility reproduces Nelson-Siegel exactly, because the
   yield adjustment is then identically zero and the loadings are shared.

If either fails, a residual cube produced by one path is no longer comparable
with one produced by the other, and every cross-model comparison built on top
of them is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from ycn.analysis.af_models import ModelSpec, ResidualModel, fit_panel
from ycn.analysis.residual_networks import residual_cube
from ycn.analysis.yield_curve_factors import extract_ns_residuals, fit_nelson_siegel


@pytest.mark.parametrize("decay", [0.5, 1.0, 1.8])
def test_ns_registry_matches_fit_nelson_siegel(maturities, decay):
    """The registry's NS adapter is the legacy per-date fit, exactly."""
    rng = np.random.default_rng(3)
    yields = 0.02 + 0.001 * rng.standard_normal((25, len(maturities)))

    fit = fit_panel(maturities, yields, ModelSpec(ResidualModel.NS, decay=decay))

    for t in range(yields.shape[0]):
        expected = fit_nelson_siegel(maturities, yields[t], decay=decay)
        assert np.array_equal(fit.residuals[t], expected.residuals)
        assert fit.factors[t, 0] == expected.level
        assert fit.rmse[t] == expected.rmse


def test_ns_registry_adjustment_is_zero(maturities):
    """Plain NS carries no yield adjustment -- that is what makes it the baseline."""
    rng = np.random.default_rng(5)
    yields = 0.02 + 0.001 * rng.standard_normal((10, len(maturities)))
    fit = fit_panel(maturities, yields, ModelSpec(ResidualModel.NS))

    assert np.array_equal(fit.adjustment, np.zeros(len(maturities)))
    assert np.array_equal(fit.sigma, np.zeros((3, 3)))


def test_ns_panel_matches_extract_ns_residuals(maturities):
    """Panel-at-once and date-by-date extraction agree cell for cell."""
    import polars as pl

    rng = np.random.default_rng(11)
    n_dates = 40
    labels = [f"{m:g}Y" for m in maturities]
    yields = 0.02 + 0.001 * rng.standard_normal((n_dates, len(maturities)))
    wide = pl.DataFrame(
        {"date": list(range(n_dates))}
        | {label: yields[:, i] for i, label in enumerate(labels)}
    )

    legacy, _ = extract_ns_residuals(wide, date_col="date", term_cols=labels, decay=1.0)
    fit = fit_panel(maturities, yields, ModelSpec(ResidualModel.NS, decay=1.0))

    assert np.array_equal(fit.residuals, legacy)


def test_cube_via_registry_matches_legacy_cube(ns_panel_long, curve_panel):
    """`model=ModelSpec(NS)` and `model=None` produce the identical cube.

    This is the seam's backward-compatibility guarantee: adding the model
    parameter changed nothing for callers that do not pass it, and the registry
    route reaches the same numbers as the untouched path.
    """
    _, _, legacy, _, legacy_skipped = residual_cube(ns_panel_long, curve_panel, "date")
    _, _, viamodel, _, model_skipped = residual_cube(
        ns_panel_long, curve_panel, "date", model=ModelSpec(ResidualModel.NS)
    )

    assert np.array_equal(legacy, viamodel, equal_nan=True)
    assert legacy_skipped == model_skipped


def test_cube_via_registry_matches_on_ragged_panel(ragged_panel_long, curve_panel):
    """Equality holds through the ragged paths too, not just the dense one."""
    _, _, legacy, _, legacy_skipped = residual_cube(
        ragged_panel_long, curve_panel, "date"
    )
    _, _, viamodel, _, model_skipped = residual_cube(
        ragged_panel_long, curve_panel, "date", model=ModelSpec(ResidualModel.NS)
    )

    assert np.array_equal(legacy, viamodel, equal_nan=True)
    assert legacy_skipped == model_skipped


def test_fits_out_is_populated_only_on_the_model_path(ns_panel_long, curve_panel):
    """`fits_out` collects PanelFits when a model is given, and stays empty otherwise."""
    legacy_fits: dict = {}
    residual_cube(ns_panel_long, curve_panel, "date", fits_out=legacy_fits)
    assert legacy_fits == {}

    model_fits: dict = {}
    issuers, _, _, _, _ = residual_cube(
        ns_panel_long,
        curve_panel,
        "date",
        model=ModelSpec(ResidualModel.NS),
        fits_out=model_fits,
    )
    assert set(model_fits) == set(issuers)
    for fit in model_fits.values():
        assert fit.factors.shape[1] == 3
        assert fit.residuals.shape[0] == fit.factors.shape[0]


@pytest.mark.parametrize("decay", [0.5, 1.0, 1.8])
def test_afns_with_zero_volatility_reproduces_ns(maturities, decay):
    """AFNS with no volatility IS Nelson-Siegel.

    ``n_refit=0`` stops after the seed pass, where the adjustment is identically
    zero. Since the loadings are shared, the fit is then the same least-squares
    problem and the residuals must agree to within accumulation order.
    """
    rng = np.random.default_rng(17)
    yields = 0.02 + 0.001 * rng.standard_normal((30, len(maturities)))

    ns = fit_panel(maturities, yields, ModelSpec(ResidualModel.NS, decay=decay))
    afns = fit_panel(
        maturities,
        yields,
        ModelSpec(ResidualModel.AFNS, decay=decay, n_refit=0),
    )

    assert np.array_equal(afns.adjustment, np.zeros(len(maturities)))
    assert np.allclose(afns.residuals, ns.residuals, rtol=0, atol=1e-15)
    assert np.allclose(afns.factors, ns.factors, rtol=0, atol=1e-13)


def test_afns_cube_with_zero_volatility_matches_legacy(ns_panel_long, curve_panel):
    """The same equivalence, end to end through the residual cube."""
    _, _, legacy, _, _ = residual_cube(ns_panel_long, curve_panel, "date")
    _, _, afns, _, _ = residual_cube(
        ns_panel_long,
        curve_panel,
        "date",
        model=ModelSpec(ResidualModel.AFNS, n_refit=0),
    )
    assert np.allclose(legacy, afns, rtol=0, atol=1e-15, equal_nan=True)


def test_afns_refit_produces_a_nonzero_adjustment(maturities):
    """With refitting on, a real volatility estimate yields a real correction."""
    rng = np.random.default_rng(23)
    n_dates = 400
    factors = np.array([0.022, -0.012, 0.010]) + np.cumsum(
        rng.normal(0.0, 3e-4, size=(n_dates, 3)), axis=0
    )
    from ycn.analysis.af_loadings import afns_loadings

    design = afns_loadings(maturities, 1.0)
    yields = factors @ design.T + rng.normal(
        0.0, 1.2e-4, size=(n_dates, len(maturities))
    )

    fit = fit_panel(maturities, yields, ModelSpec(ResidualModel.AFNS, n_refit=1))

    assert np.any(fit.adjustment != 0.0)
    assert np.all(fit.adjustment <= 0.0)
    assert np.any(np.diag(fit.sigma) > 0.0)
    assert fit.diagnostics["n_iterations"] >= 1


def test_residual_is_observed_minus_fitted(maturities):
    """A residual means ``observed - fitted``, with the adjustment inside fitted.

    The sign here is load-bearing. The model is ``y = B'X + adjustment``, so the
    factors come from regressing ``y - adjustment``. Flipping it applies the
    convexity correction twice and fits *worse* than plain Nelson-Siegel, which
    is a silent failure -- the numbers stay plausible, they are just wrong.
    """
    from ycn.analysis.af_fit import _cross_section
    from ycn.analysis.af_loadings import afns_loadings

    rng = np.random.default_rng(29)
    design = afns_loadings(maturities, 1.0)
    yields = 0.02 + 0.001 * rng.standard_normal((15, len(maturities)))
    offset = np.linspace(-1e-3, -5e-3, len(maturities))

    factors, residuals = _cross_section(design, yields, offset)
    fitted = factors @ design.T + offset

    # Exact in real arithmetic; in float64 the two orderings differ by a few ulp
    # of the yield magnitude (~0.02, so 1 ulp is about 3.5e-18).
    assert np.allclose(residuals, yields - fitted, rtol=0, atol=1e-16)


def test_recovering_exact_afns_data_leaves_no_residual(maturities):
    """Given the true offset, a noiseless AFNS curve is fitted exactly.

    The sharpest available check on the sign convention: construct yields from
    known factors plus a known adjustment, hand the estimator that adjustment,
    and demand the factors come back and the residual vanish.
    """
    from ycn.analysis.af_fit import _cross_section
    from ycn.analysis.af_loadings import (
        afns_loadings,
        afns_yield_adjustment,
        decay_to_lambda,
    )

    rng = np.random.default_rng(31)
    design = afns_loadings(maturities, 1.0)
    offset = afns_yield_adjustment(
        maturities, decay_to_lambda(1.0), np.diag([0.007, 0.011, 0.025])
    )
    true_factors = np.array([0.022, -0.012, 0.010]) + 1e-3 * rng.standard_normal(
        (20, 3)
    )
    yields = true_factors @ design.T + offset

    factors, residuals = _cross_section(design, yields, offset)

    assert np.allclose(factors, true_factors, rtol=0, atol=1e-12)
    assert np.allclose(residuals, 0.0, rtol=0, atol=1e-14)


def test_unregistered_model_raises_clearly(monkeypatch):
    """A model with no fitter fails loudly, naming what is available.

    Every model is registered, so the branch is reached by removing one rather
    than by relying on a gap in the registry -- which would silently stop testing
    anything the moment that gap was filled.
    """
    from ycn.analysis import af_models

    rng = np.random.default_rng(1)
    maturities = np.array([1.0, 2.0, 5.0, 10.0])
    yields = 0.02 + 0.001 * rng.standard_normal((5, 4))

    pruned = {
        k: v for k, v in af_models.FITTERS.items() if k is not ResidualModel.DTAFNS
    }
    monkeypatch.setattr(af_models, "FITTERS", pruned)

    with pytest.raises(NotImplementedError, match="No fitter registered"):
        fit_panel(maturities, yields, ModelSpec(ResidualModel.DTAFNS))


def test_every_model_has_a_fitter():
    """No ResidualModel may exist without something able to fit it."""
    from ycn.analysis.af_models import FITTERS

    missing = [m.value for m in ResidualModel if m not in FITTERS]
    assert not missing, f"models with no registered fitter: {missing}"
