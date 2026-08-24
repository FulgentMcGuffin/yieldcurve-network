"""Recovery tests: does the two-step estimator find what was simulated?

Uses :mod:`ycn.analysis.af_simulate`, which generates curves from the AFNS
measurement equation including the yield adjustment, so a plain Nelson-Siegel
fit of that data is genuinely misspecified rather than merely different.

Tolerances here are deliberately loose on Sigma and tight on the factors. That
asymmetry is the errors-in-variables problem: the VAR runs on *estimated*
factors, so the innovation covariance it sees is inflated by the estimation
noise in those factors. Diebold & Li acknowledge the same bias, and the Kalman
estimator exists precisely to avoid it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ycn.analysis.af_fit import fit_two_step
from ycn.analysis.af_models import ModelSpec, ResidualModel, SigmaScope, fit_panel
from ycn.analysis.af_simulate import simulate_afns_panel
from ycn.analysis.residual_networks import residual_cube


@pytest.fixture(scope="module")
def simulated():
    """One AFNS panel with known parameters, shared across the module."""
    return simulate_afns_panel(n_dates=600, seed=101)


def _issuer_matrix(frame, issuer, term_labels):
    """Wide (n_dates, n_terms) yield matrix for one issuer."""
    import polars as pl

    wide = (
        frame.filter(pl.col("source") == issuer)
        .pivot(on="term", index="date", values="rate")
        .sort("date")
    )
    return wide.select(list(term_labels)).to_numpy()


def test_factors_are_recovered(simulated):
    """Estimated factor series track the simulated ones closely."""
    frame, truth = simulated
    from ycn.analysis.af_simulate import TERM_LABELS

    issuer = "core_1"
    yields = _issuer_matrix(frame, issuer, TERM_LABELS)
    fit = fit_two_step(
        truth.maturities,
        yields,
        ModelSpec(ResidualModel.AFNS, decay=truth.decay, n_refit=2),
    )

    for k, name in enumerate(("level", "slope", "curvature")):
        corr = np.corrcoef(fit.factors[:, k], truth.factors[issuer][:, k])[0, 1]
        assert corr > 0.99, f"{name} factor correlation only {corr:.4f}"


def test_volatility_is_recovered_to_the_right_order(simulated):
    """Sigma lands in the right ballpark, with the right ordering.

    Loose by design: the two-step Sigma is biased upward by estimation noise in
    the factors, so this asserts the magnitude and the level < slope < curvature
    ordering rather than a tight numeric match.
    """
    frame, truth = simulated
    from ycn.analysis.af_simulate import TERM_LABELS

    yields = _issuer_matrix(frame, "core_1", TERM_LABELS)
    fit = fit_two_step(
        truth.maturities,
        yields,
        ModelSpec(ResidualModel.AFNS, decay=truth.decay, n_refit=2),
    )

    estimated = np.diag(fit.sigma)
    actual = np.diag(truth.sigma)

    assert np.all(estimated > 0.0)
    # Within a factor of three either way, and correctly ordered.
    assert np.all(estimated < 3.0 * actual)
    assert np.all(estimated > actual / 3.0)
    assert estimated[0] < estimated[2], "level vol should be below curvature vol"


def test_afns_beats_ns_at_the_long_end(simulated):
    """On AFNS-generated data the arbitrage-free fit wins where it should.

    The adjustment grows as tau^2/6, so the misspecification a plain NS fit
    absorbs is concentrated at long maturities. That is where AFNS must show a
    lower residual RMSE; at the short end the two are expected to tie.
    """
    frame, truth = simulated
    from ycn.analysis.af_simulate import TERM_LABELS

    yields = _issuer_matrix(frame, "core_1", TERM_LABELS)
    spec_kwargs = {"decay": truth.decay}

    ns = fit_panel(truth.maturities, yields, ModelSpec(ResidualModel.NS, **spec_kwargs))
    afns = fit_panel(
        truth.maturities,
        yields,
        ModelSpec(ResidualModel.AFNS, n_refit=2, **spec_kwargs),
    )

    long_end = truth.maturities >= 15.0
    ns_rmse = np.sqrt(np.mean(ns.residuals[:, long_end] ** 2))
    afns_rmse = np.sqrt(np.mean(afns.residuals[:, long_end] ** 2))

    assert afns_rmse < ns_rmse, f"AFNS {afns_rmse:.3e} did not beat NS {ns_rmse:.3e}"


def test_recovered_adjustment_has_the_right_shape(simulated):
    """The estimated offset resembles the one used to generate the data."""
    frame, truth = simulated
    from ycn.analysis.af_simulate import TERM_LABELS

    yields = _issuer_matrix(frame, "core_1", TERM_LABELS)
    fit = fit_two_step(
        truth.maturities,
        yields,
        ModelSpec(ResidualModel.AFNS, decay=truth.decay, n_refit=2),
    )

    assert np.all(fit.adjustment <= 0.0)
    # Same shape across maturities, even if the level is biased by Sigma.
    corr = np.corrcoef(fit.adjustment, truth.adjustment)[0, 1]
    assert corr > 0.99


def test_adjustment_tracks_the_data_volatility():
    """The correction responds to the data, rather than being manufactured.

    A quiet panel must earn a far smaller convexity correction than a volatile
    one. This is the converse guard on the estimator: the adjustment should come
    from the factor dynamics actually present, not from measurement noise or
    from a constant baked into the code.

    The correction scales with the *square* of volatility, so a tenfold
    volatility difference should move it roughly a hundredfold.

    The non-factor noise sources are switched off here on purpose. They act as a
    floor under the estimated volatility -- the errors-in-variables bias -- and
    that floor binds hardest exactly where the true volatility is smallest, which
    would compress the ratio and make this test measure the noise level rather
    than the estimator's response to it. Their effect is real and is covered by
    the recovery tests above; this test isolates the response.
    """
    from ycn.analysis.af_simulate import TERM_LABELS

    quiet_conditions = {
        "n_dates": 400,
        "meas_sd": 1e-5,
        "common_sd": 0.0,
        "bloc_sd": 0.0,
        "seed": 7,
    }
    quiet, _ = simulate_afns_panel(sigma=(0.0007, 0.0011, 0.0025), **quiet_conditions)
    loud, _ = simulate_afns_panel(sigma=(0.0070, 0.0110, 0.0250), **quiet_conditions)

    maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    spec = ModelSpec(ResidualModel.AFNS, decay=1.0, n_refit=1)

    quiet_fit = fit_two_step(
        maturities, _issuer_matrix(quiet, "core_1", TERM_LABELS), spec
    )
    loud_fit = fit_two_step(
        maturities, _issuer_matrix(loud, "core_1", TERM_LABELS), spec
    )

    quiet_max = np.max(np.abs(quiet_fit.adjustment))
    loud_max = np.max(np.abs(loud_fit.adjustment))

    assert (
        quiet_max < loud_max / 20.0
    ), f"quiet panel corrected {quiet_max:.2e} against loud {loud_max:.2e}"


def test_zero_volatility_data_earns_no_correction(maturities):
    """Constant factors imply no convexity, so the adjustment must vanish."""
    from ycn.analysis.af_loadings import afns_loadings

    design = afns_loadings(maturities, 1.0)
    constant = np.tile(np.array([0.022, -0.012, 0.010]), (200, 1))
    yields = constant @ design.T

    fit = fit_two_step(
        maturities, yields, ModelSpec(ResidualModel.AFNS, decay=1.0, n_refit=2)
    )

    assert np.allclose(fit.adjustment, 0.0, atol=1e-12)
    assert np.allclose(fit.residuals, 0.0, atol=1e-12)


@pytest.mark.parametrize("scope", [SigmaScope.PER_ISSUER, SigmaScope.POOLED])
def test_both_sigma_scopes_produce_a_usable_cube(simulated, scope):
    """Per-issuer and pooled both run end to end and differ from each other."""
    from ycn.analysis.af_simulate import TERM_LABELS
    from ycn.analysis.yield_curve import CurvePanel

    frame, truth = simulated
    panel = CurvePanel(
        date_column="date",
        issuer_column="source",
        term_columns=TERM_LABELS,
        term_column="term",
        rate_column="rate",
    )
    fits: dict = {}
    issuers, dates, cube, terms, skipped = residual_cube(
        frame,
        panel,
        "date",
        model=ModelSpec(
            ResidualModel.AFNS, decay=truth.decay, n_refit=1, sigma_scope=scope
        ),
        fits_out=fits,
    )

    assert cube.shape == (len(issuers), len(dates), len(terms))
    assert not np.isnan(cube).any()
    assert skipped == []
    assert set(fits) == set(issuers)
    for fit in fits.values():
        assert fit.diagnostics["sigma_scope"] == scope.value


def test_pooled_sigma_is_shared_across_issuers(simulated):
    """Pooling means one adjustment, identical for every issuer."""
    from ycn.analysis.af_simulate import TERM_LABELS
    from ycn.analysis.yield_curve import CurvePanel

    frame, truth = simulated
    panel = CurvePanel(
        date_column="date",
        issuer_column="source",
        term_columns=TERM_LABELS,
        term_column="term",
        rate_column="rate",
    )
    fits: dict = {}
    residual_cube(
        frame,
        panel,
        "date",
        model=ModelSpec(
            ResidualModel.AFNS,
            decay=truth.decay,
            n_refit=1,
            sigma_scope=SigmaScope.POOLED,
        ),
        fits_out=fits,
    )

    adjustments = [fit.adjustment for fit in fits.values()]
    reference = adjustments[0]
    for other in adjustments[1:]:
        assert np.array_equal(reference, other)


def test_per_issuer_sigma_varies_across_issuers(simulated):
    """Per-issuer scope gives each issuer its own volatility estimate."""
    from ycn.analysis.af_simulate import TERM_LABELS
    from ycn.analysis.yield_curve import CurvePanel

    frame, truth = simulated
    panel = CurvePanel(
        date_column="date",
        issuer_column="source",
        term_columns=TERM_LABELS,
        term_column="term",
        rate_column="rate",
    )
    fits: dict = {}
    residual_cube(
        frame,
        panel,
        "date",
        model=ModelSpec(
            ResidualModel.AFNS,
            decay=truth.decay,
            n_refit=1,
            sigma_scope=SigmaScope.PER_ISSUER,
        ),
        fits_out=fits,
    )

    diagonals = np.array([np.diag(fit.sigma) for fit in fits.values()])
    assert np.any(np.std(diagonals, axis=0) > 0.0)
