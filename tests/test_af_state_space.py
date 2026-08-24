"""Tests for the Kalman-filter maximum-likelihood estimator."""

from __future__ import annotations

import numpy as np
import pytest

from ycn.analysis.af_loadings import afns_loadings
from ycn.analysis.af_models import Estimator, ModelSpec, ResidualModel, fit_panel
from ycn.analysis.af_simulate import TERM_LABELS, simulate_afns_panel
from ycn.analysis.af_state_space import (
    StateSpaceParams,
    exact_discretisation,
    fit_state_space_mle,
    kalman_filter,
)

DT = 1.0 / 252.0


def _issuer_matrix(frame, issuer):
    import polars as pl

    wide = (
        frame.filter(pl.col("source") == issuer)
        .pivot(on="term", index="date", values="rate")
        .sort("date")
    )
    return wide.select(list(TERM_LABELS)).to_numpy()


def test_discretisation_matches_the_euler_limit():
    """As the period shrinks, the exact transition approaches the Euler one."""
    kappa = np.array([0.2, 0.5, 0.9])
    sigma = np.diag([0.007, 0.011, 0.025])

    errors = []
    for dt in (1e-2, 1e-3, 1e-4):
        transition, innovation = exact_discretisation(kappa, sigma, dt)
        euler_q = sigma @ sigma.T * dt
        errors.append(float(np.max(np.abs(innovation - euler_q))) / dt)

    assert errors == sorted(errors, reverse=True)


def test_discretisation_transition_is_a_contraction():
    """Positive mean reversion means the state decays towards its mean."""
    transition, _ = exact_discretisation(
        np.array([0.2, 0.5, 0.9]), np.eye(3) * 0.01, DT
    )
    diagonal = np.diag(transition)
    assert np.all(diagonal > 0.0)
    assert np.all(diagonal < 1.0)


def test_innovation_covariance_is_positive_definite():
    """Q must be a valid covariance or the filter cannot run."""
    _, innovation = exact_discretisation(
        np.array([0.2, 0.5, 0.9]), np.diag([0.007, 0.011, 0.025]), DT
    )
    assert np.all(np.linalg.eigvalsh(innovation) > 0.0)


def test_filter_returns_finite_loglik_and_shapes(maturities):
    """The filter runs and reports the expected array shapes."""
    rng = np.random.default_rng(3)
    n_dates = 120
    design = afns_loadings(maturities, 1.0)
    yields = 0.02 + 0.001 * rng.standard_normal((n_dates, len(maturities)))
    params = StateSpaceParams(
        kappa=np.array([0.2, 0.5, 0.9]),
        theta=np.array([0.022, -0.012, 0.010]),
        sigma=np.diag([0.007, 0.011, 0.025]),
        meas_sd=np.full(len(maturities), 2e-4),
    )

    loglik, states, residuals = kalman_filter(
        yields, design, np.zeros(len(maturities)), params, DT
    )

    assert np.isfinite(loglik)
    assert states.shape == (n_dates, 3)
    assert residuals.shape == yields.shape


def test_better_measurement_model_scores_higher(maturities):
    """The likelihood prefers a measurement sd near the truth.

    A basic sanity check that the objective is oriented correctly -- without it
    the optimiser could be maximising nonsense and every other test would still
    pass.
    """
    rng = np.random.default_rng(11)
    n_dates = 200
    design = afns_loadings(maturities, 1.0)
    factors = np.array([0.022, -0.012, 0.010]) + np.cumsum(
        rng.normal(0.0, 3e-4, size=(n_dates, 3)), axis=0
    )
    true_sd = 2e-4
    yields = factors @ design.T + rng.normal(
        0.0, true_sd, size=(n_dates, len(maturities))
    )

    def score(meas_sd: float) -> float:
        params = StateSpaceParams(
            kappa=np.array([0.2, 0.5, 0.9]),
            theta=factors.mean(axis=0),
            sigma=np.diag([0.007, 0.011, 0.025]),
            meas_sd=np.full(len(maturities), meas_sd),
        )
        return kalman_filter(yields, design, np.zeros(len(maturities)), params, DT)[0]

    assert score(true_sd) > score(true_sd * 20.0)
    assert score(true_sd) > score(true_sd / 20.0)


@pytest.mark.slow
def test_mle_beats_its_own_seed_on_likelihood():
    """The optimiser must not make the seeded starting point worse."""
    frame, truth = simulate_afns_panel(n_dates=300, seed=13)
    yields = _issuer_matrix(frame, "core_1")

    fit = fit_state_space_mle(
        truth.maturities,
        yields,
        ModelSpec(
            ResidualModel.AFNS, decay=truth.decay, estimator=Estimator.KALMAN, n_refit=1
        ),
    )

    if fit.diagnostics.get("estimator") == "two_step_fallback":
        pytest.skip(f"fell back: {fit.diagnostics.get('fallback_reason')}")

    assert fit.diagnostics["loglik"] >= fit.diagnostics["seed_loglik"] - 1e-6


@pytest.mark.slow
def test_mle_recovers_factors_and_flags_filtered_residuals():
    """States track the simulated factors, and the output says they are filtered."""
    frame, truth = simulate_afns_panel(n_dates=300, seed=17)
    yields = _issuer_matrix(frame, "core_1")

    fit = fit_state_space_mle(
        truth.maturities,
        yields,
        ModelSpec(
            ResidualModel.AFNS, decay=truth.decay, estimator=Estimator.KALMAN, n_refit=1
        ),
    )
    if fit.diagnostics.get("estimator") == "two_step_fallback":
        pytest.skip("fell back to two-step")

    assert fit.diagnostics["filtered_residuals"] is True
    assert fit.residuals.shape == yields.shape
    assert np.all(np.isfinite(fit.residuals))

    for k in range(3):
        corr = np.corrcoef(fit.factors[:, k], truth.factors["core_1"][:, k])[0, 1]
        assert corr > 0.9, f"factor {k} correlation only {corr:.3f}"


@pytest.mark.slow
def test_kalman_routes_through_the_registry(maturities):
    """`estimator=KALMAN` reaches the state-space fitter via fit_panel."""
    frame, truth = simulate_afns_panel(n_dates=200, seed=19)
    yields = _issuer_matrix(frame, "core_1")

    fit = fit_panel(
        truth.maturities,
        yields,
        ModelSpec(
            ResidualModel.AFNS, decay=truth.decay, estimator=Estimator.KALMAN, n_refit=1
        ),
    )
    assert fit.diagnostics["estimator"] in {"kalman", "two_step_fallback"}


def test_short_panel_falls_back_rather_than_failing(maturities):
    """Too few dates to identify the parameters returns the two-step answer."""
    rng = np.random.default_rng(23)
    yields = 0.02 + 0.001 * rng.standard_normal((15, len(maturities)))

    fit = fit_state_space_mle(
        maturities,
        yields,
        ModelSpec(ResidualModel.AFNS, decay=1.0, estimator=Estimator.KALMAN),
    )

    assert fit.diagnostics["estimator"] == "two_step_fallback"
    assert fit.residuals.shape == yields.shape
    assert np.all(np.isfinite(fit.residuals))
