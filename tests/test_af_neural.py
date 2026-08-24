"""Tests for the experimental neural residual model.

Skipped entirely unless the ``neural`` extra is installed. These are smoke and
contract tests, not accuracy tests: the source paper publishes no code and no
hyperparameters, so there is no published number to check against, and asserting
one would be inventing a benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest

from ycn.analysis.af_loadings import (
    afns_loadings,
    afns_yield_adjustment,
    decay_to_lambda,
)
from ycn.analysis.af_models import ModelSpec, NeuralSpec, ResidualModel, fit_panel
from ycn.analysis.af_simulate import TERM_LABELS
from ycn.analysis.af_neural import adjustment_shapes

torch = pytest.importorskip("torch", reason="needs `uv sync --extra neural`")

pytestmark = pytest.mark.neural

FAST = NeuralSpec(hidden_size=16, n_layers=1, epochs=40, learning_rate=5e-3)


def _panel(n_dates: int, maturities: np.ndarray, seed: int = 3):
    rng = np.random.default_rng(seed)
    design = afns_loadings(maturities, 1.0)
    factors = np.array([0.022, -0.012, 0.010]) + np.cumsum(
        rng.normal(0.0, 3e-4, size=(n_dates, 3)), axis=0
    )
    return factors @ design.T + rng.normal(0.0, 1.2e-4, size=(n_dates, len(maturities)))


def test_adjustment_shapes_reconstruct_the_closed_form(maturities):
    """The linear decomposition equals the closed form it is derived from.

    The training loop multiplies these shapes by squared volatilities instead of
    evaluating the closed form, so if the two ever diverge the network would be
    optimising against a different model than the one being compared.
    """
    shapes = adjustment_shapes(maturities, 1.0)
    sigma = np.array([0.007, 0.011, 0.025])

    reconstructed = shapes @ sigma**2
    expected = afns_yield_adjustment(maturities, decay_to_lambda(1.0), np.diag(sigma))

    assert np.allclose(reconstructed, expected, rtol=1e-12, atol=1e-18)


def test_adjustment_shapes_are_non_positive(maturities):
    """Each factor's contribution is a discount, never a premium."""
    shapes = adjustment_shapes(maturities, 1.0)
    assert np.all(shapes <= 0.0)
    assert shapes.shape == (len(maturities), 3)


@pytest.mark.slow
def test_neural_fit_produces_the_expected_contract(maturities):
    """A short training run returns a well-formed PanelFit."""
    yields = _panel(150, maturities)
    fit = fit_panel(
        maturities,
        yields,
        ModelSpec(ResidualModel.NEURAL_HJM, decay=1.0, neural=FAST, seed=0),
    )

    assert fit.residuals.shape == yields.shape
    assert fit.factors.shape == (len(yields), 3)
    assert fit.adjustment.shape == (len(maturities),)
    assert np.all(np.isfinite(fit.residuals))
    assert np.all(np.isfinite(fit.factors))


@pytest.mark.slow
def test_neural_fit_declares_itself_experimental_and_filtered(maturities):
    """The caveats travel with the output, not just the documentation."""
    yields = _panel(150, maturities)
    fit = fit_panel(
        maturities,
        yields,
        ModelSpec(ResidualModel.NEURAL_HJM, decay=1.0, neural=FAST, seed=0),
    )

    assert fit.diagnostics["experimental"] is True
    assert fit.diagnostics["filtered_residuals"] is True
    assert fit.diagnostics["estimator"] == "neural_hjm"


@pytest.mark.slow
def test_training_reduces_the_loss(maturities):
    """The optimiser makes progress; a flat loss would mean nothing is learning."""
    yields = _panel(150, maturities)
    fit = fit_panel(
        maturities,
        yields,
        ModelSpec(ResidualModel.NEURAL_HJM, decay=1.0, neural=FAST, seed=0),
    )

    assert fit.diagnostics["loss_decreased"] is True
    assert fit.diagnostics["final_loss"] < fit.diagnostics["initial_loss"]


@pytest.mark.slow
def test_fit_is_reproducible_for_a_fixed_seed(maturities):
    """Same seed, same answer -- otherwise no comparison is repeatable."""
    yields = _panel(120, maturities)
    spec = ModelSpec(ResidualModel.NEURAL_HJM, decay=1.0, neural=FAST, seed=7)

    first = fit_panel(maturities, yields, spec)
    second = fit_panel(maturities, yields, spec)

    assert np.allclose(first.residuals, second.residuals, rtol=0, atol=1e-12)


def test_short_panel_falls_back_to_two_step(maturities):
    """Too little history to train returns the two-step answer instead."""
    yields = _panel(20, maturities)
    fit = fit_panel(
        maturities,
        yields,
        ModelSpec(ResidualModel.NEURAL_HJM, decay=1.0, neural=FAST),
    )

    assert fit.diagnostics["estimator"] == "two_step_fallback"
    assert fit.diagnostics["filtered_residuals"] is False
    assert np.all(np.isfinite(fit.residuals))


@pytest.mark.slow
def test_learned_volatility_is_positive_and_bounded(maturities):
    """Volatilities stay in the clamped range, so the adjustment stays sane."""
    yields = _panel(150, maturities)
    fit = fit_panel(
        maturities,
        yields,
        ModelSpec(ResidualModel.NEURAL_HJM, decay=1.0, neural=FAST, seed=0),
    )

    diagonal = np.diag(fit.sigma)
    assert np.all(diagonal > 0.0)
    assert np.all(diagonal < 1.0)
    assert np.all(fit.adjustment <= 0.0)


@pytest.mark.slow
def test_adjustment_stays_economically_plausible(maturities):
    """The correction must be basis points, not percentage points.

    Regression guard. With a permissive volatility clamp and no sensible
    initialisation the network parks the volatility at its ceiling, which drives
    the convexity term to hundreds of percentage points and destroys the fit --
    while the loss still decreases, so a decreasing-loss test alone does not
    catch it.
    """
    yields = _panel(150, maturities)
    fit = fit_panel(
        maturities,
        yields,
        ModelSpec(ResidualModel.NEURAL_HJM, decay=1.0, neural=FAST, seed=0),
    )

    assert np.max(np.abs(fit.adjustment)) < 0.05, "adjustment exceeds 500bp"
    assert np.all(np.diag(fit.sigma) < 0.05), "implausible factor volatility"


@pytest.mark.slow
def test_neural_fit_captures_convexity_nelson_siegel_cannot():
    """On data with real convexity, the trained model beats the plain baseline.

    Note this must be tested on *AFNS-generated* data. On Nelson-Siegel data
    there is no convexity to find, plain Nelson-Siegel is the correct model, and
    nothing should beat it -- so a win there would be evidence of overfitting
    rather than of learning.

    Even here this is not a fair benchmark: these are filtered in-sample
    residuals, optimistically small by construction. It is a floor test.
    """
    import polars as pl

    from ycn.analysis.af_simulate import simulate_afns_panel

    frame, truth = simulate_afns_panel(n_dates=300, seed=71)
    wide = (
        frame.filter(pl.col("source") == "core_1")
        .pivot(on="term", index="date", values="rate")
        .sort("date")
    )
    yields = wide.select(list(TERM_LABELS)).to_numpy()
    common = {"decay": truth.decay, "n_refit": 1}

    ns = fit_panel(truth.maturities, yields, ModelSpec(ResidualModel.NS, **common))
    neural = fit_panel(
        truth.maturities,
        yields,
        ModelSpec(
            ResidualModel.NEURAL_HJM,
            neural=NeuralSpec(hidden_size=16, epochs=400, learning_rate=1e-2),
            seed=0,
            **common,
        ),
    )

    ns_rmse = float(np.sqrt(np.mean(ns.residuals**2)))
    neural_rmse = float(np.sqrt(np.mean(neural.residuals**2)))
    assert neural_rmse < ns_rmse, f"neural {neural_rmse:.2e} vs NS {ns_rmse:.2e}"
