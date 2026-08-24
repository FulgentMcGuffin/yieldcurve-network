"""Tests for percent-versus-decimals detection.

This is the highest-consequence thing that can go wrong. The yield adjustment
scales with the *square* of factor volatility, so misreading percent as decimals
inflates the correction by a factor of 10,000 -- enough to swamp the yields
entirely while still producing plausible-looking finite numbers.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ycn.analysis.af_models import ModelSpec, ResidualModel, detect_yield_scale
from ycn.analysis.residual_networks import residual_cube

from conftest import ISSUERS, MATURITY_YEARS, TERM_LABELS, _long_frame


def test_decimal_panel_needs_no_rescaling():
    rng = np.random.default_rng(1)
    assert detect_yield_scale(0.022 + 0.005 * rng.standard_normal(500)) == 1.0


def test_percent_panel_is_rescaled():
    rng = np.random.default_rng(2)
    assert detect_yield_scale(2.2 + 0.5 * rng.standard_normal(500)) == 0.01


def test_partly_negative_percent_panel_is_still_percent():
    """A curve straddling zero must not be mistaken for decimals.

    Euro-area issuers in 2020-21 quoted yields from about -0.6% to +1%, in
    percent. Their *median* absolute yield is below the detection threshold, so
    a median-based test classified them as decimals while their neighbours were
    classified correctly -- and the two then received corrections differing by
    four orders of magnitude.
    """
    curve = np.linspace(-0.6, 1.0, 400)  # percent
    assert detect_yield_scale(curve) == 0.01


def test_near_zero_decimal_panel_is_still_decimals():
    """The converse: genuinely small decimal yields must not become percent."""
    curve = np.linspace(-0.006, 0.010, 400)  # decimals
    assert detect_yield_scale(curve) == 1.0


def test_empty_input_is_safe():
    assert detect_yield_scale(np.array([])) == 1.0
    assert detect_yield_scale(np.array([np.nan, np.nan])) == 1.0


def _percent_panel_with_one_negative_issuer() -> pl.DataFrame:
    """Percent-quoted panel where one issuer sits near or below zero."""
    frame = _long_frame(80, ISSUERS, TERM_LABELS, MATURITY_YEARS).with_columns(
        (pl.col("rate") * 100.0).alias("rate")  # decimals -> percent
    )
    return frame.with_columns(
        pl.when(pl.col("source") == "alpha")
        .then(pl.col("rate") - 2.4)  # push this issuer's curve around zero
        .otherwise(pl.col("rate"))
        .alias("rate")
    )


def test_one_scale_is_applied_across_the_whole_panel(curve_panel):
    """Every issuer gets the same unit decision, including the negative one.

    The unit is a property of the table. Deciding per issuer means a curve that
    happens to sit near zero is scaled differently from its neighbours, and the
    resulting residual cube mixes two incompatible unit systems.
    """
    frame = _percent_panel_with_one_negative_issuer()

    fits: dict = {}
    residual_cube(
        frame,
        curve_panel,
        "date",
        model=ModelSpec(ResidualModel.AFNS, n_refit=1),
        fits_out=fits,
    )

    scales = {fit.diagnostics["yield_scale"] for fit in fits.values()}
    assert scales == {0.01}, f"issuers disagreed about units: {scales}"


def test_volatility_estimates_stay_comparable_across_issuers(curve_panel):
    """A consequence worth asserting directly: no issuer is 100x out.

    With per-issuer detection the misclassified issuer's volatility came out two
    orders of magnitude above its neighbours'. Comparable volatilities are the
    observable symptom of a coherent unit decision.
    """
    frame = _percent_panel_with_one_negative_issuer()

    fits: dict = {}
    residual_cube(
        frame,
        curve_panel,
        "date",
        model=ModelSpec(ResidualModel.AFNS, n_refit=1),
        fits_out=fits,
    )

    level_vols = np.array([np.diag(fit.sigma)[0] for fit in fits.values()])
    assert np.all(level_vols > 0.0)
    assert level_vols.max() / level_vols.min() < 10.0


def test_explicit_scale_overrides_detection(curve_panel):
    """An explicit yield_scale is respected rather than re-detected."""
    frame = _percent_panel_with_one_negative_issuer()

    fits: dict = {}
    residual_cube(
        frame,
        curve_panel,
        "date",
        model=ModelSpec(ResidualModel.AFNS, n_refit=1, yield_scale=1.0),
        fits_out=fits,
    )

    assert {fit.diagnostics["yield_scale"] for fit in fits.values()} == {1.0}


def test_residuals_come_back_in_the_input_units(curve_panel):
    """Rescaling is internal: residuals are returned in whatever came in.

    Otherwise the cube would not be comparable with the legacy Nelson-Siegel
    path, which does no rescaling at all.
    """
    decimals = _long_frame(80, ISSUERS, TERM_LABELS, MATURITY_YEARS)
    percent = decimals.with_columns((pl.col("rate") * 100.0).alias("rate"))

    spec = ModelSpec(ResidualModel.AFNS, n_refit=1)
    _, _, in_decimals, _, _ = residual_cube(decimals, curve_panel, "date", model=spec)
    _, _, in_percent, _, _ = residual_cube(percent, curve_panel, "date", model=spec)

    # Same panel, different units in and out; the ratio is the unit ratio.
    assert np.allclose(in_percent, in_decimals * 100.0, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_adjustment_is_unit_consistent(curve_panel, scale):
    """The correction is quoted in the same units as the yields it corrects."""
    frame = _long_frame(80, ISSUERS, TERM_LABELS, MATURITY_YEARS)
    if scale != 1.0:
        frame = frame.with_columns((pl.col("rate") * 100.0).alias("rate"))

    fits: dict = {}
    residual_cube(
        frame,
        curve_panel,
        "date",
        model=ModelSpec(ResidualModel.AFNS, n_refit=1),
        fits_out=fits,
    )

    for fit in fits.values():
        # In decimals the 30-year correction is basis points; in percent it is
        # a hundred times larger. Either way it must not exceed the yields.
        assert np.max(np.abs(fit.adjustment)) < 0.5 / scale
