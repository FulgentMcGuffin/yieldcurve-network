"""Contract tests for ``residual_cube`` against the unmodified NS code path.

These are written before the arbitrage-free seam is added and must keep passing
unchanged afterwards. They pin the parts of the contract every downstream
consumer relies on: cube shape, union axes, ragged-panel NaN placement, the skip
and error paths, term-label tolerance, and chunking invariance.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ycn.analysis import residual_networks as rn
from ycn.analysis.residual_networks import MIN_TERMS_FOR_NS, residual_cube
from ycn.analysis.yield_curve import CurvePanel

from conftest import ISSUERS, MATURITY_YEARS, TERM_LABELS


def test_cube_shape_and_axes(ns_panel_long, curve_panel):
    """Cube is (n_issuers, n_dates, n_terms) with sorted-union axes."""
    issuers, dates, cube, terms, skipped = residual_cube(
        ns_panel_long, curve_panel, "date"
    )

    assert cube.shape == (len(issuers), len(dates), len(terms))
    assert issuers == sorted(ISSUERS)
    assert terms == list(TERM_LABELS)  # already maturity-sorted
    assert dates == sorted(dates)
    assert skipped == []


def test_dense_panel_has_no_nans(ns_panel_long, curve_panel):
    """A panel with no gaps produces a fully populated cube."""
    _, _, cube, _, _ = residual_cube(ns_panel_long, curve_panel, "date")
    assert not np.isnan(cube).any()


def test_residuals_are_small_and_centred(ns_panel_long, curve_panel):
    """NS residuals on NS-generated data are near the injected 1.2bp noise."""
    _, _, cube, _, _ = residual_cube(ns_panel_long, curve_panel, "date")
    assert np.abs(np.nanmean(cube)) < 1e-4
    assert np.sqrt(np.nanmean(cube**2)) < 1e-3


def test_ragged_panel_nan_placement(ragged_panel_long, curve_panel):
    """Missing tenors and late starts become NaN in exactly the right cells."""
    issuers, dates, cube, terms, skipped = residual_cube(
        ragged_panel_long, curve_panel, "date"
    )

    # `echo` quotes only 3 terms, below MIN_TERMS_FOR_NS, so it is skipped.
    assert "echo" in skipped
    assert "echo" not in issuers

    # `charlie` lacks the two longest tenors -> those columns are all NaN.
    ci = issuers.index("charlie")
    for label in ("Y020p0", "Y030p0"):
        assert np.isnan(cube[ci, :, terms.index(label)]).all()
    # ...and every other column is populated.
    for label in terms:
        if label not in ("Y020p0", "Y030p0"):
            assert not np.isnan(cube[ci, :, terms.index(label)]).any()

    # `delta` starts 20 days late -> its first 20 union dates are all NaN.
    di = issuers.index("delta")
    assert np.isnan(cube[di, :20, :]).all()
    assert not np.isnan(cube[di, 20:, :]).any()


def test_skipped_issuer_does_not_abort(ragged_panel_long, curve_panel):
    """One unfittable issuer is dropped, not fatal."""
    issuers, _, _, _, skipped = residual_cube(ragged_panel_long, curve_panel, "date")
    assert len(issuers) >= 2
    assert set(issuers).isdisjoint(skipped)


def test_too_few_terms_raises(ns_panel_long, curve_panel):
    """Fewer than MIN_TERMS_FOR_NS maturities is a hard error."""
    keep = list(TERM_LABELS[: MIN_TERMS_FOR_NS - 1])
    trimmed = ns_panel_long.filter(pl.col("term").is_in(keep))
    with pytest.raises(ValueError, match="at least 4 maturities"):
        residual_cube(trimmed, curve_panel, "date")


def test_fewer_than_two_issuers_raises(ns_panel_long, curve_panel):
    """A single fitted issuer cannot form a network."""
    one = ns_panel_long.filter(pl.col("source") == "alpha")
    with pytest.raises(ValueError, match="Fewer than two issuers"):
        residual_cube(one, curve_panel, "date")


def test_fewer_than_three_dates_raises(ns_panel_long, curve_panel):
    """At least three dates must carry residuals."""
    first_two = sorted(ns_panel_long.get_column("date").unique().to_list())[:2]
    short = ns_panel_long.filter(pl.col("date").is_in(first_two))
    with pytest.raises(ValueError, match="at least 3 are"):
        residual_cube(short, curve_panel, "date")


@pytest.mark.parametrize(
    ("labels", "years"),
    [
        (("6M", "1Y", "2Y", "5Y", "10Y"), (0.5, 1.0, 2.0, 5.0, 10.0)),
        (("4W", "26W", "1Y", "5Y", "10Y"), (4 / 52, 26 / 52, 1.0, 5.0, 10.0)),
        (("30D", "180D", "1Y", "5Y", "10Y"), (30 / 365, 180 / 365, 1.0, 5.0, 10.0)),
    ],
)
def test_alternative_term_spellings(labels, years):
    """``6M``/``4W``/``30D`` survive ``_canonical_maturities``.

    ``extract_ns_residuals`` parses maturities by string surgery and would raise
    on these, so this guards the rename that ``residual_cube`` applies first.
    """
    from conftest import _long_frame

    panel = CurvePanel(
        date_column="date",
        issuer_column="source",
        term_columns=labels,
        term_column="term",
        rate_column="rate",
    )
    long = _long_frame(30, ISSUERS[:3], labels, years)
    issuers, _, cube, terms, skipped = residual_cube(long, panel, "date")

    assert skipped == []
    assert cube.shape == (3, 30, len(labels))
    assert not np.isnan(cube).any()


@pytest.mark.parametrize("chunk", [7, 10_000])
def test_chunking_invariance(ns_panel_long, curve_panel, monkeypatch, chunk):
    """Cube is byte-identical regardless of NS_CHUNK_DATES.

    The fit is per-date and stateless, so block size must not matter. This is
    the guard that protects any future two-pass refactor of the fitting loop.
    """
    _, _, baseline, _, _ = residual_cube(ns_panel_long, curve_panel, "date")

    monkeypatch.setattr(rn, "NS_CHUNK_DATES", chunk)
    _, _, chunked, _, _ = residual_cube(ns_panel_long, curve_panel, "date")

    assert np.array_equal(baseline, chunked, equal_nan=True)


def test_decay_changes_residuals(ns_panel_long, curve_panel):
    """The decay kwarg reaches the fit (guards against it being ignored)."""
    _, _, at_one, _, _ = residual_cube(ns_panel_long, curve_panel, "date", decay=1.0)
    _, _, at_two, _, _ = residual_cube(ns_panel_long, curve_panel, "date", decay=2.0)
    assert not np.allclose(at_one, at_two)


def test_progress_and_status_callbacks_fire(ns_panel_long, curve_panel):
    """Both optional callbacks are invoked, per the repo's worker contract."""
    progress_calls: list[tuple[int, int, str]] = []
    residual_cube(
        ns_panel_long,
        curve_panel,
        "date",
        progress=lambda i, n, msg: progress_calls.append((i, n, msg)),
    )
    assert progress_calls
    assert all(0 <= i <= n for i, n, _ in progress_calls)
