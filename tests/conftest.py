"""Shared fixtures for the ycn test suite.

Every panel here is built in memory as a Polars frame plus a ``CurvePanel``
constructed directly, so no test touches DuckDB or the filesystem. Curves are
generated with the same ``exp(-m/decay)`` scale convention and the same decimal
yield units as ``scripts/make_fake_par_rates.py``.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from ycn.analysis.yield_curve import CurvePanel

# Maturities in years, matching make_fake_par_rates.py's grid.
MATURITY_YEARS: tuple[float, ...] = (
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
    30.0,
)

# Term labels in the zero-padded form the real database uses (``Y000p5``).
TERM_LABELS: tuple[str, ...] = (
    "Y000p5",
    "Y001p0",
    "Y002p0",
    "Y003p0",
    "Y005p0",
    "Y007p0",
    "Y010p0",
    "Y015p0",
    "Y020p0",
    "Y030p0",
)

ISSUERS: tuple[str, ...] = ("alpha", "bravo", "charlie", "delta", "echo")

# Generator decay, in the repo's SCALE convention: loadings use exp(-m/decay).
GEN_DECAY = 1.8


def nelson_siegel(
    taus: np.ndarray, b0: float, b1: float, b2: float, decay: float = GEN_DECAY
) -> np.ndarray:
    """Nelson-Siegel curve in the repo's scale convention.

    Mirrors ``scripts/make_fake_par_rates.py`` exactly so fixtures and the
    shipped synthetic generator cannot drift apart.

    Args:
        taus: Maturities in years.
        b0: Level factor.
        b1: Slope factor.
        b2: Curvature factor.
        decay: Scale parameter; loadings use ``exp(-tau / decay)``.

    Returns:
        Yields at ``taus``, in decimals.
    """
    x = taus / decay
    slope_load = (1.0 - np.exp(-x)) / x
    curve_load = slope_load - np.exp(-x)
    return b0 + b1 * slope_load + b2 * curve_load


def _long_frame(
    n_dates: int,
    issuers: tuple[str, ...],
    term_labels: tuple[str, ...],
    maturities: tuple[float, ...],
    *,
    seed: int = 7,
    drop: dict[str, tuple[str, ...]] | None = None,
    late_start: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Build a long ``(date, source, term, rate)`` panel from NS curves.

    Args:
        n_dates: Number of business-day observations.
        issuers: Issuer names.
        term_labels: Term column labels, aligned with ``maturities``.
        maturities: Maturities in years.
        seed: RNG seed.
        drop: Per-issuer term labels to omit entirely (ragged tenor coverage).
        late_start: Per-issuer count of leading dates to omit (ragged history).

    Returns:
        Long frame with one row per (date, issuer, term).
    """
    rng = np.random.default_rng(seed)
    taus = np.asarray(maturities, dtype=float)
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_dates)]

    drop = drop or {}
    late_start = late_start or {}
    base = np.array([0.022, -0.012, 0.010])  # level, slope, curvature

    rows: list[dict[str, object]] = []
    for issuer in issuers:
        walk = np.cumsum(rng.normal(0.0, 0.00025, size=(n_dates, 3)), axis=0)
        factors = base + walk
        omitted = set(drop.get(issuer, ()))
        skip_before = late_start.get(issuer, 0)
        for t, day in enumerate(dates):
            if t < skip_before:
                continue
            b0, b1, b2 = factors[t]
            curve = nelson_siegel(taus, b0, b1, b2)
            curve = curve + rng.normal(0.0, 0.00012, size=curve.shape)
            for label, rate in zip(term_labels, curve):
                if label in omitted:
                    continue
                rows.append(
                    {"date": day, "source": issuer, "term": label, "rate": float(rate)}
                )

    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "source": pl.Utf8,
            "term": pl.Utf8,
            "rate": pl.Float64,
        },
    ).sort("date", "source", "term")


@pytest.fixture
def maturities() -> np.ndarray:
    """Maturity grid in years."""
    return np.asarray(MATURITY_YEARS, dtype=float)


@pytest.fixture
def curve_panel() -> CurvePanel:
    """Column-role assignment matching the fixture frames."""
    return CurvePanel(
        date_column="date",
        issuer_column="source",
        term_columns=TERM_LABELS,
        term_column="term",
        rate_column="rate",
    )


@pytest.fixture
def ns_panel_long() -> pl.DataFrame:
    """Dense 60-date x 5-issuer x 10-term panel, no gaps."""
    return _long_frame(60, ISSUERS, TERM_LABELS, MATURITY_YEARS)


@pytest.fixture
def ragged_panel_long() -> pl.DataFrame:
    """Deliberately ragged panel exercising every skip path.

    ``charlie`` lacks the two longest tenors, ``delta`` starts 20 days late, and
    ``echo`` quotes only three tenors so it must land in ``skipped`` rather than
    aborting the run.
    """
    return _long_frame(
        60,
        ISSUERS,
        TERM_LABELS,
        MATURITY_YEARS,
        drop={
            "charlie": ("Y020p0", "Y030p0"),
            "echo": TERM_LABELS[3:],  # leaves only 3 terms -> below MIN_TERMS_FOR_NS
        },
        late_start={"delta": 20},
    )
