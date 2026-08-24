"""Simulate panels from a known arbitrage-free model.

Ground truth matters here: the two-step estimator has an errors-in-variables
bias that is easy to argue about and hard to quantify without data whose true
parameters you already know. Everything in this module exists so a test can
assert that what came out resembles what went in.

Curves are generated from the AFNS measurement equation *including* the yield
adjustment, so a plain Nelson-Siegel fit of this data is genuinely misspecified
-- which is what makes the AFNS-beats-NS comparison meaningful rather than
circular.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from .af_loadings import afns_loadings, afns_yield_adjustment, decay_to_lambda

# Long-run factor means: 2.2% level, -1.2% slope, 1.0% curvature. Chosen to match
# scripts/make_fake_par_rates.py so the two generators produce comparable panels.
DEFAULT_THETA = (0.022, -0.012, 0.010)

# Annual mean-reversion rates. Level reverts slowly (near unit-root under Q, as
# AFNS requires), curvature fastest -- the usual empirical ordering.
DEFAULT_KAPPA = (0.15, 0.50, 0.90)

# Annual factor volatilities, in decimals. Within the range CDR (2011) report.
DEFAULT_SIGMA = (0.0070, 0.0110, 0.0250)

DEFAULT_MATURITIES = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)

TERM_LABELS = (
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

# Blocs and credit spreads mirroring make_fake_par_rates.py.
BLOCS: dict[str, float] = {
    "core": 0.0000,
    "semi_core": 0.0025,
    "periphery": 0.0090,
    "non_emu": 0.0015,
}


@dataclass(frozen=True)
class AfnsTruth:
    """The parameters a simulated panel was generated from."""

    decay: float  # scale convention
    theta: np.ndarray  # (3,) long-run factor means
    kappa: np.ndarray  # (3,) annual mean-reversion rates
    sigma: np.ndarray  # (3, 3) volatility matrix
    dt: float
    maturities: np.ndarray
    factors: dict[str, np.ndarray]  # per issuer, (n_dates, 3)
    adjustment: np.ndarray  # (n_terms,) the -A(tau)/tau offset used


def simulate_factors(
    n_dates: int,
    theta: np.ndarray,
    kappa: np.ndarray,
    sigma_diag: np.ndarray,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact-discretisation Ornstein-Uhlenbeck factor paths.

    Uses the closed-form transition for a diagonal mean-reversion matrix rather
    than an Euler step, so the simulated volatility matches ``sigma_diag`` at any
    ``dt`` instead of only in the small-step limit.

    Args:
        n_dates: Path length.
        theta: (3,) long-run means.
        kappa: (3,) annual mean-reversion rates, strictly positive.
        sigma_diag: (3,) annual volatilities.
        dt: Step in years.
        rng: Random source.

    Returns:
        (n_dates, 3) factor path, started at its stationary distribution.
    """
    decayed = np.exp(-kappa * dt)
    # Stationary and conditional variances of a diagonal OU process.
    stationary_var = sigma_diag**2 / (2.0 * kappa)
    step_var = stationary_var * (1.0 - decayed**2)

    factors = np.empty((n_dates, 3), dtype=float)
    factors[0] = theta + rng.normal(0.0, np.sqrt(stationary_var))
    for t in range(1, n_dates):
        factors[t] = (
            theta
            + decayed * (factors[t - 1] - theta)
            + rng.normal(0.0, np.sqrt(step_var))
        )
    return factors


def simulate_afns_panel(
    n_dates: int = 750,
    *,
    issuers: dict[str, str] | None = None,
    maturities: tuple[float, ...] = DEFAULT_MATURITIES,
    term_labels: tuple[str, ...] = TERM_LABELS,
    decay: float = 1.0,
    theta: tuple[float, ...] = DEFAULT_THETA,
    kappa: tuple[float, ...] = DEFAULT_KAPPA,
    sigma: tuple[float, ...] = DEFAULT_SIGMA,
    dt: float = 1.0 / 252.0,
    meas_sd: float = 1.2e-4,
    common_sd: float = 0.5e-4,
    bloc_sd: float = 1.5e-4,
    seed: int = 42,
) -> tuple[pl.DataFrame, AfnsTruth]:
    """Generate a long panel from the AFNS measurement equation.

    Deviations from the fitted curve come in three layers: a component shared by
    every issuer at a given date and tenor, one shared within a bloc, and one
    purely idiosyncratic. Without the shared layers every issuer's residuals
    would be independent by construction, every correlation network would come
    out empty, and the network half of the comparison harness would have nothing
    to measure. Real curve panels have exactly this structure -- a tenor that is
    dislocated for one issuer is usually dislocated for its neighbours too.

    The default scales are chosen so the bloc layer dominates the global one:
    within a bloc the residual correlation lands near 0.63 and across blocs near
    0.06, which straddles the 0.3 edge threshold and yields networks with real
    community structure. Raising ``common_sd`` much above ``bloc_sd`` produces a
    complete graph instead, which -- as the module docstring of
    :mod:`ycn.analysis.residual_networks` notes -- says very little.

    Args:
        n_dates: Business days to simulate.
        issuers: Mapping of issuer name to bloc; defaults to three per bloc.
        maturities: Maturities in years.
        term_labels: Column labels aligned with ``maturities``.
        decay: Decay in the repo's **scale** convention.
        theta: Long-run factor means.
        kappa: Annual mean-reversion rates.
        sigma: Annual factor volatilities (diagonal Sigma).
        dt: Observation interval in years.
        meas_sd: Idiosyncratic per-issuer deviation, in decimals.
        common_sd: Deviation shared by every issuer at a date and tenor.
        bloc_sd: Deviation shared within a bloc at a date and tenor.
        seed: RNG seed.

    Returns:
        ``(long_frame, truth)`` where the frame has columns
        ``(date, source, term, rate)`` in decimals.
    """
    if issuers is None:
        issuers = {f"{bloc}_{i}": bloc for bloc in BLOCS for i in range(1, 4)}

    rng = np.random.default_rng(seed)
    taus = np.asarray(maturities, dtype=float)
    theta_arr = np.asarray(theta, dtype=float)
    kappa_arr = np.asarray(kappa, dtype=float)
    sigma_diag = np.asarray(sigma, dtype=float)
    sigma_mat = np.diag(sigma_diag)

    design = afns_loadings(taus, decay)
    adjustment = afns_yield_adjustment(taus, decay_to_lambda(decay), sigma_mat)

    start = date(2019, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_dates)]

    n_terms = len(taus)
    # Curve deviations shared across issuers, and within each bloc. These are
    # what make the residual correlation networks non-trivial.
    common = rng.normal(0.0, common_sd, size=(n_dates, n_terms))
    bloc_deviation = {
        bloc: rng.normal(0.0, bloc_sd, size=(n_dates, n_terms)) for bloc in BLOCS
    }

    rows: list[dict[str, object]] = []
    factor_paths: dict[str, np.ndarray] = {}
    for issuer, bloc in issuers.items():
        factors = simulate_factors(n_dates, theta_arr, kappa_arr, sigma_diag, dt, rng)
        factor_paths[issuer] = factors
        curves = factors @ design.T + adjustment + BLOCS[bloc]
        curves = (
            curves
            + common
            + bloc_deviation[bloc]
            + rng.normal(0.0, meas_sd, size=curves.shape)
        )
        for t, day in enumerate(dates):
            for label, rate in zip(term_labels, curves[t]):
                rows.append(
                    {"date": day, "source": issuer, "term": label, "rate": float(rate)}
                )

    frame = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "source": pl.Utf8,
            "term": pl.Utf8,
            "rate": pl.Float64,
        },
    ).sort("date", "source", "term")

    truth = AfnsTruth(
        decay=decay,
        theta=theta_arr,
        kappa=kappa_arr,
        sigma=sigma_mat,
        dt=dt,
        maturities=taus,
        factors=factor_paths,
        adjustment=adjustment,
    )
    return frame, truth
