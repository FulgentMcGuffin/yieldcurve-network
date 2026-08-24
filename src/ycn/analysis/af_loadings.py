"""Factor loadings and no-arbitrage yield adjustments.

Two conventions meet in this module and they are reciprocals of each other:

* This repo parameterises Nelson-Siegel by a **scale**, ``exp(-tau / decay)``,
  and calls it ``decay`` (see :func:`ycn.analysis.yield_curve_factors.ns_basis`).
* The arbitrage-free literature parameterises it by a **rate**,
  ``exp(-lambda * tau)``, and calls it ``lambda``.

So ``lambda = 1 / decay``. The boundary is enforced by naming: a parameter named
``decay`` is always a scale, a parameter named ``lambda_`` is always a rate.

The split is deliberate rather than cosmetic. **Loadings keep the scale
convention** so they can be handed to :func:`ns_basis` untouched: the reciprocal
round-trip is not exact in floating point (``1 / (1 / 1.8) == 1.8000000000000003``),
and a loading that differs in its last bits would quietly weaken the guarantee
that a zero-volatility AFNS fit reproduces legacy Nelson-Siegel residuals
*exactly*. **The adjustment keeps the rate convention** so it can be read
line-by-line against the published closed form; a last-bit difference there is
multiplied by a variance of order 1e-4 and is irrelevant.

The key structural fact this module exists to exploit: in both AFNS and DTAFNS
the no-arbitrage correction is **state-independent** -- a function of
``(lambda, Sigma, tau)`` carrying no factor. It is therefore a fixed per-maturity
offset, so an arbitrage-free cross-sectional fit is ordinary least squares on
adjusted yields against the *same* Nelson-Siegel loadings:

    y(tau) + A(tau)/tau  =  B(tau)' X

References:
    Christensen, Diebold & Rudebusch (2011), "The affine arbitrage-free class of
    Nelson-Siegel term structure models", J. Econometrics 164(1), 4-20.
    Proposition 1 and section 2.3. See :mod:`ycn.analysis.af_references`.
"""

from __future__ import annotations

import numpy as np

from .yield_curve_factors import ns_basis

__all__ = [
    "decay_to_lambda",
    "lambda_to_decay",
    "afns_loadings",
    "afns_yield_adjustment",
    "dtafns_period_decay",
    "dtafns_loadings",
    "dtafns_yield_adjustment",
]


def decay_to_lambda(decay: float) -> float:
    """Scale convention (repo) to rate convention (literature).

    Args:
        decay: Scale parameter; loadings use ``exp(-tau / decay)``.

    Returns:
        The equivalent rate ``lambda = 1 / decay``.

    Raises:
        ValueError: If ``decay`` is not strictly positive.
    """
    if not decay > 0.0:
        raise ValueError(f"decay must be strictly positive, got {decay!r}")
    return 1.0 / float(decay)


def lambda_to_decay(lambda_: float) -> float:
    """Rate convention (literature) to scale convention (repo).

    Args:
        lambda_: Rate parameter; loadings use ``exp(-lambda * tau)``.

    Returns:
        The equivalent scale ``decay = 1 / lambda``.

    Raises:
        ValueError: If ``lambda_`` is not strictly positive.
    """
    if not lambda_ > 0.0:
        raise ValueError(f"lambda_ must be strictly positive, got {lambda_!r}")
    return 1.0 / float(lambda_)


def afns_loadings(maturities: np.ndarray, decay: float) -> np.ndarray:
    """AFNS design matrix -- identical to the Nelson-Siegel loadings.

    AFNS shares its factor loadings with plain Nelson-Siegel exactly; the whole
    difference is the additive yield adjustment. This delegates to
    :func:`ns_basis` rather than re-deriving the columns, for two reasons:
    ``ns_basis`` floors maturities at ``1e-6`` instead of taking the analytic
    ``tau -> 0`` limit, and it must receive ``decay`` unconverted, since routing
    it through ``lambda`` and back is not exact in floating point. Either
    departure would leave AFNS loadings differing from NS loadings in their last
    bits and break the exact-equivalence guarantee.

    Args:
        maturities: (n_terms,) maturities in years.
        decay: Decay in the repo's **scale** convention, as ``ns_basis`` takes it.

    Returns:
        (n_terms, 3) design matrix with columns level, slope, curvature.
    """
    f0, f1, f2 = ns_basis(maturities, decay)
    return np.column_stack([f0, f1, f2])


def afns_yield_adjustment(
    maturities: np.ndarray, lambda_: float, sigma: np.ndarray
) -> np.ndarray:
    """The AFNS yield-adjustment term ``-A(tau)/tau``.

    Christensen-Diebold-Rudebusch (2011) Proposition 1 with ``theta^Q = 0``. The
    term is a pure convexity correction: it depends only on the volatility
    matrix, the decay and the maturity, never on the state. That is what makes
    the arbitrage-free cross-sectional fit ordinary least squares on
    ``y + A(tau)/tau``.

    Only six combinations of Sigma's nine entries are identified (CDR section
    2.3), so the maximally flexible identified specification is lower-triangular.
    Both the diagonal and the correlated cases are handled here.

    The adjustment is non-positive and grows in magnitude with maturity -- the
    leading ``tau^2 / 6`` term means arbitrage-free yields sit progressively
    below their Nelson-Siegel counterparts at the long end, which is exactly the
    misspecification a plain NS residual absorbs.

    Args:
        maturities: (n_terms,) maturities in years. Must be positive.
        lambda_: Decay in the **rate** convention.
        sigma: (3, 3) volatility matrix, or a length-3 vector read as its
            diagonal. Must be in the same yield units as the data.

    Returns:
        (n_terms,) array of ``-A(tau)/tau``, to be **added** to observed yields
        before the least-squares fit.
    """
    tau = np.asarray(maturities, dtype=float)
    if np.any(tau <= 0.0):
        raise ValueError("maturities must be strictly positive")
    lam = float(lambda_)
    if not lam > 0.0:
        raise ValueError(f"lambda_ must be strictly positive, got {lambda_!r}")

    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 1:
        sigma = np.diag(sigma)
    if sigma.shape != (3, 3):
        raise ValueError(f"sigma must be (3, 3) or length 3, got {sigma.shape}")

    # CDR's six identified volatility combinations: A..C are row norms, D..F the
    # cross-row inner products.
    a_bar = float(sigma[0] @ sigma[0])
    b_bar = float(sigma[1] @ sigma[1])
    c_bar = float(sigma[2] @ sigma[2])
    d_bar = float(sigma[0] @ sigma[1])
    e_bar = float(sigma[0] @ sigma[2])
    f_bar = float(sigma[1] @ sigma[2])

    if not any(abs(v) > 0.0 for v in (a_bar, b_bar, c_bar, d_bar, e_bar, f_bar)):
        # Sigma = 0 means no convexity correction, and AFNS collapses to NS.
        return np.zeros_like(tau)

    exp1 = np.exp(-lam * tau)
    exp2 = np.exp(-2.0 * lam * tau)
    # The two recurring "averaged decay" shapes, (1 - e^{-k*lambda*tau}) / tau.
    avg1 = (1.0 - exp1) / tau
    avg2 = (1.0 - exp2) / tau

    lam2, lam3 = lam**2, lam**3

    term_a = tau**2 / 6.0

    term_b = 1.0 / (2.0 * lam2) - avg1 / lam3 + avg2 / (4.0 * lam3)

    term_c = (
        1.0 / (2.0 * lam2)
        + exp1 / lam2
        - tau * exp2 / (4.0 * lam)
        - 3.0 * exp2 / (4.0 * lam2)
        - 2.0 * avg1 / lam3
        + 5.0 * avg2 / (8.0 * lam3)
    )

    term_d = tau / (2.0 * lam) + exp1 / lam2 - avg1 / lam3

    term_e = (
        3.0 * exp1 / lam2 + tau / (2.0 * lam) + tau * exp1 / lam - 3.0 * avg1 / lam3
    )

    term_f = (
        1.0 / lam2
        + exp1 / lam2
        - exp2 / (2.0 * lam2)
        - 3.0 * avg1 / lam3
        + 3.0 * avg2 / (4.0 * lam3)
    )

    c_over_tau = (
        a_bar * term_a
        + b_bar * term_b
        + c_bar * term_c
        + d_bar * term_d
        + e_bar * term_e
        + f_bar * term_f
    )
    return -c_over_tau


# ---------------------------------------------------------------------------
# Discrete-time AFNS
# ---------------------------------------------------------------------------


def dtafns_period_decay(decay: float, dt: float) -> float:
    """Per-period decay from the continuous scale.

    DTAFNS parameterises its risk-neutral mean reversion by a decay in ``(0, 1)``
    applied once per observation period, where the continuous model uses a rate
    per year. Matching the two means ``1 - lambda_period = exp(-lambda_rate * dt)``,
    which is what makes the discrete loadings converge to the continuous ones as
    the period shrinks.

    Args:
        decay: Decay in the repo's **scale** convention.
        dt: Observation interval in years.

    Returns:
        The per-period decay, strictly inside ``(0, 1)``.
    """
    return float(1.0 - np.exp(-decay_to_lambda(decay) * float(dt)))


def _dtafns_bond_loadings(periods: np.ndarray, period_decay: float) -> np.ndarray:
    """DTAFNS bond loadings ``B(n)``, before dividing by maturity.

    Args:
        periods: (k,) maturities measured in observation periods.
        period_decay: Per-period decay in ``(0, 1)``.

    Returns:
        (k, 3) array of ``[B1, B2, B3]``.
    """
    n = np.asarray(periods, dtype=float)
    lam = float(period_decay)
    rho = 1.0 - lam  # per-period persistence

    b1 = n
    b2 = (1.0 - rho**n) / lam
    b3 = (1.0 - rho ** (n - 1.0)) / lam - (n - 1.0) * rho ** (n - 1.0)
    return np.column_stack([b1, b2, b3])


def dtafns_loadings(maturities: np.ndarray, decay: float, dt: float) -> np.ndarray:
    """DTAFNS yield loadings.

    The discrete-time analogue of :func:`afns_loadings`. Loadings are the bond
    loadings divided by maturity in periods, and converge to the continuous
    Nelson-Siegel loadings as ``dt`` shrinks.

    Args:
        maturities: (n_terms,) maturities in years. Must exceed one period.
        decay: Decay in the repo's **scale** convention.
        dt: Observation interval in years.

    Returns:
        (n_terms, 3) design matrix with columns level, slope, curvature.
    """
    tau = np.asarray(maturities, dtype=float)
    if np.any(tau <= 0.0):
        raise ValueError("maturities must be strictly positive")
    periods = tau / float(dt)
    if np.any(periods < 1.0):
        raise ValueError(
            "every maturity must span at least one observation period; "
            f"shortest is {periods.min():.3f} periods at dt={dt:g}"
        )
    return (
        _dtafns_bond_loadings(periods, dtafns_period_decay(decay, dt))
        / periods[:, None]
    )


def dtafns_yield_adjustment(
    maturities: np.ndarray, decay: float, sigma: np.ndarray, dt: float
) -> np.ndarray:
    """The DTAFNS yield-adjustment term.

    Discrete time replaces the continuous model's integral with a finite sum. The
    affine recursion gives ``log A(n+1) = log A(n) + (1/2) dt^2 B(n)' S S' B(n)``
    where ``S`` is the **per-period** volatility, so with ``theta^Q = 0``

        log A(n) = (dt^3 / 2) * sum_{s=1}^{n-1} B(s)' Sigma Sigma' B(s)

    and the yield adjustment is ``-log A(n) / (dt * n)``.

    The third power of ``dt`` rather than the second is not a typo. ``sigma``
    here is **annualised**, matching the convention
    :func:`afns_yield_adjustment` and the rest of this package use, whereas the
    recursion's own volatility is per period: ``S S' = Sigma Sigma' * dt``. Using
    the second power silently inflates the correction by a factor of ``1 / dt``
    -- roughly 250 on a daily grid -- which is large enough to swamp the yields
    entirely and is caught by the convergence test.

    This is evaluated as the sum itself rather than by transcribing the paper's
    closed form in geometric series. The sum is the definition, it is exact, and
    it is cheap -- a few thousand terms per maturity. Transcribing an algebraic
    identity that could not be checked against the source would risk a silent
    error in exchange for nothing. The implementation is instead validated
    against :func:`afns_yield_adjustment`, which it must converge to as ``dt``
    shrinks; see ``tests/test_af_loadings.py``.

    Args:
        maturities: (n_terms,) maturities in years.
        decay: Decay in the repo's **scale** convention.
        sigma: (3, 3) volatility matrix, or a length-3 vector read as its
            diagonal, in the same yield units as the data.
        dt: Observation interval in years.

    Returns:
        (n_terms,) array to be used exactly as the AFNS adjustment is.
    """
    tau = np.asarray(maturities, dtype=float)
    if np.any(tau <= 0.0):
        raise ValueError("maturities must be strictly positive")

    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 1:
        sigma = np.diag(sigma)
    if sigma.shape != (3, 3):
        raise ValueError(f"sigma must be (3, 3) or length 3, got {sigma.shape}")
    if not np.any(sigma):
        return np.zeros_like(tau)

    step = float(dt)
    period_decay = dtafns_period_decay(decay, step)
    gram = sigma @ sigma.T  # Sigma Sigma'

    # Maturities in periods. Rounded to whole periods for the summation bound:
    # the model is defined on a period grid, and on the daily and monthly grids
    # used here every common tenor already lands on an integer. The worst case
    # is half a period -- one day in thirty years.
    periods = tau / step
    horizons = np.maximum(np.rint(periods).astype(int), 1)

    adjustment = np.zeros_like(tau)
    largest = int(horizons.max())
    if largest < 2:
        return adjustment

    # B(s) for every whole period up to the longest maturity, computed once and
    # shared across maturities via a cumulative sum.
    steps = np.arange(1, largest, dtype=float)
    loadings = _dtafns_bond_loadings(steps, period_decay)
    quadratic = np.einsum("si,ij,sj->s", loadings, gram, loadings)
    cumulative = np.concatenate([[0.0], np.cumsum(quadratic)])

    # dt^3: dt^2 from the recursion, one more from converting the annualised
    # Sigma to the per-period volatility the recursion actually uses.
    log_a = 0.5 * step**3 * cumulative[horizons - 1]
    return -log_a / (step * periods)
