"""Two-step estimator for arbitrage-free Nelson-Siegel models.

One driver serves every arbitrage-free model in the registry, because they
differ only in two pure functions: the factor loadings and the state-independent
yield adjustment. Both are supplied by a :class:`_Kernel`; everything else --
the cross-sectional pass, the VAR pass, the volatility bootstrap and the
assembly of :class:`~ycn.analysis.af_models.PanelFit` -- is shared.

The estimator follows Diebold & Li (2006): fix the decay, recover the factors by
least squares per date, then model the factor series. The one addition is that
the no-arbitrage correction enters the cross-sectional step as a fixed offset.
The model is ``y(tau) = B(tau)' X + adjustment(tau)``, so the factors come from
regressing the de-adjusted yields:

    y(tau) - adjustment(tau)  =  B(tau)' X

Since the adjustment depends on the volatility matrix, and the volatility matrix
is estimated from the factor series, the two are bootstrapped: start from zero
volatility (which is exactly plain Nelson-Siegel), fit, estimate Sigma, rebuild
the offset, and refit.

The mean-reversion matrix K never enters the cross-sectional fit -- only Sigma
and the decay do -- which keeps the whole estimator closed-form and keeps the
matrix logarithm off the critical path. K is computed only for reporting.

Known limitation, inherited from the two-step approach: the VAR runs on
*estimated* factors, so Sigma carries errors-in-variables bias. That is
acknowledged in Diebold & Li and is precisely why the Kalman estimator exists as
an opt-in refinement; see :mod:`ycn.analysis.af_state_space`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cholesky, logm

from .af_loadings import (
    afns_loadings,
    afns_yield_adjustment,
    decay_to_lambda,
    dtafns_loadings,
    dtafns_yield_adjustment,
)
from .af_models import (
    ModelSpec,
    PanelFit,
    ResidualModel,
    SigmaScope,
    detect_yield_scale,
)

# Panel-level decay grid used when `ModelSpec.decay` is None. AFNS treats the
# decay as a single constant for the whole panel (CDR 2011, section 2.2), so
# unlike the plain NS grid search this picks one value for every date, not one
# per date. Bounds match `fit_nelson_siegel`'s defaults, in the scale convention.
DECAY_GRID = np.linspace(0.05, 2.0, 50)

# Minimum factor observations before a VAR is worth running. Below this the
# volatility estimate is noise and the adjustment is left at zero.
MIN_DATES_FOR_VAR = 12


@dataclass(frozen=True)
class _Kernel:
    """The two pure functions that distinguish one arbitrage-free model.

    Both take ``(maturities, decay, ...)`` with the decay in the repo's **scale**
    convention and perform any conversion internally, so this driver never has
    to know which convention a given model's literature uses -- a rate per year
    for AFNS, a decay per period for DTAFNS.
    """

    loadings: Callable[[np.ndarray, float, float], np.ndarray]
    adjustment: Callable[[np.ndarray, float, np.ndarray, float], np.ndarray]


def _afns_loadings(maturities: np.ndarray, decay: float, dt: float) -> np.ndarray:
    """AFNS loadings, adapted to the driver's signature."""
    del dt  # continuous-time AFNS has no observation-interval dependence
    return afns_loadings(maturities, decay)


def _afns_adjustment(
    maturities: np.ndarray, decay: float, sigma: np.ndarray, dt: float
) -> np.ndarray:
    """AFNS adjustment, adapted to the driver's scale-convention signature."""
    del dt
    return afns_yield_adjustment(maturities, decay_to_lambda(decay), sigma)


KERNELS: dict[ResidualModel, _Kernel] = {
    ResidualModel.AFNS: _Kernel(loadings=_afns_loadings, adjustment=_afns_adjustment),
    ResidualModel.DTAFNS: _Kernel(
        loadings=dtafns_loadings, adjustment=dtafns_yield_adjustment
    ),
}


def _cross_section(
    design: np.ndarray, yields: np.ndarray, adjustment: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares factors and residuals for every date at once.

    The model is ``y(tau) = B(tau)' X + adjustment(tau)``, so the factors are
    recovered by regressing the *de-adjusted* yields ``y - adjustment`` on the
    loadings. Getting that sign wrong does not merely fail to help -- it applies
    the convexity correction twice and fits worse than plain Nelson-Siegel.

    The design matrix is fixed once the decay is fixed, so the whole panel is
    one ``lstsq`` call rather than a Python loop over dates.

    Args:
        design: (n_terms, 3) loadings.
        yields: (n_dates, n_terms) observed yields.
        adjustment: (n_terms,) the model's additive yield term, ``-A(tau)/tau``.

    Returns:
        ``(factors, residuals)`` shaped ``(n_dates, 3)`` and
        ``(n_dates, n_terms)``. Residuals are ``observed - fitted`` where the
        fitted yield includes the adjustment, so they carry the same meaning and
        the same units as plain Nelson-Siegel residuals.
    """
    deadjusted = yields - adjustment
    factors = np.linalg.lstsq(design, deadjusted.T, rcond=None)[0].T
    # resid = (y - adj) - X beta is identical to y - (X beta + adj), i.e.
    # observed minus the model's own fitted yield.
    return factors, deadjusted - factors @ design.T


def _estimate_sigma(
    factors: np.ndarray, dt: float, correlated: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Volatility matrix and mean reversion from a VAR(1) on the factors.

    Fits ``X[t+1] = c + Phi X[t] + eps`` by least squares and maps the residual
    covariance to a continuous-time volatility via ``Sigma Sigma' = cov / dt``.
    Sigma itself is recovered as the Cholesky factor, which is lower triangular
    and therefore exactly CDR's maximally-flexible identified form.

    Args:
        factors: (n_dates, 3) factor series.
        dt: Observation interval in years.
        correlated: Keep off-diagonal terms; otherwise force a diagonal Sigma.

    Returns:
        ``(sigma, kappa_p)``. ``kappa_p`` is ``None`` when the matrix logarithm
        is undefined, which happens whenever Phi has a non-positive real
        eigenvalue -- reporting-only, so it never blocks the fit.
    """
    n_dates = factors.shape[0]
    if n_dates < MIN_DATES_FOR_VAR:
        return np.zeros((3, 3)), None

    lagged, current = factors[:-1], factors[1:]
    design = np.column_stack([np.ones(len(lagged)), lagged])
    coef = np.linalg.lstsq(design, current, rcond=None)[0]  # (4, 3)
    resid = current - design @ coef
    dof = max(len(current) - design.shape[1], 1)
    cov = resid.T @ resid / dof

    sigma_sq = cov / dt  # this is Sigma Sigma'
    if correlated:
        try:
            sigma = cholesky(sigma_sq, lower=True)
        except (LinAlgError, np.linalg.LinAlgError):
            # Not positive definite -- fall back rather than abandon the fit.
            sigma = np.diag(np.sqrt(np.clip(np.diag(sigma_sq), 0.0, None)))
    else:
        sigma = np.diag(np.sqrt(np.clip(np.diag(sigma_sq), 0.0, None)))

    phi = coef[1:].T  # (3, 3)
    kappa_p: np.ndarray | None
    try:
        # Reporting only, and undefined whenever Phi has a non-positive real
        # eigenvalue -- which a short or near-constant factor series routinely
        # produces. Warnings are suppressed rather than surfaced because there
        # is nothing the caller can do and nothing downstream depends on it.
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidate = -np.real(logm(phi)) / dt
        kappa_p = candidate if np.all(np.isfinite(candidate)) else None
    except (ValueError, LinAlgError, np.linalg.LinAlgError):
        kappa_p = None

    return sigma, kappa_p


def _select_decay(kernel: _Kernel, maturities: np.ndarray, yields: np.ndarray) -> float:
    """Panel-level decay grid search on the zero-adjustment fit.

    AFNS treats the decay as one constant for the whole panel, so this picks a
    single value by total residual sum of squares rather than one per date.
    """
    zero = np.zeros(len(maturities))
    best_decay, best_sse = float(DECAY_GRID[0]), np.inf
    for candidate in DECAY_GRID:
        design = kernel.loadings(maturities, float(candidate), dt)
        _, resid = _cross_section(design, yields, zero)
        sse = float(np.sum(resid**2))
        if sse < best_sse:
            best_decay, best_sse = float(candidate), sse
    return best_decay


def fit_two_step(
    maturities: np.ndarray,
    yields: np.ndarray,
    spec: ModelSpec,
    *,
    on_chunk: Callable[[], None] | None = None,
    sigma_override: np.ndarray | None = None,
) -> PanelFit:
    """Fit one issuer's panel by the two-step arbitrage-free estimator.

    Args:
        maturities: (n_terms,) maturities in years.
        yields: (n_dates, n_terms) observed yields, no NaN, in the caller's units.
        spec: Model, decay, volatility scope and iteration count.
        on_chunk: Cancellation/GIL-handover hook, called once per pass.
        sigma_override: Use this volatility matrix instead of estimating one.
            Supplied by the pooled scope, which estimates Sigma across every
            issuer before refitting each of them.

    Returns:
        The fitted panel. Residuals are returned in the caller's original yield
        units, so the residual cube stays comparable across models regardless of
        whether the panel was quoted in percent or decimals.

    Raises:
        NotImplementedError: If ``spec.model`` has no kernel.
    """
    kernel = KERNELS.get(spec.model)
    if kernel is None:
        raise NotImplementedError(
            f"No two-step kernel for {spec.model.value!r}. "
            f"Available: {sorted(m.value for m in KERNELS)}."
        )

    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)
    n_dates, n_terms = yields.shape

    # Work in decimals throughout: the adjustment scales with the square of the
    # volatility, so percent input would inflate it by a factor of 10,000.
    scale = (
        spec.yield_scale if spec.yield_scale is not None else detect_yield_scale(yields)
    )
    scaled = yields * scale

    decay = (
        spec.decay
        if spec.decay is not None
        else _select_decay(kernel, maturities, scaled)
    )
    design = kernel.loadings(maturities, decay, spec.dt)

    adjustment = np.zeros(n_terms, dtype=float)
    sigma = np.zeros((3, 3), dtype=float)
    kappa_p: np.ndarray | None = None
    sigma_path: list[list[float]] = []

    # Seed pass has zero adjustment, so it is exactly plain Nelson-Siegel; each
    # refit folds in the volatility implied by the previous pass's factors.
    passes = 1 if sigma_override is not None else max(int(spec.n_refit), 0) + 1
    for index in range(passes):
        factors, residuals = _cross_section(design, scaled, adjustment)

        if sigma_override is not None:
            sigma = np.asarray(sigma_override, dtype=float)
        elif index < passes - 1:
            sigma, kappa_p = _estimate_sigma(factors, spec.dt, spec.correlated_sigma)
        else:
            break  # final pass: factors are already fitted against the final offset

        sigma_path.append([float(v) for v in np.diag(sigma)])
        adjustment = kernel.adjustment(maturities, decay, sigma, spec.dt)
        if on_chunk is not None:
            on_chunk()

    if sigma_override is not None:
        # One extra pass so the returned factors reflect the pooled offset.
        factors, residuals = _cross_section(design, scaled, adjustment)
        _, kappa_p = _estimate_sigma(factors, spec.dt, spec.correlated_sigma)

    rmse = np.sqrt(np.mean(residuals**2, axis=1))
    converged = len(sigma_path) < 2 or bool(
        np.allclose(sigma_path[-1], sigma_path[-2], rtol=1e-3, atol=1e-12)
    )

    return PanelFit(
        residuals=residuals / scale,  # back to the caller's units
        factors=factors / scale,
        decay=np.full(n_dates, decay, dtype=float),
        rmse=rmse / scale,
        adjustment=adjustment / scale,
        sigma=sigma,
        kappa_p=kappa_p,
        diagnostics={
            "model": spec.model.value,
            "n_dates": n_dates,
            "decay": decay,
            "yield_scale": scale,
            "sigma_scope": spec.sigma_scope.value,
            "sigma_path": sigma_path,
            "n_iterations": len(sigma_path),
            "sigma_converged": converged,
            "correlated_sigma": spec.correlated_sigma,
        },
    )


def pooled_sigma(fits: dict[str, PanelFit], dt: float, correlated: bool) -> np.ndarray:
    """One volatility matrix estimated across every issuer's factor series.

    Pooling is far better identified than a per-issuer VAR on a short or ragged
    history, and it makes the no-arbitrage offset common across issuers -- so a
    cross-issuer residual correlation then reflects genuine deviation rather
    than differences in each issuer's estimated convexity.

    Args:
        fits: Seed-pass fits keyed by issuer.
        dt: Observation interval in years.
        correlated: Keep off-diagonal terms.

    Returns:
        (3, 3) volatility matrix; zeros when no issuer has enough history.
    """
    innovations: list[np.ndarray] = []
    for fit in fits.values():
        # PanelFit.factors is rescaled to the caller's original yield units
        # (see the `factors / scale` line above), but the VAR here must run in
        # the same decimal units the cross-sectional fit and the adjustment
        # kernel use internally -- undo that rescaling with the fit's own
        # recorded `yield_scale` before pooling. Left as `fit.factors`, a
        # percent-quoted panel (scale=0.01) inflates the pooled Sigma ~10,000x,
        # which then inflates the adjustment by the same factor since it scales
        # with Sigma squared.
        factors = fit.factors * fit.diagnostics.get("yield_scale", 1.0)
        if factors.shape[0] < MIN_DATES_FOR_VAR:
            continue
        lagged, current = factors[:-1], factors[1:]
        design = np.column_stack([np.ones(len(lagged)), lagged])
        coef = np.linalg.lstsq(design, current, rcond=None)[0]
        innovations.append(current - design @ coef)

    if not innovations:
        return np.zeros((3, 3))

    stacked = np.vstack(innovations)
    dof = max(stacked.shape[0] - 4, 1)
    sigma_sq = (stacked.T @ stacked / dof) / dt
    if correlated:
        try:
            return cholesky(sigma_sq, lower=True)
        except (LinAlgError, np.linalg.LinAlgError):
            pass
    return np.diag(np.sqrt(np.clip(np.diag(sigma_sq), 0.0, None)))
