"""Kalman-filter maximum-likelihood estimation for arbitrage-free models.

The opt-in alternative to :mod:`ycn.analysis.af_fit`. The two-step estimator
recovers factors by least squares and then fits a VAR to them, which means the
volatility matrix is estimated from *measured* factors and inherits their
estimation noise -- the errors-in-variables bias Diebold & Li (2006) acknowledge.
Treating the factors as latent states and estimating every parameter jointly by
maximum likelihood avoids that, at the cost of a non-convex optimisation.

Because it is slower and can converge to a local optimum, this is a refinement
rather than the default: it is seeded from the two-step fit, so its starting
point is already a reasonable answer.

State-space form, with ``B`` the model's loadings and ``adj`` its yield
adjustment::

    X[t]  =  (I - F) theta  +  F X[t-1]  +  eta,   eta ~ N(0, Q)
    y[t]  =  B X[t] + adj                +  eps,   eps ~ N(0, R)

``F`` and ``Q`` come from the exact Ornstein-Uhlenbeck discretisation rather than
an Euler step, so the fitted volatility means the same thing at any observation
interval.

Out of scope, deliberately: extended and unscented filters, shadow-rate or other
lower-bound models, regime switching, Bayesian estimation, and joint estimation
across issuers. See :mod:`ycn.analysis.af_references` for the models evaluated
and not built.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .af_fit import KERNELS, fit_two_step
from .af_models import ModelSpec, PanelFit, detect_yield_scale

# Mean-reversion rates are bounded away from zero and from implausibly fast
# reversion. Zero would make the stationary variance diverge; the upper bound
# keeps the optimiser out of regions where the state is white noise.
KAPPA_BOUNDS = (1e-3, 50.0)
SIGMA_BOUNDS = (1e-8, 1.0)
MEAS_SD_BOUNDS = (1e-8, 1.0)

# Guard against a singular innovation covariance in the likelihood.
JITTER = 1e-12


@dataclass(frozen=True)
class StateSpaceParams:
    """Parameters of the discretised state-space model.

    Args:
        kappa: (3,) annual mean-reversion rates.
        theta: (3,) long-run factor means.
        sigma: (3, 3) volatility matrix.
        meas_sd: (n_terms,) measurement standard deviations.
    """

    kappa: np.ndarray
    theta: np.ndarray
    sigma: np.ndarray
    meas_sd: np.ndarray


def exact_discretisation(
    kappa: np.ndarray, sigma: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Transition matrix and innovation covariance for a diagonal-K OU process.

    With ``K`` diagonal the matrix exponential is elementwise and the integral
    ``Q = int_0^dt exp(-K s) Sigma Sigma' exp(-K' s) ds`` has the closed form
    ``Q_ij = (Sigma Sigma')_ij (1 - exp(-(k_i + k_j) dt)) / (k_i + k_j)``. Using
    the exact form rather than ``Q ~ Sigma Sigma' dt`` matters because the fitted
    volatility is compared against the two-step estimate and against the
    simulator's ground truth, both of which are annualised.

    Args:
        kappa: (3,) mean-reversion rates, strictly positive.
        sigma: (3, 3) volatility matrix.
        dt: Observation interval in years.

    Returns:
        ``(F, Q)`` with ``F`` diagonal.
    """
    kappa = np.asarray(kappa, dtype=float)
    transition = np.diag(np.exp(-kappa * dt))
    gram = sigma @ sigma.T
    rates = kappa[:, None] + kappa[None, :]
    innovation = gram * (1.0 - np.exp(-rates * dt)) / rates
    return transition, innovation


def kalman_filter(
    yields: np.ndarray,
    design: np.ndarray,
    adjustment: np.ndarray,
    params: StateSpaceParams,
    dt: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run the filter and return the log-likelihood, states and residuals.

    Args:
        yields: (n_dates, n_terms) observations.
        design: (n_terms, 3) loadings.
        adjustment: (n_terms,) additive yield term.
        params: Model parameters.
        dt: Observation interval in years.

    Returns:
        ``(loglik, filtered_states, residuals)``. Residuals are
        ``observed - fitted`` using the *filtered* state, so they are the closest
        state-space analogue of a cross-sectional residual.
    """
    n_dates, n_terms = yields.shape
    transition, innovation = exact_discretisation(params.kappa, params.sigma, dt)
    drift = (np.eye(3) - transition) @ params.theta
    meas_cov = np.diag(params.meas_sd**2)

    # Start from the stationary distribution so the first observations are not
    # dominated by an arbitrary prior.
    state = params.theta.copy()
    gram = params.sigma @ params.sigma.T
    rates = params.kappa[:, None] + params.kappa[None, :]
    covariance = gram / rates

    loglik = 0.0
    states = np.empty((n_dates, 3), dtype=float)
    residuals = np.empty((n_dates, n_terms), dtype=float)
    constant = n_terms * np.log(2.0 * np.pi)

    for t in range(n_dates):
        # Predict.
        state = drift + transition @ state
        covariance = transition @ covariance @ transition.T + innovation

        # Update.
        innovation_mean = yields[t] - (design @ state + adjustment)
        innovation_cov = design @ covariance @ design.T + meas_cov
        innovation_cov.flat[:: n_terms + 1] += JITTER

        try:
            cholesky = np.linalg.cholesky(innovation_cov)
        except np.linalg.LinAlgError:
            return -np.inf, states, residuals

        solved = np.linalg.solve(cholesky, innovation_mean)
        log_det = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
        loglik -= 0.5 * (constant + log_det + float(solved @ solved))

        gain = covariance @ design.T @ np.linalg.inv(innovation_cov)
        state = state + gain @ innovation_mean
        covariance = covariance - gain @ design @ covariance

        states[t] = state
        residuals[t] = yields[t] - (design @ state + adjustment)

    return loglik, states, residuals


def _pack(params: StateSpaceParams, per_maturity: bool) -> np.ndarray:
    """Parameters to an unconstrained vector for the optimiser."""
    meas = params.meas_sd if per_maturity else params.meas_sd[:1]
    return np.concatenate(
        [
            np.log(np.clip(params.kappa, *KAPPA_BOUNDS)),
            params.theta,
            np.log(np.clip(np.diag(params.sigma), *SIGMA_BOUNDS)),
            np.log(np.clip(meas, *MEAS_SD_BOUNDS)),
        ]
    )


def _unpack(vector: np.ndarray, n_terms: int, per_maturity: bool) -> StateSpaceParams:
    """Inverse of :func:`_pack`.

    Positive quantities are carried in logs so the optimiser is unconstrained
    and cannot step onto a negative variance.
    """
    kappa = np.exp(vector[0:3])
    theta = vector[3:6]
    sigma = np.diag(np.exp(vector[6:9]))
    raw_meas = np.exp(vector[9:])
    meas_sd = raw_meas if per_maturity else np.full(n_terms, raw_meas[0])
    return StateSpaceParams(kappa=kappa, theta=theta, sigma=sigma, meas_sd=meas_sd)


def fit_state_space_mle(
    maturities: np.ndarray,
    yields: np.ndarray,
    spec: ModelSpec,
    *,
    on_chunk: Callable[[], None] | None = None,
    sigma_override: np.ndarray | None = None,
    max_iter: int = 200,
) -> PanelFit:
    """Estimate one issuer's panel by Kalman-filter maximum likelihood.

    Seeded from :func:`~ycn.analysis.af_fit.fit_two_step`, so the optimiser
    starts from a sensible answer rather than an arbitrary one -- which matters
    on a likelihood surface known to be flat and multimodal.

    Args:
        maturities: (n_terms,) maturities in years.
        yields: (n_dates, n_terms) observed yields in the caller's units.
        spec: Model and estimation settings. ``decay`` is held fixed.
        on_chunk: Cancellation/GIL-handover hook.
        sigma_override: Pooled volatility, if the caller estimated one. When
            given, the volatility is held at this value and only the remaining
            parameters are estimated.
        max_iter: Optimiser iteration cap.

    Returns:
        The fitted panel, with **filtered** residuals in the caller's units.

    Raises:
        NotImplementedError: If ``spec.model`` has no kernel.
    """
    kernel = KERNELS.get(spec.model)
    if kernel is None:
        raise NotImplementedError(
            f"No state-space kernel for {spec.model.value!r}. "
            f"Available: {sorted(m.value for m in KERNELS)}."
        )

    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)
    n_dates, n_terms = yields.shape

    seed = fit_two_step(
        maturities, yields, spec, on_chunk=on_chunk, sigma_override=sigma_override
    )
    if n_dates < 3 * n_terms:
        # Too little data to identify the extra parameters; the two-step answer
        # is the better one and is already computed.
        seed.diagnostics["estimator"] = "two_step_fallback"
        seed.diagnostics["fallback_reason"] = "insufficient dates for MLE"
        return seed

    scale = (
        spec.yield_scale if spec.yield_scale is not None else detect_yield_scale(yields)
    )
    scaled = yields * scale
    decay = float(seed.decay[0])
    design = kernel.loadings(maturities, decay, spec.dt)

    seed_sigma = seed.sigma if np.any(seed.sigma) else np.diag([0.005, 0.010, 0.020])
    seed_kappa = np.array([0.2, 0.5, 0.9])
    if seed.kappa_p is not None:
        candidate = np.clip(np.abs(np.diag(seed.kappa_p)), *KAPPA_BOUNDS)
        if np.all(np.isfinite(candidate)):
            seed_kappa = candidate
    seed_meas = max(float(np.mean(seed.rmse)) * scale, MEAS_SD_BOUNDS[0])

    start = StateSpaceParams(
        kappa=seed_kappa,
        theta=seed.factors.mean(axis=0) * scale,
        sigma=seed_sigma,
        meas_sd=np.full(n_terms, seed_meas),
    )
    per_maturity = bool(spec.correlated_sigma)  # reuse the flag: richer error model

    def negative_loglik(vector: np.ndarray) -> float:
        params = _unpack(vector, n_terms, per_maturity)
        if sigma_override is not None:
            params = StateSpaceParams(
                kappa=params.kappa,
                theta=params.theta,
                sigma=np.asarray(sigma_override, dtype=float),
                meas_sd=params.meas_sd,
            )
        adjustment = kernel.adjustment(maturities, decay, params.sigma, spec.dt)
        loglik, _, _ = kalman_filter(scaled, design, adjustment, params, spec.dt)
        return -loglik if np.isfinite(loglik) else 1e12

    initial = _pack(start, per_maturity)
    result = minimize(
        negative_loglik,
        initial,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-10},
    )
    if on_chunk is not None:
        on_chunk()

    best = (
        result.x if result.x is not None and np.all(np.isfinite(result.x)) else initial
    )
    params = _unpack(best, n_terms, per_maturity)
    if sigma_override is not None:
        params = StateSpaceParams(
            kappa=params.kappa,
            theta=params.theta,
            sigma=np.asarray(sigma_override, dtype=float),
            meas_sd=params.meas_sd,
        )

    adjustment = kernel.adjustment(maturities, decay, params.sigma, spec.dt)
    loglik, states, residuals = kalman_filter(
        scaled, design, adjustment, params, spec.dt
    )

    if not np.isfinite(loglik):
        # The optimiser wandered somewhere unusable; the seed is still valid.
        seed.diagnostics["estimator"] = "two_step_fallback"
        seed.diagnostics["fallback_reason"] = "non-finite likelihood at the optimum"
        return seed

    return PanelFit(
        residuals=residuals / scale,
        factors=states / scale,
        decay=np.full(n_dates, decay, dtype=float),
        rmse=np.sqrt(np.mean(residuals**2, axis=1)) / scale,
        adjustment=adjustment / scale,
        sigma=params.sigma,
        kappa_p=np.diag(params.kappa),
        diagnostics={
            "model": spec.model.value,
            "estimator": "kalman",
            "n_dates": n_dates,
            "decay": decay,
            "yield_scale": scale,
            "sigma_scope": spec.sigma_scope.value,
            "loglik": float(loglik),
            "seed_loglik": float(-negative_loglik(initial)),
            "converged": bool(result.success),
            "n_iterations": int(result.nit),
            "sigma_converged": bool(result.success),
            "optimiser_message": str(result.message),
            "theta": params.theta.tolist(),
            "meas_sd": params.meas_sd.tolist(),
            # Filtered residuals are not the same object as a cross-sectional
            # residual; flagged so the harness can say so.
            "filtered_residuals": True,
        },
    )
