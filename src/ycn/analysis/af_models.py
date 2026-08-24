"""Model registry for residual-producing curve fits.

The residual cube is the contract every downstream consumer depends on, and it
does not care how the residuals were produced. This module makes the producer
pluggable: a :class:`ModelSpec` selects one, every fitter returns the same
:class:`PanelFit`, and :func:`fit_panel` dispatches.

Adding a model means writing two pure functions (loadings and yield adjustment)
in :mod:`ycn.analysis.af_loadings` and registering one entry in ``FITTERS`` --
see ``.mex/patterns/add-residual-model.md``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

from .yield_curve_factors import fit_nelson_siegel

# Cheap existence check -- ``find_spec`` locates the package without importing
# it, so probing this at GUI start-up never pays torch's import cost (or forces
# it onto every user) just to decide whether to grey out a checkbox. Mirrors
# the ``ACE_AVAILABLE`` pattern in ``analysis/measures.py``.
NEURAL_AVAILABLE = importlib.util.find_spec("torch") is not None
NEURAL_IMPORT_ERROR: str | None = (
    None if NEURAL_AVAILABLE else "torch is not installed (uv sync --extra neural)"
)

# Above this absolute yield the panel is almost certainly quoted in percent
# rather than decimals. Real curves sit well under 50% and well over 0.5 in
# percent terms, so the gap is wide and the test is safe.
PERCENT_DETECT_THRESHOLD = 0.5

# Quantile of |yield| compared against that threshold. A high quantile rather
# than the median because a panel can legitimately sit near zero -- euro-area
# curves in 2020-21 were partly negative -- and its median absolute yield then
# falls below the threshold even though it is quoted in percent.
PERCENT_DETECT_QUANTILE = 0.9


class ResidualModel(str, Enum):
    """Which curve model produces the residuals."""

    NS = "ns"
    AFNS = "afns"
    DTAFNS = "dtafns"
    NEURAL_HJM = "neural_hjm"

    @property
    def label(self) -> str:
        return {
            ResidualModel.NS: "Nelson-Siegel",
            ResidualModel.AFNS: "Arbitrage-Free Nelson-Siegel",
            ResidualModel.DTAFNS: "Discrete-Time AFNS",
            ResidualModel.NEURAL_HJM: "Neural HJM (experimental)",
        }[self]

    @property
    def is_arbitrage_free(self) -> bool:
        """True when the model carries a no-arbitrage yield adjustment."""
        return self is not ResidualModel.NS


class Estimator(str, Enum):
    """How the model's parameters are estimated."""

    TWO_STEP = "two_step"
    KALMAN = "kalman"

    @property
    def label(self) -> str:
        return {
            Estimator.TWO_STEP: "Two-step (cross-section + VAR)",
            Estimator.KALMAN: "Kalman filter + MLE",
        }[self]


class SigmaScope(str, Enum):
    """Whether the volatility matrix is fitted per issuer or pooled.

    The yield adjustment is a function of Sigma, so this decides whether every
    issuer gets its own no-arbitrage correction or a common one. Pooling is
    better identified and makes cross-issuer residual correlation reflect
    genuine deviation rather than differences in estimated convexity.
    """

    PER_ISSUER = "per_issuer"
    POOLED = "pooled"

    @property
    def label(self) -> str:
        return {
            SigmaScope.PER_ISSUER: "Per issuer",
            SigmaScope.POOLED: "Pooled across issuers",
        }[self]


@dataclass(frozen=True)
class NeuralSpec:
    """Hyperparameters for the experimental neural HJM fitter."""

    hidden_size: int = 32
    n_layers: int = 1
    epochs: int = 600
    learning_rate: float = 1e-2
    # Weight on the no-arbitrage penalty, relative to a reconstruction term that
    # is normalised by the panel's own variance so the two are comparable.
    drift_weight: float = 0.1
    smooth_weight: float = 1e-3  # weight on the state-increment penalty


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to reproduce one residual fit.

    Args:
        model: Which curve model to fit.
        decay: Decay in the repo's **scale** convention -- loadings use
            ``exp(-tau / decay)``. ``None`` triggers a per-date grid search.
            Note the arbitrage-free literature uses the reciprocal *rate*
            convention; see :mod:`ycn.analysis.af_loadings`.
        estimator: Two-step by default; Kalman is an opt-in refinement.
        sigma_scope: Per-issuer or pooled volatility estimation.
        n_refit: Extra cross-sectional passes after the ``Sigma = 0`` seed pass.
        correlated_sigma: Full lower-triangular Sigma rather than diagonal.
        dt: Observation interval in years, for the VAR-to-Sigma mapping.
        yield_scale: Multiplier taking input yields to decimals. ``None``
            auto-detects percent input; see :func:`detect_yield_scale`.
        seed: RNG seed, for the neural fitter.
        neural: Neural fitter hyperparameters.
    """

    model: ResidualModel = ResidualModel.NS
    decay: float | None = 1.0
    estimator: Estimator = Estimator.TWO_STEP
    sigma_scope: SigmaScope = SigmaScope.PER_ISSUER
    n_refit: int = 1
    correlated_sigma: bool = False
    dt: float = 1.0 / 252.0
    yield_scale: float | None = None
    seed: int = 0
    neural: NeuralSpec | None = None

    @property
    def name(self) -> str:
        """Short identifier used as the ``model`` column in comparison frames."""
        parts = [self.model.value]
        if self.model.is_arbitrage_free:
            parts.append(self.sigma_scope.value)
            if self.estimator is not Estimator.TWO_STEP:
                parts.append(self.estimator.value)
        if self.decay is not None and self.decay != 1.0:
            parts.append(f"d{self.decay:g}")
        return "/".join(parts)


@dataclass
class PanelFit:
    """One issuer's fitted curve series.

    ``residuals`` deliberately matches ``NSFactors.residuals`` in shape and
    units -- ``yield - fitted``, unstandardised, in the caller's original yield
    units -- so the residual cube and every downstream consumer are unaffected
    by which model produced it.
    """

    residuals: np.ndarray  # (n_dates, n_terms)
    factors: np.ndarray  # (n_dates, 3)
    decay: np.ndarray  # (n_dates,) scale convention
    rmse: np.ndarray  # (n_dates,)
    adjustment: np.ndarray  # (n_terms,) the -A(tau)/tau offset; zeros for NS
    sigma: np.ndarray  # (3, 3)
    kappa_p: np.ndarray | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


class PanelFitter(Protocol):
    """Signature every registered fitter must satisfy.

    ``**kwargs`` carries estimator-specific extras -- currently only
    ``sigma_override``, used by the pooled volatility scope. A fitter that does
    not recognise an extra must ignore it rather than fail.
    """

    def __call__(
        self,
        maturities: np.ndarray,
        yields: np.ndarray,
        spec: ModelSpec,
        *,
        on_chunk: Callable[[], None] | None = None,
        **kwargs: object,
    ) -> PanelFit: ...


def detect_yield_scale(yields: np.ndarray) -> float:
    """Multiplier taking ``yields`` to decimals.

    The arbitrage-free yield adjustment scales with the square of the factor
    volatility, so feeding percent where decimals are expected inflates the
    correction by a factor of 10,000. The two databases in this project disagree
    -- ``data/ycs_fake.duckdb`` is decimals, ``ycs_data.duckdb`` par_rates is
    percent -- so the scale is detected rather than assumed.

    **Call this on the whole panel, not one issuer at a time.** The unit is a
    property of the table, and deciding per issuer is not merely redundant: a
    partly-negative curve such as a 2020-21 euro-area issuer has a low absolute
    yield throughout, so it is classified as decimals while its percent-quoted
    neighbours are classified correctly, and the two then get corrections that
    differ by a factor of 10,000. :func:`residual_cube` resolves the scale once
    and passes it down for exactly this reason.

    Args:
        yields: Observed yields, any shape.

    Returns:
        ``0.01`` when the panel looks like percent, otherwise ``1.0``.
    """
    finite = np.asarray(yields, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    magnitude = float(np.quantile(np.abs(finite), PERCENT_DETECT_QUANTILE))
    return 0.01 if magnitude > PERCENT_DETECT_THRESHOLD else 1.0


def fit_ns_panel(
    maturities: np.ndarray,
    yields: np.ndarray,
    spec: ModelSpec,
    *,
    on_chunk: Callable[[], None] | None = None,
    **kwargs: object,
) -> PanelFit:
    """Plain Nelson-Siegel, fitted independently per date.

    The registry's reference implementation and the zero-adjustment baseline
    every arbitrage-free model is compared against. Delegates to
    :func:`ycn.analysis.yield_curve_factors.fit_nelson_siegel` so it cannot
    drift from the legacy code path.

    Accepts and ignores ``sigma_override``: plain NS has no volatility term, so
    a pooled-scope sweep leaves it unchanged.
    """
    del kwargs  # NS has no estimator-specific extras
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)
    n_dates, n_terms = yields.shape

    residuals = np.empty((n_dates, n_terms), dtype=float)
    factors = np.empty((n_dates, 3), dtype=float)
    decays = np.empty(n_dates, dtype=float)
    rmses = np.empty(n_dates, dtype=float)

    for t in range(n_dates):
        ns = fit_nelson_siegel(maturities, yields[t], decay=spec.decay)
        residuals[t] = ns.residuals
        factors[t] = (ns.level, ns.slope, ns.curvature)
        decays[t] = ns.decay
        rmses[t] = ns.rmse
        if on_chunk is not None and (t + 1) % 200 == 0:
            on_chunk()

    return PanelFit(
        residuals=residuals,
        factors=factors,
        decay=decays,
        rmse=rmses,
        adjustment=np.zeros(n_terms, dtype=float),
        sigma=np.zeros((3, 3), dtype=float),
        diagnostics={"model": ResidualModel.NS.value, "n_dates": n_dates},
    )


def _fit_arbitrage_free(
    maturities: np.ndarray,
    yields: np.ndarray,
    spec: ModelSpec,
    *,
    on_chunk: Callable[[], None] | None = None,
    **kwargs: object,
) -> PanelFit:
    """Dispatch an arbitrage-free model to its estimator.

    Imported lazily because :mod:`ycn.analysis.af_fit` imports this module for
    :class:`ModelSpec` and :class:`PanelFit`.
    """
    from .af_fit import fit_two_step

    if spec.model is ResidualModel.NEURAL_HJM:
        from .af_neural import fit_neural_hjm

        return fit_neural_hjm(maturities, yields, spec, on_chunk=on_chunk, **kwargs)
    if spec.estimator is Estimator.KALMAN:
        from .af_state_space import fit_state_space_mle

        return fit_state_space_mle(
            maturities, yields, spec, on_chunk=on_chunk, **kwargs
        )
    return fit_two_step(maturities, yields, spec, on_chunk=on_chunk, **kwargs)


FITTERS: dict[ResidualModel, PanelFitter] = {
    ResidualModel.NS: fit_ns_panel,
    ResidualModel.AFNS: _fit_arbitrage_free,
    ResidualModel.DTAFNS: _fit_arbitrage_free,
    # Requires the `neural` extra; raises a clear ImportError without it.
    ResidualModel.NEURAL_HJM: _fit_arbitrage_free,
}


def available_models() -> list[tuple[str, str]]:
    """Registered ``(value, label)`` pairs, for CLI and settings menus."""
    return [(m.value, m.label) for m in ResidualModel if m in FITTERS]


def fit_panel(
    maturities: np.ndarray,
    yields: np.ndarray,
    spec: ModelSpec,
    *,
    on_chunk: Callable[[], None] | None = None,
    **kwargs: object,
) -> PanelFit:
    """Dispatch one issuer's panel to the fitter named by ``spec``.

    Args:
        maturities: (n_terms,) maturities in years.
        yields: (n_dates, n_terms) observed yields, no NaN.
        spec: Which model and how to estimate it.
        on_chunk: Cancellation/GIL-handover hook, called periodically.
        **kwargs: Estimator-specific extras, currently ``sigma_override``.

    Returns:
        The fitted panel, with residuals in the caller's original yield units.

    Raises:
        NotImplementedError: If ``spec.model`` has no registered fitter.
    """
    fitter = FITTERS.get(spec.model)
    if fitter is None:
        raise NotImplementedError(
            f"No fitter registered for {spec.model.value!r}. "
            f"Registered: {sorted(m.value for m in FITTERS)}."
        )
    return fitter(maturities, yields, spec, on_chunk=on_chunk, **kwargs)
