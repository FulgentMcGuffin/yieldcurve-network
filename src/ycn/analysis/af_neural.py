"""Experimental neural residual model, after Gao & Hyndman (2025).

EXPERIMENTAL AND PARTIAL. Read this before using the output.

The source paper (arXiv:2511.17892) publishes neither code nor hyperparameters,
so this is a documented *subset* inspired by it rather than a reproduction, and
its numbers cannot be checked against anything published.

What is reproduced
------------------
* The dynamic Nelson-Siegel parameterisation, so the yields come out through the
  same loadings as every other model here and the residuals stay comparable.
* A recurrent (LSTM) latent state driving time-varying factors and volatility,
  rather than a single volatility matrix for the whole sample.
* A no-arbitrage restriction imposed as a **soft penalty** during training,
  weighted against reconstruction error, in the spirit of the paper's
  arbitrage-error regularisation.

What is simplified, and how
---------------------------
* **The restriction itself.** The paper penalises departures from the HJM
  forward-rate drift condition. This penalises departures from the AFNS
  structural link instead: the same volatility matrix must govern both the
  factor dynamics and the yield adjustment. Christensen, Diebold and Rudebusch
  make that tie explicit -- "the choice of the volatility matrix Sigma affects
  both the P-dynamics and the yield function through the yield-adjustment term"
  -- and a model that used one volatility for the curve and another for the
  dynamics would admit arbitrage. It is a genuine no-arbitrage condition in the
  same family, and it reuses the closed form already validated elsewhere in this
  package; it is not the paper's condition.
* No extended Kalman filter, no particle filter, no convolutional LSTM over the
  maturity axis, and no Bayesian derivation of the update.
* All hyperparameters are chosen here for stability, not fidelity.

Why the residuals are not directly comparable
---------------------------------------------
These are **filtered, in-sample** residuals from a model trained on the whole
panel, not the out-of-sample residuals of a cross-sectional fit. They are
optimistically small for that reason alone. Both this module and the comparison
harness flag them, and the harness prints a warning when any are present.

Requires the optional extra::

    uv sync --extra neural
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .af_loadings import afns_loadings, afns_yield_adjustment, decay_to_lambda
from .af_models import ModelSpec, NeuralSpec, PanelFit, detect_yield_scale

# Below this many observations the network has nothing to learn from and the
# two-step estimator is the better answer.
MIN_DATES_FOR_TRAINING = 60

# Bounds on the learned log factor volatility, in decimals per year. The upper
# bound corresponds to ~3%, comfortably above the ~0.7-2.5% CDR (2011) report;
# the initial value sits at ~0.7%, so training starts where the answer lives.
LOG_SIGMA_MIN = -11.0
LOG_SIGMA_MAX = -3.5
LOG_SIGMA_INIT = -5.0

_IMPORT_HINT = (
    "The neural HJM model requires torch, which is an optional extra.\n"
    "Install it with:  uv sync --extra neural"
)


def _require_torch():
    """Import torch, or explain how to get it.

    Imported lazily so the core package stays torch-free and ``uv sync`` without
    the extra keeps working.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_IMPORT_HINT) from exc
    return torch


def adjustment_shapes(maturities: np.ndarray, decay: float) -> np.ndarray:
    """Per-maturity basis of the yield adjustment, one column per factor.

    For a diagonal volatility matrix the AFNS adjustment is exactly linear in the
    squared volatilities, so it decomposes as ``adj = shapes @ sigma**2``. That
    turns a closed form full of exponentials into a matrix product, which is both
    differentiable and cheap inside a training loop -- and, more importantly,
    reuses :func:`~ycn.analysis.af_loadings.afns_yield_adjustment` rather than
    restating it, so the two cannot diverge.

    Args:
        maturities: (n_terms,) maturities in years.
        decay: Decay in the repo's **scale** convention.

    Returns:
        (n_terms, 3) array whose product with ``sigma**2`` is the adjustment.
    """
    lambda_ = decay_to_lambda(decay)
    columns = []
    for index in range(3):
        unit = np.zeros(3)
        unit[index] = 1.0
        columns.append(afns_yield_adjustment(maturities, lambda_, np.diag(unit)))
    return np.column_stack(columns)


class _CurveNet:
    """LSTM producing per-date factors and volatilities from a yield panel.

    Defined as a factory rather than a module subclass so that importing this
    file does not require torch.
    """

    def __new__(cls, torch, n_terms: int, spec: NeuralSpec):
        nn = torch.nn

        class CurveNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_terms,
                    hidden_size=spec.hidden_size,
                    num_layers=spec.n_layers,
                    batch_first=True,
                )
                self.factor_head = nn.Linear(spec.hidden_size, 3)
                # One volatility per factor, shared across dates within a pass
                # but re-derived from the sequence each forward call.
                self.log_sigma_head = nn.Linear(spec.hidden_size, 3)
                # Start from a plausible annual factor volatility (~0.7%) with a
                # flat head, so the first adjustment is the right order of
                # magnitude. From a default initialisation the volatility starts
                # anywhere in the clamp range, and at the top of that range the
                # adjustment reaches hundreds of percentage points and swamps the
                # curve it is supposed to correct.
                nn.init.zeros_(self.log_sigma_head.weight)
                nn.init.constant_(self.log_sigma_head.bias, LOG_SIGMA_INIT)

            def forward(self, yields, anchor, spread):
                hidden, _ = self.lstm(yields.unsqueeze(0))
                hidden = hidden.squeeze(0)
                # The head predicts a deviation from the sample's own level and
                # scale rather than the factor outright. Without this anchor the
                # level factor has to travel from zero to roughly 0.02 through a
                # loss whose gradient at that distance is tiny, and training
                # stalls long before it arrives.
                factors = anchor + self.factor_head(hidden) * spread
                # Clamped so an early bad step cannot produce an adjustment that
                # dwarfs the yields and stalls training. The upper bound is a
                # ~3% annual factor volatility, comfortably above anything CDR
                # report and far below the level at which the convexity term
                # overwhelms the curve.
                log_sigma = torch.clamp(
                    self.log_sigma_head(hidden), LOG_SIGMA_MIN, LOG_SIGMA_MAX
                )
                return factors, torch.exp(log_sigma)

        return CurveNet()


def fit_neural_hjm(
    maturities: np.ndarray,
    yields: np.ndarray,
    spec: ModelSpec,
    *,
    on_chunk: Callable[[], None] | None = None,
    sigma_override: np.ndarray | None = None,
    **kwargs: object,
) -> PanelFit:
    """Fit the experimental neural residual model to one issuer's panel.

    Args:
        maturities: (n_terms,) maturities in years.
        yields: (n_dates, n_terms) observed yields in the caller's units.
        spec: Model settings; ``spec.neural`` carries the hyperparameters.
        on_chunk: Cancellation/GIL-handover hook, called periodically.
        sigma_override: Ignored -- this model learns its own volatility path.
        **kwargs: Ignored.

    Returns:
        The fitted panel. ``diagnostics["filtered_residuals"]`` is ``True``, and
        callers comparing this against cross-sectional models must say so.

    Raises:
        ImportError: If torch is not installed.
    """
    del sigma_override, kwargs
    torch = _require_torch()

    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)
    n_dates, n_terms = yields.shape

    if n_dates < MIN_DATES_FOR_TRAINING:
        from .af_fit import fit_two_step
        from .af_models import ResidualModel

        # This model is built on the AFNS loadings and adjustment, so AFNS is
        # the right thing to fall back to. `NEURAL_HJM` has no two-step kernel
        # of its own and would raise.
        fallback = fit_two_step(
            maturities,
            yields,
            replace(spec, model=ResidualModel.AFNS),
            on_chunk=on_chunk,
        )
        fallback.diagnostics["model"] = spec.model.value
        fallback.diagnostics["estimator"] = "two_step_fallback"
        fallback.diagnostics["fallback_reason"] = (
            f"only {n_dates} dates; need {MIN_DATES_FOR_TRAINING} to train"
        )
        fallback.diagnostics["filtered_residuals"] = False
        fallback.diagnostics["experimental"] = True
        return fallback

    neural = spec.neural or NeuralSpec()
    scale = (
        spec.yield_scale if spec.yield_scale is not None else detect_yield_scale(yields)
    )
    decay = spec.decay if spec.decay is not None else 1.0

    torch.manual_seed(spec.seed)
    scaled = torch.tensor(yields * scale, dtype=torch.float32)
    design = torch.tensor(afns_loadings(maturities, decay), dtype=torch.float32)
    shapes = torch.tensor(adjustment_shapes(maturities, decay), dtype=torch.float32)

    # Centring keeps the LSTM input near unit scale; the offset is added back
    # implicitly because the level factor is free.
    centre = scaled.mean()
    spread = scaled.std().clamp(min=1e-6)
    normalised = (scaled - centre) / spread

    model = _CurveNet(torch, n_terms, neural)
    optimiser = torch.optim.Adam(model.parameters(), lr=neural.learning_rate)
    dt = torch.tensor(float(spec.dt), dtype=torch.float32)

    # Anchor the level factor at the sample mean; the slope and curvature start
    # at zero, which is already the right neighbourhood.
    anchor = torch.tensor([float(centre), 0.0, 0.0], dtype=torch.float32)
    # Reconstruction is measured relative to the panel's own variance. Yields in
    # decimals give squared errors of order 1e-8, whereas the arbitrage penalty
    # below is normalised to order one -- left unscaled, the penalty dominates
    # the loss completely and the network never learns to fit the curve at all.
    yield_variance = torch.clamp(scaled.var(), min=1e-12)

    history: list[float] = []
    for epoch in range(neural.epochs):
        optimiser.zero_grad()
        factors, sigma = model(normalised, anchor, spread)

        adjustment = shapes @ (sigma**2).T  # (n_terms, n_dates)
        fitted = factors @ design.T + adjustment.T
        reconstruction = torch.mean((scaled - fitted) ** 2) / yield_variance

        # No-arbitrage tie: the volatility driving the yield adjustment must be
        # the volatility of the factor process itself. See the module docstring
        # for how this differs from the paper's HJM drift condition.
        #
        # Compared in log space. Both sides are variances of order 1e-5, so a
        # squared difference is ~1e-10 while the reconstruction term is order
        # one; and rescaling by either side's magnitude divides by something
        # near zero and explodes. Logs make the penalty scale-free, symmetric in
        # over- and under-statement, and bounded for any positive input.
        increments = factors[1:] - factors[:-1]
        realised = increments.var(dim=0, unbiased=False) / dt
        implied = (sigma[1:] ** 2).mean(dim=0)
        floor = 1e-14
        arbitrage = torch.mean(
            (torch.log(realised + floor) - torch.log(implied + floor)) ** 2
        )

        smoothness = torch.mean(increments**2)

        loss = (
            reconstruction
            + neural.drift_weight * arbitrage
            + neural.smooth_weight * smoothness
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        history.append(float(loss.detach()))
        if on_chunk is not None and (epoch + 1) % 25 == 0:
            on_chunk()

    with torch.no_grad():
        factors, sigma = model(normalised, anchor, spread)
        adjustment = shapes @ (sigma**2).T
        fitted = factors @ design.T + adjustment.T
        residuals = (scaled - fitted).numpy().astype(float)
        factors_out = factors.numpy().astype(float)
        mean_sigma = sigma.mean(dim=0).numpy().astype(float)
        mean_adjustment = adjustment.mean(dim=1).numpy().astype(float)

    return PanelFit(
        residuals=residuals / scale,
        factors=factors_out / scale,
        decay=np.full(n_dates, decay, dtype=float),
        rmse=np.sqrt(np.mean(residuals**2, axis=1)) / scale,
        adjustment=mean_adjustment / scale,
        sigma=np.diag(mean_sigma),
        kappa_p=None,
        diagnostics={
            "model": spec.model.value,
            "estimator": "neural_hjm",
            "n_dates": n_dates,
            "decay": decay,
            "yield_scale": scale,
            "sigma_scope": spec.sigma_scope.value,
            "epochs": neural.epochs,
            "final_loss": history[-1] if history else float("nan"),
            "initial_loss": history[0] if history else float("nan"),
            "loss_decreased": bool(len(history) > 1 and history[-1] < history[0]),
            "n_iterations": neural.epochs,
            "sigma_converged": True,
            # These are in-sample filtered residuals, not cross-sectional ones.
            "filtered_residuals": True,
            "experimental": True,
        },
    )
