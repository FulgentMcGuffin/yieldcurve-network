"""Side-by-side comparison of residual-producing curve models.

Fits each model once, then reports what actually differs: how large the
residuals are by maturity, how much structure they still contain, what the
estimator inferred, and -- the question that motivates the whole exercise --
whether the resulting networks disagree about *who deviates together* or only
about the level of the residuals.

Each model's cube is built exactly once and reused for both the residual
diagnostics and the network pass, so adding a model costs one fit rather than
one fit per consumer.

A result worth knowing before reading any output
------------------------------------------------
For any model whose yield adjustment is **constant over time** -- which covers
AFNS and DTAFNS as estimated here, since the volatility matrix is fitted once
over the whole sample -- the residual correlation networks are *provably
identical* to the plain Nelson-Siegel ones.

The reason is short. Writing ``P`` for the least-squares residual maker,

    resid_af = (y - adj) - X (X\\(y - adj)) = resid_ns - P(adj)

and ``P(adj)`` carries no time index, so each issuer-tenor residual series is
shifted by a constant. Pearson correlation is invariant under adding a constant,
so every pairwise correlation, every edge and every network metric is unchanged.

So the arbitrage-free models change residual *levels* substantially -- typically
an order of magnitude at the long end -- while leaving *who deviates together*
exactly as it was. Their value here is in fit quality and diagnostics, not in
network topology. What would move the networks is an adjustment that varies over
time: a rolling-window volatility estimate, or the filtered residuals of the
neural model. ``tests/test_af_compare.py`` pins this invariance so that if a
future model does move the networks, it is because it genuinely differs rather
than because something broke.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import polars as pl
from scipy import stats

from .af_models import ModelSpec, PanelFit, ResidualModel
from .residual_networks import (
    ProgressCallback,
    ResidualNetworkResult,
    StatusCallback,
    _networks_from_cube,
    residual_cube,
)
from .yield_curve import CurvePanel, NetworkKind, parse_term_years

# Residuals from a filtered state-space model are not the same statistical
# object as those from a cross-sectional fit, so any model listed here is
# flagged in the output rather than silently compared like for like.
FILTERED_RESIDUAL_MODELS = frozenset({ResidualModel.NEURAL_HJM})


@dataclass
class ModelComparison:
    """Everything the harness produces, one row per model where applicable."""

    residuals: pl.DataFrame
    factors: pl.DataFrame
    networks: pl.DataFrame
    agreement: pl.DataFrame
    cubes: dict[str, np.ndarray] = field(default_factory=dict)
    results: dict[str, ResidualNetworkResult] = field(default_factory=dict)
    fits: dict[str, dict[str, PanelFit]] = field(default_factory=dict)
    issuers: list[str] = field(default_factory=list)
    dates: list = field(default_factory=list)
    terms: list[str] = field(default_factory=list)

    @property
    def graphs(self) -> dict[tuple[str, str], nx.Graph]:
        """Every graph keyed by ``(model, label)``."""
        return {
            (name, label): graph
            for name, result in self.results.items()
            for label, graph in result.graphs.items()
        }


def _autocorrelations(series: np.ndarray, lags: int) -> tuple[list[float], float]:
    """Sample autocorrelations and a Ljung-Box p-value for one residual series.

    A curve model that has captured the shape of the term structure should leave
    residuals with little serial dependence. Persistent residuals are the
    signature of a missing systematic component -- which is precisely what the
    arbitrage-free adjustment is meant to supply.

    Args:
        series: One residual series, possibly containing NaN.
        lags: Number of lags to report.

    Returns:
        ``(acf, p_value)``. Both are NaN-filled when the series is too short.
    """
    values = series[~np.isnan(series)]
    n = values.size
    if n < max(3 * lags, 12):
        return [float("nan")] * lags, float("nan")

    centred = values - values.mean()
    denominator = float(centred @ centred)
    if denominator <= 0.0:
        return [float("nan")] * lags, float("nan")

    acf = [
        float(centred[lag:] @ centred[:-lag] / denominator)
        for lag in range(1, lags + 1)
    ]
    finite = [r for r in acf if np.isfinite(r)]
    if len(finite) < lags:
        return acf, float("nan")

    statistic = (
        n * (n + 2) * sum(r**2 / (n - lag) for lag, r in enumerate(acf, start=1))
    )
    return acf, float(stats.chi2.sf(statistic, df=lags))


def _residual_diagnostics(
    name: str,
    spec: ModelSpec,
    cube: np.ndarray,
    terms: list[str],
    lags: int,
) -> list[dict[str, object]]:
    """Per-maturity residual size and serial dependence for one model."""
    rows: list[dict[str, object]] = []
    for index, term in enumerate(terms):
        slab = cube[:, :, index]
        finite = slab[~np.isnan(slab)]

        per_issuer = [_autocorrelations(slab[i], lags) for i in range(slab.shape[0])]
        acfs = np.array([entry[0] for entry in per_issuer], dtype=float)
        pvalues = np.array([entry[1] for entry in per_issuer], dtype=float)

        row: dict[str, object] = {
            "model": name,
            "model_kind": spec.model.value,
            "sigma_scope": spec.sigma_scope.value,
            "term": term,
            "maturity_years": parse_term_years(term),
            "n_obs": int(finite.size),
            "rmse": float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan"),
            "mae": float(np.mean(np.abs(finite))) if finite.size else float("nan"),
            "filtered_residuals": spec.model in FILTERED_RESIDUAL_MODELS,
        }
        for lag in range(lags):
            column = acfs[:, lag] if acfs.size else np.array([np.nan])
            row[f"acf_{lag + 1}"] = (
                float(np.nanmean(column))
                if np.any(np.isfinite(column))
                else float("nan")
            )
        row["ljung_box_p"] = (
            float(np.nanmean(pvalues)) if np.any(np.isfinite(pvalues)) else float("nan")
        )
        rows.append(row)
    return rows


def _factor_diagnostics(
    name: str, spec: ModelSpec, fits: dict[str, PanelFit], skipped: list[str]
) -> dict[str, object]:
    """What the estimator inferred, averaged across issuers."""
    row: dict[str, object] = {
        "model": name,
        "model_kind": spec.model.value,
        "sigma_scope": spec.sigma_scope.value,
        "estimator": spec.estimator.value,
        "decay": spec.decay,
        "n_issuers": len(fits),
        "n_skipped": len(skipped),
    }
    if not fits:
        return row

    diagonals = np.array([np.diag(fit.sigma) for fit in fits.values()], dtype=float)
    for index, label in enumerate(("11", "22", "33")):
        row[f"sigma_{label}"] = float(np.mean(diagonals[:, index]))

    row["mean_rmse"] = float(np.mean([np.mean(fit.rmse) for fit in fits.values()]))
    row["max_abs_adjustment"] = float(
        np.max([np.max(np.abs(fit.adjustment)) for fit in fits.values()])
    )
    row["n_iterations"] = int(
        np.max([fit.diagnostics.get("n_iterations", 0) for fit in fits.values()])
    )
    row["sigma_converged"] = bool(
        all(fit.diagnostics.get("sigma_converged", True) for fit in fits.values())
    )
    row["yield_scale"] = float(
        np.median([fit.diagnostics.get("yield_scale", 1.0) for fit in fits.values()])
    )

    eigenvalues = [
        np.sort(np.real(np.linalg.eigvals(fit.kappa_p)))
        for fit in fits.values()
        if fit.kappa_p is not None
    ]
    if eigenvalues:
        stacked = np.array(eigenvalues, dtype=float)
        for index in range(3):
            row[f"kappa_eig_{index + 1}"] = float(np.mean(stacked[:, index]))
    return row


def _agreement(
    left_name: str,
    right_name: str,
    left_cube: np.ndarray,
    right_cube: np.ndarray,
    left_result: ResidualNetworkResult,
    right_result: ResidualNetworkResult,
) -> list[dict[str, object]]:
    """How far two models disagree, on residuals and on network structure.

    The residual columns say whether the models put the deviations in different
    places; the Jaccard column says whether that changes the answer the networks
    give. A pair can agree closely on residual levels and still disagree on
    edges, or the reverse -- which is exactly what makes the comparison worth
    running rather than assuming.
    """
    shared = ~np.isnan(left_cube) & ~np.isnan(right_cube)
    left_flat, right_flat = left_cube[shared], right_cube[shared]
    if left_flat.size >= 2 and np.std(left_flat) > 0 and np.std(right_flat) > 0:
        correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
    else:
        correlation = float("nan")

    overall = {
        "model_a": left_name,
        "model_b": right_name,
        "label": "__all__",
        "residual_corr": correlation,
        "mean_abs_diff": (
            float(np.mean(np.abs(left_flat - right_flat)))
            if left_flat.size
            else float("nan")
        ),
        "edge_jaccard": float("nan"),
    }

    rows = [overall]
    for label in left_result.label_order:
        left_graph = left_result.graphs.get(label)
        right_graph = right_result.graphs.get(label)
        if left_graph is None or right_graph is None:
            continue
        left_edges = {frozenset(edge) for edge in left_graph.edges()}
        right_edges = {frozenset(edge) for edge in right_graph.edges()}
        union = left_edges | right_edges
        rows.append(
            {
                "model_a": left_name,
                "model_b": right_name,
                "label": label,
                "residual_corr": float("nan"),
                "mean_abs_diff": float("nan"),
                # Two empty edge sets are vacuously identical, but reporting 1.0
                # would read as "the models agree perfectly" when it means "there
                # was nothing to agree about". NaN forces that to be noticed.
                "edge_jaccard": (
                    float(len(left_edges & right_edges) / len(union))
                    if union
                    else float("nan")
                ),
                "n_edges_a": len(left_edges),
                "n_edges_b": len(right_edges),
            }
        )
    return rows


def compare_residual_models(
    long: pl.DataFrame,
    panel: CurvePanel,
    kind: NetworkKind,
    date_column: str,
    *,
    specs: Sequence[ModelSpec],
    threshold: float = 0.3,
    acf_lags: int = 5,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> ModelComparison:
    """Fit every model once and report how they differ.

    Args:
        long: Long ``(date, issuer, term, rate)`` frame.
        panel: Column-role assignment.
        kind: Which axis becomes the network layer.
        date_column: Date column name.
        specs: Models to compare. Names come from :attr:`ModelSpec.name`, so a
            sweep over decays or volatility scopes stays distinguishable.
        threshold: Absolute correlation above which a network edge is kept.
        acf_lags: Autocorrelation lags to report per maturity.
        progress: Per-issuer progress callback.
        status: Human-readable status callback.

    Returns:
        The assembled comparison.

    Raises:
        ValueError: If ``specs`` is empty or contains duplicate names.
    """
    if not specs:
        raise ValueError("compare_residual_models needs at least one ModelSpec")
    names = [spec.name for spec in specs]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"ModelSpec names must be unique; repeated: {sorted(duplicates)}. "
            "Vary the model, volatility scope, estimator or decay to separate them."
        )

    comparison = ModelComparison(
        residuals=pl.DataFrame(),
        factors=pl.DataFrame(),
        networks=pl.DataFrame(),
        agreement=pl.DataFrame(),
    )
    residual_rows: list[dict[str, object]] = []
    factor_rows: list[dict[str, object]] = []
    network_frames: list[pl.DataFrame] = []

    for spec, name in zip(specs, names):
        if status is not None:
            status(f"compare: fitting {name}…")

        fits: dict[str, PanelFit] = {}
        # Every model, including plain NS, goes through the registry here. The
        # registry route is proven equal to the legacy path by
        # tests/test_afns_equivalence.py, and routing all of them the same way
        # keeps `fits` populated for NS so the factor table has a baseline row.
        issuers, dates, cube, terms, skipped = residual_cube(
            long,
            panel,
            date_column,
            decay=spec.decay,
            model=spec,
            fits_out=fits,
            progress=progress,
            status=status,
        )

        result = _networks_from_cube(
            issuers,
            dates,
            cube,
            terms,
            skipped,
            long=long,
            panel=panel,
            kind=kind,
            date_column=date_column,
            threshold=threshold,
            progress=progress,
            status=status,
        )

        comparison.cubes[name] = cube
        comparison.results[name] = result
        comparison.fits[name] = fits
        comparison.issuers = issuers
        comparison.dates = dates
        comparison.terms = terms

        residual_rows.extend(_residual_diagnostics(name, spec, cube, terms, acf_lags))
        factor_rows.append(_factor_diagnostics(name, spec, fits, skipped))
        if not result.metrics.is_empty():
            network_frames.append(
                result.metrics.with_columns(pl.lit(name).alias("model"))
            )

    agreement_rows: list[dict[str, object]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            agreement_rows.extend(
                _agreement(
                    left,
                    right,
                    comparison.cubes[left],
                    comparison.cubes[right],
                    comparison.results[left],
                    comparison.results[right],
                )
            )

    comparison.residuals = pl.DataFrame(residual_rows)
    comparison.factors = pl.DataFrame(factor_rows)
    comparison.networks = (
        pl.concat(network_frames, how="diagonal") if network_frames else pl.DataFrame()
    )
    comparison.agreement = (
        pl.DataFrame(agreement_rows) if agreement_rows else pl.DataFrame()
    )
    return comparison


def summarize(comparison: ModelComparison) -> pl.DataFrame:
    """One headline row per model: residual size, persistence, correction size.

    ``rmse_long`` is the number to watch. The arbitrage-free correction grows
    with the square of maturity, so if it is doing real work it shows up at the
    long end and barely moves the short end.
    """
    if comparison.residuals.is_empty():
        return pl.DataFrame()

    acf_columns = [c for c in comparison.residuals.columns if c.startswith("acf_")]
    first_lag = "acf_1" if "acf_1" in acf_columns else None

    aggregations = [
        pl.col("rmse").mean().alias("rmse_all"),
        pl.col("rmse")
        .filter(pl.col("maturity_years") >= 15.0)
        .mean()
        .alias("rmse_long"),
        pl.col("rmse")
        .filter(pl.col("maturity_years") <= 2.0)
        .mean()
        .alias("rmse_short"),
        pl.col("n_obs").sum().alias("n_obs"),
        pl.col("filtered_residuals").first().alias("filtered_residuals"),
    ]
    if first_lag is not None:
        aggregations.append(pl.col(first_lag).abs().mean().alias("mean_abs_acf1"))

    summary = comparison.residuals.group_by("model").agg(aggregations)
    if not comparison.factors.is_empty():
        keep = [
            c
            for c in ("model", "max_abs_adjustment", "sigma_11", "n_skipped")
            if c in comparison.factors.columns
        ]
        summary = summary.join(comparison.factors.select(keep), on="model", how="left")
    return summary.sort("rmse_long")
