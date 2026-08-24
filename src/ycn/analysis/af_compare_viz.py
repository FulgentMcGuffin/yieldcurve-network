"""Plotnine renderers for the model comparison harness.

Uses plotnine rather than the matplotlib of :mod:`ycn.analysis.residual_viz`,
which is written the way it is because it renders into a Qt canvas. The harness
has no GUI, so it can use the grammar-of-graphics API directly.

Every renderer returns a ``ggplot``, so a caller can save it, draw it, or add
further layers.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_col,
    geom_hline,
    geom_line,
    geom_point,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient2,
    scale_x_log10,
    scale_y_continuous,
    theme,
    theme_minimal,
)

from .af_compare import ModelComparison

# Residuals are quoted in decimals, so basis points are the readable unit.
BP = 1e4


def plot_rmse_by_maturity(comparison: ModelComparison) -> ggplot:
    """Residual RMSE against maturity, one line per model.

    The headline chart. An arbitrage-free correction grows with the square of
    maturity, so if it is doing real work the lines separate at the long end and
    sit on top of each other at the short end.
    """
    data = comparison.residuals.select("model", "maturity_years", "rmse").with_columns(
        (pl.col("rmse") * BP).alias("rmse_bp")
    )

    return (
        ggplot(data, aes("maturity_years", "rmse_bp", colour="model"))
        + geom_line()
        + geom_point(size=1.8)
        + scale_x_log10()
        + labs(
            title="Residual RMSE by maturity",
            subtitle="Lower is better; arbitrage-free models should gain at the long end",
            x="Maturity (years, log scale)",
            y="RMSE (bp)",
            colour="Model",
        )
        + theme_minimal()
    )


def plot_residual_acf(comparison: ModelComparison, lag: int = 1) -> ggplot:
    """Residual autocorrelation at one lag, by maturity and model.

    Persistent residuals mean the curve model left systematic structure behind.
    A model that has absorbed the term-structure shape should sit near zero.
    """
    column = f"acf_{lag}"
    if column not in comparison.residuals.columns:
        raise ValueError(
            f"lag {lag} was not computed; available: "
            f"{[c for c in comparison.residuals.columns if c.startswith('acf_')]}"
        )

    data = comparison.residuals.select("model", "maturity_years", column)
    return (
        ggplot(data, aes("maturity_years", column, colour="model"))
        + geom_hline(yintercept=0.0, linetype="dashed", alpha=0.4)
        + geom_line()
        + geom_point(size=1.8)
        + scale_x_log10()
        + labs(
            title=f"Residual autocorrelation at lag {lag}",
            subtitle="Closer to zero means less structure left in the residual",
            x="Maturity (years, log scale)",
            y=f"ACF (lag {lag})",
            colour="Model",
        )
        + theme_minimal()
    )


def plot_acf_heatmap(comparison: ModelComparison) -> ggplot:
    """Every autocorrelation lag, as a model-by-lag grid faceted on maturity."""
    lags = sorted(
        int(c.split("_")[1])
        for c in comparison.residuals.columns
        if c.startswith("acf_")
    )
    if not lags:
        raise ValueError("no autocorrelation columns in the comparison")

    data = comparison.residuals.unpivot(
        index=["model", "maturity_years"],
        on=[f"acf_{lag}" for lag in lags],
        variable_name="lag",
        value_name="acf",
    ).with_columns(pl.col("lag").str.replace("acf_", "").cast(pl.Int32).alias("lag"))

    return (
        ggplot(data, aes("factor(lag)", "model", fill="acf"))
        + geom_tile()
        + facet_wrap("~maturity_years", labeller=lambda v: f"{float(v):g}Y")
        + scale_fill_gradient2(
            low="#2c7bb6", mid="#f7f7f7", high="#d7191c", midpoint=0.0
        )
        + labs(
            title="Residual autocorrelation by lag and maturity",
            x="Lag",
            y="Model",
            fill="ACF",
        )
        + theme_minimal()
        + theme(axis_text_y=element_text(size=7))
    )


def plot_network_metric(
    comparison: ModelComparison, metric: str = "modularity"
) -> ggplot:
    """One network metric across layers, coloured by model.

    Answers whether swapping the curve model changes the network's shape, not
    just the residuals feeding it.
    """
    if metric not in comparison.networks.columns:
        raise ValueError(
            f"{metric!r} is not a network metric; available: "
            f"{sorted(c for c in comparison.networks.columns if c != 'model')}"
        )

    data = comparison.networks.select("model", "label", metric).drop_nulls(metric)
    return (
        ggplot(data, aes("label", metric, colour="model", group="model"))
        + geom_line()
        + geom_point(size=2.0)
        + labs(
            title=f"Network {metric.replace('_', ' ')} by layer",
            x="Layer",
            y=metric.replace("_", " "),
            colour="Model",
        )
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )


def plot_edge_agreement(comparison: ModelComparison) -> ggplot:
    """Edge-set Jaccard similarity between model pairs, per layer.

    The question the whole harness exists to answer: a value near one means the
    models disagree about residual *levels* but agree about who moves together,
    so the network conclusions are robust to the choice of curve model. Values
    well below one mean the choice changes the answer.
    """
    data = (
        comparison.agreement.filter(pl.col("label") != "__all__")
        .drop_nulls("edge_jaccard")
        .with_columns(
            (pl.col("model_a") + pl.lit("  vs  ") + pl.col("model_b")).alias("pair")
        )
    )
    if data.is_empty():
        raise ValueError("no per-layer agreement rows; compare at least two models")

    return (
        ggplot(data, aes("label", "edge_jaccard", colour="pair", group="pair"))
        + geom_line()
        + geom_point(size=2.0)
        + scale_y_continuous(limits=(0.0, 1.0))
        + labs(
            title="Do the models agree on network edges?",
            subtitle="1.0 means identical edge sets; lower means the model choice matters",
            x="Layer",
            y="Edge-set Jaccard similarity",
            colour="Model pair",
        )
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )


def plot_yield_adjustment(comparison: ModelComparison) -> ggplot:
    """The fitted no-arbitrage correction by maturity, one bar group per model.

    Shows directly how much of the residual difference is the convexity term
    rather than anything estimated from the data's cross-section.
    """
    from .yield_curve import parse_term_years

    # Averaged per maturity rather than by stacking the arrays. Panels are
    # ragged, so different issuers carry different tenor counts and stacking
    # them would either raise or silently misalign one issuer's long end
    # against another's short end.
    rows: list[dict[str, object]] = []
    for name, fits in comparison.fits.items():
        by_maturity: dict[float, list[float]] = {}
        for fit in fits.values():
            terms = fit.diagnostics.get("terms")
            labels = terms if terms is not None else comparison.terms
            if len(labels) != len(fit.adjustment):
                continue
            for term, value in zip(labels, fit.adjustment):
                years = parse_term_years(term)
                if years is not None:
                    by_maturity.setdefault(float(years), []).append(float(value))

        rows.extend(
            {
                "model": name,
                "maturity_years": years,
                "adjustment_bp": float(np.mean(values)) * BP,
                "n_issuers": len(values),
            }
            for years, values in sorted(by_maturity.items())
        )

    data = pl.DataFrame(rows)
    if data.is_empty():
        raise ValueError("no fitted adjustments to plot")

    return (
        ggplot(data, aes("factor(maturity_years)", "adjustment_bp", fill="model"))
        + geom_col(position="dodge")
        + labs(
            title="Fitted no-arbitrage yield adjustment",
            subtitle="Convexity pushes arbitrage-free yields below Nelson-Siegel",
            x="Maturity (years)",
            y="Adjustment (bp)",
            fill="Model",
        )
        + theme_minimal()
    )
