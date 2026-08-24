"""Temporal evolution of the multi-layer network and of the curve's factors.

Three independent time series come out of one rolling-window pass:

1. **Multiplex structure** (notebook chapters C/F/G) -- rebuild the whole
   multiplex inside each window, then track intra/inter edge composition and
   the community count *k* each of the five k-selection methods picks.
2. **Curve factors** (chapter J) -- Nelson-Siegel level/slope/curvature of the
   market-average curve, their volatility, and a Gaussian-mixture regime label
   per window.
3. **Correlation stress** (chapter K) -- stability of the residual correlation
   structure, condensed into a 0-100 stress indicator.

Windows come from :func:`evolution.generate_windows`, so the MLN evolution uses
the same rolling/expanding schedule as the single-network evolution did.

Pure polars/numpy/networkx, safe to run on a worker thread.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import count

import networkx as nx
import numpy as np
import polars as pl

from .af_models import ModelSpec, fit_panel
from .evolution import CommunityMethod, EvolutionConfig, generate_windows
from .mln import (
    MLNConfig,
    build_layer_graphs,
    build_multilayer_network,
    layer_values_of,
)
from .multilayer_communities import METHOD_LABELS, detect_multilayer_communities
from .network import pivot_to_wide
from .residual_networks import (
    _canonical_maturities,
    _canonical_maturity_values,
    _dense_rows,
)
from .yield_curve import CurvePanel
from .yield_curve_factors import (
    classify_yield_curve_regimes,
    compute_factor_trajectories,
    fit_nelson_siegel_batch,
)
from .config import PipelineConfig

ProgressCallback = Callable[[int, int, str], None]
StatusCallback = Callable[[str], None]

# ASE latent dimension for multiplex community detection, and the cap on k.
# Mirrors the notebook's ASE_N_COMPONENTS / MAX_COMMUNITIES.
ASE_N_COMPONENTS = 8

# Stress indicator above this counts as a stressed window.
STRESS_THRESHOLD = 50.0

# Fallback rolling schedule for the factor and stress trajectories, used only
# when a caller does not supply one. The GUI always passes the user's evolution
# window/step, so these two series honour the Evolution Settings dialog exactly
# as the multiplex windows do.
FACTOR_WINDOW = 30
FACTOR_STEP = 10

# The four stress series the Cov / Cov(t) tabs plot, with display labels.
STRESS_SERIES: dict[str, str] = {
    "avg_abs_corr": "Average |correlation|",
    "corr_variance": "Correlation variance",
    "n_sig_edges": "Strongly correlated pairs (|r| > 0.5)",
    "stress_pct": "Stress indicator (0-100)",
}


@dataclass
class MLNEvolutionResult:
    """Everything the four evolution tabs render."""

    edge_types: pl.DataFrame  # window_idx, date_end, intra/inter counts + pct
    community_k: pl.DataFrame  # method, window_idx, date_end, n_clusters, score
    communities: pl.DataFrame  # method, window, node, layer, community
    factors: pl.DataFrame  # window_idx, date_end, {level,slope,curvature}_{mean,std}
    regimes: pl.DataFrame  # window_idx, date_end, regime
    stress: pl.DataFrame  # window_idx, date_end, the four STRESS_SERIES + band
    windows: list[tuple[date, date]] = field(default_factory=list)
    layer_column: str = "layer"
    node_column: str = "node"


def as_date(value) -> date:
    """Coerce a date-ish value to ``datetime.date``.

    ``polars.Series.to_numpy()`` on a Date column yields object-dtype numpy
    scalars, which polars then refuses to re-ingest ("cannot cast 'Object'
    type"). Anything crossing back into a DataFrame goes through here.
    """
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]").astype(date)
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _window_bounds(
    long: pl.DataFrame, date_column: str, cfg: EvolutionConfig
) -> list[tuple[date, date]]:
    """The rolling/expanding ``(start, end)`` schedule for this panel."""
    dates = sorted(long.get_column(date_column).unique().to_list())
    return [
        (start, end)
        for start, end, _ in generate_windows(
            dates,
            window_size=cfg.window_size,
            step=cfg.step,
            expanding=cfg.expanding,
        )
    ]


def compute_multiplex_evolution(
    long: pl.DataFrame,
    cfg: PipelineConfig,
    mcfg: MLNConfig,
    evo: EvolutionConfig,
    *,
    edge_settings: dict | None = None,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[tuple[date, date]]]:
    """Rebuild the multiplex per window; track edge types and community k.

    Returns ``(edge_types, community_k, communities, windows)``. Community
    detection runs all five k-selection methods on every window so the
    "#Communities by Method" panel can compare them; each window's k depends
    only on that window, so there is no lookahead.
    """
    schedule = _window_bounds(long, cfg.date_column, evo)
    if not schedule:
        n_dates = long.get_column(cfg.date_column).n_unique()
        raise ValueError(
            f"No windows: {n_dates} dates is fewer than the window size "
            f"({evo.window_size})."
        )

    layer_column = mcfg.layer_column
    edge_rows: list[dict] = []
    k_rows: list[dict] = []
    community_rows: list[dict] = []
    windows: list[tuple[date, date]] = []
    total = len(schedule)

    for w_idx, (start, end) in enumerate(schedule):
        windows.append((start, end))
        if status is not None:
            status(f"Evolution: window {w_idx + 1}/{total} ({start} → {end})")

        sub = long.filter(
            (pl.col(cfg.date_column) >= start) & (pl.col(cfg.date_column) <= end)
        )
        values = layer_values_of(sub, layer_column)
        if len(values) < 2:
            continue

        # Forward a checkpoint (not the real scale) into the per-pair loop:
        # without it the only cancellation point is the end of a whole window,
        # which on a large panel is minutes, and the GIL is never handed back.
        def _checkpoint(_done=0, _total=0, _desc="", _w=w_idx) -> None:
            if progress is not None:
                progress(_w, total, f"window {_w + 1}/{total}")

        layer_graphs = build_layer_graphs(
            sub,
            cfg,
            layer_column,
            layer_values=values,
            progress=_checkpoint,
            edge_settings=edge_settings,
        )
        multiplex = build_multilayer_network(layer_graphs, values)

        intra = sum(
            1 for _, _, d in multiplex.edges(data=True) if d.get("layer") == "intra"
        )
        inter = sum(
            1 for _, _, d in multiplex.edges(data=True) if d.get("layer") == "inter"
        )
        edge_rows.append(
            {
                "window_idx": w_idx,
                "date_start": start,
                "date_end": end,
                "intra_edges": intra,
                "inter_edges": inter,
                "total_edges": intra + inter,
                "n_nodes": multiplex.number_of_nodes(),
            }
        )

        if multiplex.number_of_nodes() >= max(evo.min_nodes, 3):
            for method in CommunityMethod:
                try:
                    found = detect_multilayer_communities(
                        multiplex,
                        method,
                        max_clusters=mcfg.max_communities,
                        ase_n_components=ASE_N_COMPONENTS,
                    )
                except Exception as exc:  # noqa: BLE001 -- one method must not abort
                    if status is not None:
                        status(
                            f"Evolution: {METHOD_LABELS[method]} failed on window "
                            f"{w_idx} ({exc})"
                        )
                    continue
                k_rows.append(
                    {
                        "method": METHOD_LABELS[method],
                        "window_idx": w_idx,
                        "date_end": end,
                        "n_clusters": int(found.get("n_clusters", 0)),
                        "score": float(
                            found.get("score", found.get("inertia", float("nan")))
                            or float("nan")
                        ),
                    }
                )
                for (node, layer), community in found.get("communities", {}).items():
                    community_rows.append(
                        {
                            "method": METHOD_LABELS[method],
                            "window_idx": w_idx,
                            "date_end": end,
                            "node": str(node),
                            "layer": str(layer),
                            "community": int(community),
                        }
                    )

        if progress is not None:
            progress(w_idx + 1, total, f"window {w_idx + 1}/{total}")

    edge_types = pl.DataFrame(edge_rows)
    if not edge_types.is_empty():
        edge_types = edge_types.with_columns(
            (100.0 * pl.col("intra_edges") / pl.col("total_edges")).alias("pct_intra"),
            (100.0 * pl.col("inter_edges") / pl.col("total_edges")).alias("pct_inter"),
        )
    return edge_types, pl.DataFrame(k_rows), pl.DataFrame(community_rows), windows


def compute_curve_factors(
    long: pl.DataFrame,
    panel: CurvePanel,
    date_column: str,
    *,
    model: ModelSpec | None = None,
    window_size: int = 30,
    step_size: int = 10,
    n_regimes: int = 3,
    status: StatusCallback | None = None,
    log_prefix: str = "Evolution",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Factor trajectories of the market-average curve.

    The curve is averaged across issuers first, so this describes the market
    rather than any single issuer -- matching the notebook's ``ns_market``.
    ``model=None`` -- the default -- fits plain Nelson-Siegel independently
    per date, unchanged from before this function took a ``model`` argument.
    A :class:`ModelSpec` instead fits that model **jointly across the whole
    series** (``af_models.fit_panel`` treats the market-average curve as a
    single-issuer panel), which is what lets a model with a genuinely
    time-varying correction -- Neural HJM -- produce a factor trajectory that
    differs from the Nelson-Siegel one, rather than a fixed relabelling of it.

    ``log_prefix`` tags every status line this function emits (default
    ``"Evolution"``, matching the NS evolution worker); pass ``"Neural"`` when
    called from the Neural-HJM worker so the process log stays as clearly
    attributed as it already is for the rest of that stage. A model-based fit
    also reports a checkpoint every 25 training epochs -- otherwise the market
    curve's fit, unlike the fast per-date NS path, gives no sign of life until
    it finishes.
    """
    market = (
        long.group_by([date_column, panel.term_column])
        .agg(pl.col(panel.rate_column).mean().alias(panel.rate_column))
        .sort(date_column)
    )
    wide = pivot_to_wide(
        market,
        date_column=date_column,
        name_column=panel.term_column,
        value_column=panel.rate_column,
    )
    maturities = [c for c in wide.columns if c != date_column]
    if len(maturities) < 4:
        raise ValueError(
            f"Nelson-Siegel needs at least 4 maturities; got {len(maturities)}."
        )

    if model is None:
        if status is not None:
            status(f"{log_prefix}: fitting market-average Nelson-Siegel factors…")
        ns_market = fit_nelson_siegel_batch(
            wide, date_col=date_column, maturities=maturities, decay=1.0
        )
        if ns_market.is_empty():
            raise ValueError(
                "Nelson-Siegel could not be fitted on any date. Every date "
                "needs a complete market-average curve across the selected "
                "maturities."
            )
        market_factors = ns_market.select([date_column, "level", "slope", "curvature"])
    else:
        if status is not None:
            status(f"{log_prefix}: fitting market-average {model.model.label} factors…")
        canonical = _canonical_maturities(maturities)
        canon_wide = wide.rename(canonical)
        terms = [canonical[m] for m in maturities]
        dates, yields = _dense_rows(canon_wide, date_column, terms)
        if not dates:
            raise ValueError(
                "No date has a complete market-average curve across the "
                "selected maturities."
            )
        checkpoints = count(1)
        on_chunk = (
            (
                lambda: status(
                    f"{log_prefix}: market-average {model.model.label} fit "
                    f"in progress (checkpoint {next(checkpoints)})…"
                )
            )
            if status is not None
            else None
        )
        fit = fit_panel(
            _canonical_maturity_values(terms), yields, model, on_chunk=on_chunk
        )
        market_factors = pl.DataFrame(
            {
                date_column: dates,
                "level": fit.factors[:, 0],
                "slope": fit.factors[:, 1],
                "curvature": fit.factors[:, 2],
            }
        )

    # compute_factor_trajectories selects a column literally named "date".
    if date_column != "date":
        market_factors = market_factors.rename({date_column: "date"})

    stats = compute_factor_trajectories(
        market_factors, window_size=window_size, step_size=step_size
    )
    if not stats.get("windows"):
        raise ValueError(
            "No factor windows produced; widen the date range or reduce the "
            "evolution window size."
        )

    factors = pl.DataFrame(
        {
            "window_idx": np.arange(len(stats["windows"])),
            "date_end": [as_date(w[1]) for w in stats["windows"]],
            "level_mean": stats["level_mean"],
            "level_std": stats["level_std"],
            "slope_mean": stats["slope_mean"],
            "slope_std": stats["slope_std"],
            "curvature_mean": stats["curvature_mean"],
            "curvature_std": stats["curvature_std"],
        }
    )

    regimes = pl.DataFrame()
    try:
        matrix = factors.select(
            ["level_mean", "slope_mean", "curvature_mean"]
        ).to_numpy()
        classified = classify_yield_curve_regimes(matrix, n_regimes=n_regimes)
        names = list(classified.regime_names)
        labels = list(classified.regime_labels)
        regimes = pl.DataFrame(
            {
                "window_idx": factors.get_column("window_idx"),
                "date_end": factors.get_column("date_end"),
                "regime": [
                    names[int(labels[min(i, len(labels) - 1)])]
                    for i in range(factors.height)
                ],
            }
        )
    except Exception as exc:  # noqa: BLE001 -- regimes are decorative, not required
        if status is not None:
            status(f"Evolution: regime classification skipped ({exc})")

    return factors, regimes


def compute_stress_metrics(
    cube: np.ndarray,
    dates: list,
    *,
    window_size: int = 30,
    step_size: int = 10,
) -> pl.DataFrame:
    """Rolling stability of the residual correlation structure.

    Uses the mid-maturity slice of the residual cube, as the notebook does: a
    single representative tenor keeps the indicator one-dimensional and
    comparable across windows.
    """
    n_sources, n_dates, n_terms = cube.shape
    term_idx = n_terms // 2
    rows: list[dict] = []

    for w_start in range(0, n_dates - window_size + 1, step_size):
        w_end = w_start + window_size
        series = cube[:, w_start:w_end, term_idx]
        series = series[:, ~np.isnan(series).any(axis=0)]
        if series.shape[1] <= 1:
            continue
        corr = np.corrcoef(series)
        upper = np.abs(corr[np.triu_indices_from(corr, k=1)])
        upper = upper[np.isfinite(upper)]
        if upper.size == 0:
            continue
        rows.append(
            {
                "window_idx": len(rows),
                "date_end": as_date(dates[w_end - 1]),
                "avg_abs_corr": float(np.mean(upper)),
                "corr_variance": float(np.var(upper)),
                "n_sig_edges": int(np.sum(upper > 0.5)),
            }
        )

    stress = pl.DataFrame(rows)
    if stress.is_empty():
        return stress

    # 0 = calmest window seen, 100 = complete loss of the strongest correlation
    # level in the sample. Relative by construction, like the notebook's.
    peak = float(stress.get_column("avg_abs_corr").max() or 0.0)
    stress = stress.with_columns(
        ((1.0 - pl.col("avg_abs_corr") / peak) * 100.0 if peak else pl.lit(0.0)).alias(
            "stress_pct"
        )
    )
    return stress.with_columns(
        pl.when(pl.col("stress_pct") > STRESS_THRESHOLD)
        .then(pl.lit("Stressed (>50)"))
        .otherwise(pl.lit("Calm"))
        .alias("stress_band")
    )
