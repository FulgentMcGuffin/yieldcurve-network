"""Nelson-Siegel residual networks, one per component-network label.

A yield-curve panel is dominated by its level/slope/curvature factors: raw rates
across issuers or maturities co-move so strongly that a correlation network is
almost complete and says little. Stripping the fitted Nelson-Siegel curve from
each issuer first leaves the *idiosyncratic* part, and a network built on those
residuals shows who actually deviates together.

The residual cube is ``(issuer, date, term)``, so both component-network
directions fall out of the same object:

* ``ISSUER_BY_TERM`` -- slice a term, correlate across issuers -> issuer network
* ``TERM_BY_ISSUER`` -- slice an issuer, correlate across terms  -> term network

Everything here is pure numpy/polars/networkx so it can run on a worker thread.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import networkx as nx
import numpy as np
import polars as pl

from .af_fit import pooled_sigma
from .af_models import ModelSpec, PanelFit, SigmaScope, detect_yield_scale, fit_panel
from .cancellation import ComputationCancelled
from .network import pivot_to_wide
from .yield_curve import CurvePanel, NetworkKind, parse_term_years, sort_terms
from .yield_curve_factors import extract_ns_residuals

StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int, str], None]

# Nelson-Siegel has three factors plus a decay, so a curve with fewer points
# than this cannot be fitted meaningfully.
MIN_TERMS_FOR_NS = 4

# An integer column only qualifies as a shape variable below this many distinct
# values -- a shape scale with dozens of levels is unreadable, and matplotlib
# runs out of distinguishable markers well before that.
MAX_SHAPE_LEVELS = 10

# Dates per Nelson-Siegel fitting block. ``extract_ns_residuals`` fits each date
# independently, so slicing the frame and concatenating is exact -- but it turns
# one opaque multi-second call into a sequence of short ones, which is what lets
# the worker check for cancellation and hand the GIL back to the GUI thread
# often enough to keep the window responsive.
NS_CHUNK_DATES = 200


@dataclass
class ResidualNetworkResult:
    """Artifacts from the residual-network pass."""

    metrics: pl.DataFrame  # one row per label; the chart's source frame
    coverage: pl.DataFrame  # (issuer, min_date, max_date, coverage_days, n_obs)
    graphs: dict[str, nx.Graph] = field(default_factory=dict)
    label_column: str = "label"
    label_order: list[str] = field(default_factory=list)
    node_label: str = "node"
    skipped: list[str] = field(default_factory=list)


def coverage_frame(
    long: pl.DataFrame, panel: CurvePanel, date_column: str
) -> pl.DataFrame:
    """Per-issuer first/last observation and span, for the Coverage chart."""
    if long.is_empty():
        return pl.DataFrame(
            schema={
                panel.issuer_column: pl.Utf8,
                "min_date": pl.Date,
                "max_date": pl.Date,
                "coverage_days": pl.Int64,
                "n_obs": pl.UInt32,
            }
        )
    return (
        long.group_by(panel.issuer_column)
        .agg(
            pl.col(date_column).min().alias("min_date"),
            pl.col(date_column).max().alias("max_date"),
            pl.len().alias("n_obs"),
        )
        .with_columns(
            (pl.col("max_date") - pl.col("min_date"))
            .dt.total_days()
            .alias("coverage_days")
        )
        .sort("coverage_days", descending=True)
    )


def _canonical_maturities(terms: list[str]) -> dict[str, str]:
    """Map term labels to a ``<years>Y`` form ``extract_ns_residuals`` can read.

    That function derives maturities by string surgery on the column name
    (``strip "Y"``, ``"p" -> "."``), which handles ``10Y`` and ``Y000p5`` but
    raises on ``6M``/``4W``/``30D`` -- labels ``detect_panel`` accepts happily.
    Renaming first means the NS fit works for every spelling this project
    recognises, instead of failing every issuer with a misleading error.
    """
    canonical: dict[str, str] = {}
    for term in terms:
        years = parse_term_years(term)
        canonical[term] = f"{years:g}Y" if years is not None else term
    return canonical


def _fit_residuals_chunked(
    wide: pl.DataFrame,
    date_column: str,
    terms: list[str],
    decay: float | None,
    on_chunk: Callable[[], None] | None,
) -> tuple[np.ndarray, list]:
    """``extract_ns_residuals`` over one issuer, in date blocks.

    Identical output to a single call -- the fit is per-date and carries no
    state across rows -- but ``on_chunk`` runs between blocks, giving the caller
    a cancellation point and the interpreter a GIL handover inside what is
    otherwise the longest uninterrupted stretch of the whole run.
    """
    canonical = _canonical_maturities(terms)
    wide = wide.rename(canonical)
    fit_terms = [canonical[t] for t in terms]

    matrices: list[np.ndarray] = []
    dates: list = []
    for offset in range(0, wide.height, NS_CHUNK_DATES):
        block = wide.slice(offset, NS_CHUNK_DATES)
        matrix, meta = extract_ns_residuals(
            block, date_col=date_column, term_cols=fit_terms, decay=decay
        )
        if matrix.size:
            matrices.append(matrix)
            dates.extend(meta["dates"])
        if on_chunk is not None:
            on_chunk()
    if not matrices:
        return np.empty((0, len(terms)), dtype=float), []
    return np.vstack(matrices), dates


def _canonical_maturity_values(canonical_terms: list[str]) -> np.ndarray:
    """Maturities in years, derived exactly as ``extract_ns_residuals`` does.

    Deliberately repeats that function's string surgery rather than calling
    ``parse_term_years``, because ``_canonical_maturities`` formats labels with
    ``%g`` and so rounds to six significant digits. Re-parsing the label is what
    the legacy path sees, and any model compared against it must see the same
    numbers -- for ``4W`` that is 0.0769231, not 0.07692307692307693.
    """
    return np.array(
        [float(term.replace("Y", "").replace("p", ".")) for term in canonical_terms],
        dtype=float,
    )


def _dense_rows(
    wide: pl.DataFrame, date_column: str, terms: list[str]
) -> tuple[list, np.ndarray]:
    """Dates with a complete curve, and the matching yield matrix.

    Applies exactly the row filter ``extract_ns_residuals`` applies -- drop any
    date with a NaN at any requested tenor -- factored out so the legacy NS path
    and the model path drop precisely the same dates and their residual cubes
    stay comparable cell for cell.

    Args:
        wide: One issuer's curves, dates as rows.
        date_column: Date column name.
        terms: Term columns to read, in maturity order.

    Returns:
        ``(dates, yields)`` with ``yields`` of shape ``(n_kept, n_terms)``.
    """
    dates: list = []
    rows: list[np.ndarray] = []
    for row in wide.iter_rows(named=True):
        values = np.array([row[term] for term in terms], dtype=float)
        if np.any(np.isnan(values)):
            continue
        dates.append(row[date_column])
        rows.append(values)
    if not rows:
        return [], np.empty((0, len(terms)), dtype=float)
    return dates, np.vstack(rows)


def _fit_residuals_model(
    wide: pl.DataFrame,
    date_column: str,
    terms: list[str],
    spec: ModelSpec,
    on_chunk: Callable[[], None] | None,
    **kwargs: object,
) -> tuple[np.ndarray, list, PanelFit]:
    """Fit one issuer's whole panel with the model named by ``spec``.

    Unlike the per-date NS path this is a panel fit: the two-step estimator needs
    the full date series to estimate the volatility matrix that drives the
    yield adjustment. One consequence is that a failure here loses the issuer
    rather than a single date -- which :func:`residual_cube` already handles by
    routing it to ``skipped``.
    """
    canonical = _canonical_maturities(terms)
    wide = wide.rename(canonical)
    fit_terms = [canonical[t] for t in terms]
    maturities = _canonical_maturity_values(fit_terms)

    dates, yields = _dense_rows(wide, date_column, fit_terms)
    if not dates:
        empty = np.empty((0, len(terms)), dtype=float)
        return (
            empty,
            [],
            PanelFit(
                residuals=empty,
                factors=np.empty((0, 3), dtype=float),
                decay=np.empty(0, dtype=float),
                rmse=np.empty(0, dtype=float),
                adjustment=np.zeros(len(terms), dtype=float),
                sigma=np.zeros((3, 3), dtype=float),
            ),
        )

    fit = fit_panel(maturities, yields, spec, on_chunk=on_chunk, **kwargs)
    # Which tenors this issuer actually had. Panels are ragged, so a consumer
    # cannot assume the adjustment lines up with the cube's union term axis.
    fit.diagnostics["terms"] = list(terms)
    fit.diagnostics["maturities"] = maturities.tolist()
    return fit.residuals, dates, fit


def residual_cube(
    long: pl.DataFrame,
    panel: CurvePanel,
    date_column: str,
    *,
    decay: float | None = 1.0,
    model: ModelSpec | None = None,
    fits_out: dict[str, PanelFit] | None = None,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> tuple[list[str], list, np.ndarray, list[str], list[str]]:
    """Fit a curve model per issuer and align residuals onto shared dates.

    Returns ``(issuers, dates, cube, terms, skipped)`` where ``cube`` has shape
    ``(n_issuers, n_dates, n_terms)``. Issuers whose curve cannot be fitted are
    dropped rather than aborting the run; they come back in ``skipped``.

    Both the date and term axes are the **union** across issuers, with gaps
    left as NaN. Curve panels are ragged -- issuers quote different tenors and
    start on different days -- and intersecting the axes here would let one
    late-starting or short-curve issuer truncate every network. Dropping is
    instead done per layer in :func:`build_residual_network`, where it costs
    only the layers that actually lack the data.

    Args:
        long: Long ``(date, issuer, term, rate)`` frame.
        panel: Column-role assignment.
        date_column: Date column name.
        decay: Nelson-Siegel decay for the default path. Ignored when ``model``
            is given, which carries its own ``decay``.
        model: Which curve model to fit. ``None`` -- the default -- runs the
            plain per-date Nelson-Siegel path unchanged, so every existing
            caller and every published metric keeps its exact behaviour.
        fits_out: If given, populated with the ``PanelFit`` per issuer. Only the
            model path produces these; the legacy path leaves it empty.
        progress: Per-issuer progress callback.
        status: Human-readable status callback.
    """
    terms = sort_terms(long.get_column(panel.term_column).unique().to_list())
    if len(terms) < MIN_TERMS_FOR_NS:
        raise ValueError(
            f"Nelson-Siegel needs at least {MIN_TERMS_FOR_NS} maturities; "
            f"the current selection has {len(terms)}."
        )

    if model is not None and model.yield_scale is None:
        # Resolve the percent-versus-decimals question once, from the whole
        # panel. Left to each issuer's own fit, a partly-negative curve is
        # classified as decimals while its percent-quoted neighbours are not,
        # and because the adjustment scales with the square of volatility the
        # two then differ by a factor of 10,000.
        model = replace(
            model,
            yield_scale=detect_yield_scale(
                long.get_column(panel.rate_column).to_numpy()
            ),
        )
        if status is not None:
            unit = "percent" if model.yield_scale != 1.0 else "decimals"
            status(f"NS: yields detected as {unit}")

    issuers = sorted(long.get_column(panel.issuer_column).unique().to_list())
    residuals: dict[str, np.ndarray] = {}
    used_terms: dict[str, list[str]] = {}
    used_dates: dict[str, list] = {}
    fits: dict[str, PanelFit] = {}
    skipped: list[str] = []

    def sweep(**extra: object) -> None:
        """Fit every issuer once, recording results in the enclosing scope.

        Run twice under the pooled volatility scope: the first pass supplies the
        factor series the pooled Sigma is estimated from, the second refits every
        issuer against the resulting common offset.
        """
        residuals.clear()
        used_terms.clear()
        used_dates.clear()
        fits.clear()
        skipped.clear()

        total = max(len(issuers), 1)
        for index, issuer in enumerate(issuers):
            sub = long.filter(pl.col(panel.issuer_column) == issuer)
            tick = (
                (lambda i=index, s=issuer: progress(i, total, f"NS fit: {s}"))
                if progress is not None
                else None
            )
            try:
                wide = pivot_to_wide(
                    sub,
                    date_column=date_column,
                    name_column=panel.term_column,
                    value_column=panel.rate_column,
                )
                present = [t for t in terms if t in wide.columns]
                if len(present) < MIN_TERMS_FOR_NS:
                    skipped.append(issuer)
                    continue
                if model is None:
                    matrix, fitted_dates = _fit_residuals_chunked(
                        wide, date_column, present, decay, tick
                    )
                else:
                    matrix, fitted_dates, fit = _fit_residuals_model(
                        wide, date_column, present, model, tick, **extra
                    )
                    if fitted_dates:
                        fits[issuer] = fit
                if not fitted_dates:
                    skipped.append(issuer)
                    continue
            except ComputationCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 -- one bad issuer must not abort
                if status is not None:
                    status(f"NS: skipping {issuer} ({exc})")
                skipped.append(issuer)
                continue
            residuals[issuer] = np.asarray(matrix, dtype=float)
            used_terms[issuer] = present
            used_dates[issuer] = fitted_dates
            if progress is not None:
                progress(index + 1, total, f"NS fit: {issuer}")

    sweep()

    if (
        model is not None
        and model.sigma_scope is SigmaScope.POOLED
        and model.model.is_arbitrage_free
        and fits
    ):
        if status is not None:
            status("NS: pooling volatility across issuers…")
        shared = pooled_sigma(fits, model.dt, model.correlated_sigma)
        sweep(sigma_override=shared)

    if fits_out is not None:
        fits_out.update(fits)

    if len(residuals) < 2:
        raise ValueError(
            "Fewer than two issuers produced Nelson-Siegel residuals, so no "
            "residual network can be built."
        )

    kept = list(residuals)
    all_dates = sorted(set().union(*(set(used_dates[s]) for s in kept)))
    if len(all_dates) < 3:
        raise ValueError(
            f"Only {len(all_dates)} dates carry residuals; at least 3 are "
            "needed. Widen the date range."
        )

    term_index = {term: i for i, term in enumerate(terms)}
    date_index = {day: j for j, day in enumerate(all_dates)}
    cube = np.full((len(kept), len(all_dates), len(terms)), np.nan, dtype=float)
    for i, issuer in enumerate(kept):
        cols = [term_index[t] for t in used_terms[issuer]]
        rows = [date_index[d] for d in used_dates[issuer]]
        # np.ix_ scatters the issuer's (dates x its terms) block into its slots
        # on the union axes in one shot.
        cube[i][np.ix_(rows, cols)] = residuals[issuer]

    return kept, all_dates, cube, terms, skipped


def build_residual_network(
    series: np.ndarray,
    names: list[str],
    threshold: float,
) -> nx.Graph:
    """Correlation network over rows of ``series`` (one row per node).

    Edges are kept on ``|corr| > threshold``: an anti-correlated residual pair
    is as much a relationship as a correlated one, unlike the raw-rate networks
    where sign carries direction of co-movement.

    Nodes with no data at all in this slice are dropped before the date filter
    -- otherwise one issuer that does not quote this tenor would make every
    date incomplete and empty the layer.
    """
    arr = np.asarray(series, dtype=float)
    present = ~np.isnan(arr).all(axis=1)
    arr = arr[present]
    kept = [n for n, keep in zip(names, present) if keep]

    graph = nx.Graph()
    graph.add_nodes_from(kept)
    if arr.shape[0] < 2:
        return graph

    arr = arr[:, ~np.isnan(arr).any(axis=0)]
    if arr.shape[1] < 3:
        return graph

    corr = np.nan_to_num(np.corrcoef(arr), nan=0.0)
    corr = np.atleast_2d(corr)
    for i, left in enumerate(kept):
        for j in range(i + 1, len(kept)):
            value = float(corr[i, j])
            if abs(value) > threshold:
                graph.add_edge(left, kept[j], weight=value)
    return graph


def summarize_residual_network(graph: nx.Graph, label: str) -> dict[str, object]:
    """Network-level metrics for one residual graph (one chart row)."""
    nan = float("nan")
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    row: dict[str, object] = {
        "label": label,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "is_connected": False,
        "n_components": 0,
        "largest_component_size": 0,
        "diameter": nan,
        "radius": nan,
        "avg_clustering": nan,
        "transitivity": nan,
        "n_triangles": 0,
        "modularity": nan,
        "avg_shortest_path_length": nan,
        "shortest_path_length": nan,
        "avg_eccentricity": nan,
        "density": nan,
        "edge_density": nan,
        "avg_degree": nan,
        "degree_assortativity": nan,
        "avg_neighbor_degree": nan,
        "n_cliques": 0,
        "spectral_radius": nan,
    }
    if n_nodes == 0:
        return row

    components = list(nx.connected_components(graph))
    largest = max(components, key=len)
    giant = graph.subgraph(largest).copy()
    row["n_components"] = len(components)
    row["largest_component_size"] = len(largest)
    row["is_connected"] = len(components) == 1

    if len(largest) >= 2 and nx.is_connected(giant):
        row["diameter"] = float(nx.diameter(giant))
        row["radius"] = float(nx.radius(giant))
        row["avg_shortest_path_length"] = float(nx.average_shortest_path_length(giant))
        row["avg_eccentricity"] = float(np.mean(list(nx.eccentricity(giant).values())))
        row["shortest_path_length"] = float(
            min(
                nx.shortest_path_length(giant, u, v)
                for u in giant
                for v in giant
                if u != v
            )
        )

    row["avg_clustering"] = float(nx.average_clustering(graph))
    transitivity = nx.transitivity(graph)
    row["transitivity"] = float(transitivity) if transitivity else nan
    row["n_triangles"] = int(sum(nx.triangles(graph).values()) // 3)

    try:
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
        row["modularity"] = (
            float(nx.algorithms.community.modularity(graph, communities))
            if communities
            else nan
        )
    except Exception:  # noqa: BLE001 -- modularity is undefined on some graphs
        row["modularity"] = nan

    row["density"] = float(nx.density(graph))
    max_edges = n_nodes * (n_nodes - 1) / 2
    row["edge_density"] = float(n_edges / max_edges) if max_edges else nan
    row["avg_degree"] = float(2 * n_edges / n_nodes)
    row["degree_assortativity"] = (
        float(nx.degree_assortativity_coefficient(graph)) if n_edges else nan
    )
    row["avg_neighbor_degree"] = float(
        np.mean(list(nx.average_neighbor_degree(graph).values()))
    )
    row["n_cliques"] = len(list(nx.find_cliques(graph)))

    adjacency = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()), weight=None)
    row["spectral_radius"] = float(max(abs(np.linalg.eigvals(adjacency)).real))
    return row


def compute_residual_networks(
    long: pl.DataFrame,
    panel: CurvePanel,
    kind: NetworkKind,
    date_column: str,
    *,
    threshold: float = 0.3,
    decay: float | None = 1.0,
    model: ModelSpec | None = None,
    fits_out: dict[str, PanelFit] | None = None,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> ResidualNetworkResult:
    """One residual network per component-network label, plus its metrics.

    For ``ISSUER_BY_TERM`` there is one network per maturity whose nodes are
    issuers; for ``TERM_BY_ISSUER`` one network per issuer whose nodes are
    maturities. Both read the same ``(issuer, date, term)`` residual cube.

    See :func:`residual_cube` for ``decay``, ``model`` and ``fits_out``.
    """
    if status is not None:
        status("NS: fitting Nelson-Siegel residuals per issuer…")
    issuers, dates, cube, terms, skipped = residual_cube(
        long,
        panel,
        date_column,
        decay=decay,
        model=model,
        fits_out=fits_out,
        progress=progress,
        status=status,
    )
    if status is not None:
        status(f"NS: {len(issuers)} issuers x {len(dates)} dates x {len(terms)} terms")

    return _networks_from_cube(
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


def _networks_from_cube(
    issuers: list[str],
    dates: list,
    cube: np.ndarray,
    terms: list[str],
    skipped: list[str],
    *,
    long: pl.DataFrame,
    panel: CurvePanel,
    kind: NetworkKind,
    date_column: str,
    threshold: float,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> ResidualNetworkResult:
    """Build every layer's network from an already-fitted residual cube.

    Split out of :func:`compute_residual_networks` so a caller comparing several
    models can fit each one once and reuse its cube, instead of paying for the
    fit again per model.
    """
    if kind is NetworkKind.ISSUER_BY_TERM:
        labels, nodes, label_column, node_label = (
            terms,
            issuers,
            panel.term_column,
            panel.issuer_column,
        )

        def slice_for(idx: int) -> np.ndarray:
            return cube[:, :, idx]

    else:
        labels, nodes, label_column, node_label = (
            issuers,
            terms,
            panel.issuer_column,
            panel.term_column,
        )

        def slice_for(idx: int) -> np.ndarray:
            # (dates, terms) -> (terms, dates): nodes must be the row axis.
            return cube[idx, :, :].T

    graphs: dict[str, nx.Graph] = {}
    rows: list[dict[str, object]] = []
    total = max(len(labels), 1)
    for idx, label in enumerate(labels):
        if status is not None:
            status(f"NS: residual network {idx + 1}/{len(labels)} ({label})")
        graph = build_residual_network(slice_for(idx), nodes, threshold)
        graphs[label] = graph
        rows.append(summarize_residual_network(graph, label))
        if progress is not None:
            progress(idx + 1, total, f"{label}")

    metrics = pl.DataFrame(rows) if rows else pl.DataFrame()
    return ResidualNetworkResult(
        metrics=metrics,
        coverage=coverage_frame(long, panel, date_column),
        graphs=graphs,
        label_column=label_column,
        label_order=list(labels),
        node_label=node_label,
        skipped=skipped,
    )


def aesthetic_columns(metrics: pl.DataFrame) -> dict[str, list[str]]:
    """Which metric columns may drive each aesthetic of the chart.

    ``numeric`` drives y / fill / size. ``discrete`` drives shape: booleans and
    strings always qualify, integers only when low-cardinality -- a shape scale
    with dozens of levels is unreadable and matplotlib runs out of markers.
    """
    numeric: list[str] = []
    discrete: list[str] = []
    if metrics.is_empty():
        return {"numeric": numeric, "discrete": discrete}

    for name, dtype in metrics.schema.items():
        if name == "label":
            continue
        if dtype == pl.Boolean or dtype == pl.Utf8:
            discrete.append(name)
        elif dtype.is_integer():
            numeric.append(name)
            if metrics.get_column(name).n_unique() <= MAX_SHAPE_LEVELS:
                discrete.append(name)
        elif dtype.is_float():
            numeric.append(name)
    return {"numeric": numeric, "discrete": discrete}
