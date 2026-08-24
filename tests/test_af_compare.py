"""Tests for the model comparison harness."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ycn.analysis.af_compare import compare_residual_models, summarize
from ycn.analysis.af_models import ModelSpec, ResidualModel, SigmaScope
from ycn.analysis.residual_networks import summarize_residual_network
from ycn.analysis.yield_curve import NetworkKind

SPECS = (
    ModelSpec(ResidualModel.NS),
    ModelSpec(ResidualModel.AFNS, n_refit=1),
    ModelSpec(ResidualModel.DTAFNS, n_refit=1),
)


@pytest.fixture(scope="module")
def comparison_inputs():
    """A simulated AFNS panel and its CurvePanel, shared across the module."""
    from ycn.analysis.af_simulate import TERM_LABELS, simulate_afns_panel
    from ycn.analysis.yield_curve import CurvePanel

    frame, truth = simulate_afns_panel(n_dates=250, seed=71)
    panel = CurvePanel(
        date_column="date",
        issuer_column="source",
        term_columns=TERM_LABELS,
        term_column="term",
        rate_column="rate",
    )
    return frame, panel, truth


@pytest.fixture(scope="module")
def comparison(comparison_inputs):
    frame, panel, _ = comparison_inputs
    return compare_residual_models(
        frame,
        panel,
        NetworkKind.ISSUER_BY_TERM,
        "date",
        specs=SPECS,
    )


def test_every_model_appears_in_every_frame(comparison):
    """Each spec contributes rows to residuals, factors and networks."""
    expected = {spec.name for spec in SPECS}
    assert set(comparison.residuals.get_column("model").unique()) == expected
    assert set(comparison.factors.get_column("model").unique()) == expected
    assert set(comparison.networks.get_column("model").unique()) == expected
    assert set(comparison.cubes) == expected


def test_residual_frame_has_one_row_per_model_and_term(comparison):
    """Residual diagnostics are per maturity, per model."""
    n_terms = len(comparison.terms)
    assert comparison.residuals.height == len(SPECS) * n_terms
    assert comparison.residuals.get_column("rmse").is_finite().all()
    assert comparison.residuals.get_column("maturity_years").is_not_null().all()


def test_acf_columns_are_present_and_bounded(comparison):
    """Autocorrelations are reported for every requested lag and stay in range."""
    for lag in range(1, 6):
        column = comparison.residuals.get_column(f"acf_{lag}")
        finite = column.drop_nans().drop_nulls()
        assert finite.len() > 0
        assert finite.abs().max() <= 1.0 + 1e-9


def test_network_frame_carries_every_metric(comparison):
    """The model column is added without dropping any network metric."""
    import networkx as nx

    expected = set(summarize_residual_network(nx.Graph(), "x")) | {"model"}
    assert expected <= set(comparison.networks.columns)


def test_agreement_covers_every_pair(comparison):
    """One overall row plus one per label, for each unordered model pair."""
    pairs = comparison.agreement.select("model_a", "model_b").unique()
    n_models = len(SPECS)
    assert pairs.height == n_models * (n_models - 1) // 2

    overall = comparison.agreement.filter(pl.col("label") == "__all__")
    assert overall.height == pairs.height
    assert overall.get_column("residual_corr").is_finite().all()


def test_edge_jaccard_is_a_proportion(comparison):
    """Jaccard similarity stays within [0, 1] wherever it is reported."""
    jaccard = comparison.agreement.get_column("edge_jaccard").drop_nans().drop_nulls()
    assert jaccard.len() > 0
    assert jaccard.min() >= 0.0
    assert jaccard.max() <= 1.0


def test_models_are_not_identical(comparison):
    """AFNS must actually differ from NS, or the harness has nothing to show."""
    ns = comparison.cubes[ModelSpec(ResidualModel.NS).name]
    afns = comparison.cubes[ModelSpec(ResidualModel.AFNS, n_refit=1).name]
    assert not np.allclose(ns, afns, equal_nan=True)


def test_afns_wins_at_the_long_end_on_afns_data(comparison):
    """On data generated with convexity, the arbitrage-free models fit it better."""
    long_end = comparison.residuals.filter(
        pl.col("model_kind").is_in(["ns", "afns"]) & (pl.col("maturity_years") >= 15.0)
    )

    by_model = long_end.group_by("model_kind").agg(pl.col("rmse").mean().alias("rmse"))
    lookup = dict(zip(by_model.get_column("model_kind"), by_model.get_column("rmse")))
    assert lookup["afns"] < lookup["ns"]


def test_summarize_ranks_by_long_end_rmse(comparison):
    """The headline table is one row per model, best long-end fit first."""
    summary = summarize(comparison)
    assert summary.height == len(SPECS)
    values = summary.get_column("rmse_long").to_list()
    assert values == sorted(values)
    assert "mean_abs_acf1" in summary.columns


def test_duplicate_spec_names_are_rejected(comparison_inputs):
    """Two identical specs would overwrite each other's results silently."""
    frame, panel, _ = comparison_inputs
    with pytest.raises(ValueError, match="unique"):
        compare_residual_models(
            frame,
            panel,
            NetworkKind.ISSUER_BY_TERM,
            "date",
            specs=[ModelSpec(ResidualModel.NS), ModelSpec(ResidualModel.NS)],
        )


def test_empty_specs_are_rejected(comparison_inputs):
    frame, panel, _ = comparison_inputs
    with pytest.raises(ValueError, match="at least one"):
        compare_residual_models(
            frame, panel, NetworkKind.ISSUER_BY_TERM, "date", specs=[]
        )


def test_sigma_scopes_stay_distinguishable(comparison_inputs):
    """Per-issuer and pooled produce separate, differently named rows."""
    frame, panel, _ = comparison_inputs
    specs = [
        ModelSpec(ResidualModel.AFNS, n_refit=1, sigma_scope=SigmaScope.PER_ISSUER),
        ModelSpec(ResidualModel.AFNS, n_refit=1, sigma_scope=SigmaScope.POOLED),
    ]
    result = compare_residual_models(
        frame, panel, NetworkKind.ISSUER_BY_TERM, "date", specs=specs
    )

    names = [spec.name for spec in specs]
    assert len(set(names)) == 2
    assert set(result.cubes) == set(names)
    assert not np.allclose(
        result.cubes[names[0]], result.cubes[names[1]], equal_nan=True
    )


def test_graphs_property_is_keyed_by_model_and_label(comparison):
    """Graphs are addressable per (model, label) for downstream plotting."""
    graphs = comparison.graphs
    assert graphs
    for (model, label), graph in graphs.items():
        assert model in comparison.cubes
        assert label in comparison.terms
        assert graph.number_of_nodes() >= 0


def test_time_invariant_adjustment_shifts_residuals_by_a_constant(comparison):
    """The arbitrage-free correction moves residual levels, not their dynamics.

    Because the volatility matrix is fitted once over the whole sample, the
    adjustment carries no time index, so each issuer-tenor residual series is
    shifted by a constant. Asserted to machine precision because it is exact
    algebra, not an approximation.
    """
    ns = comparison.cubes[ModelSpec(ResidualModel.NS).name]
    afns = comparison.cubes[ModelSpec(ResidualModel.AFNS, n_refit=1).name]

    difference = afns - ns
    assert np.nanmean(np.abs(difference)) > 1e-6, "models are indistinguishable"
    # Zero variation through time for every (issuer, tenor) pair.
    assert float(np.nanmax(np.nanstd(difference, axis=1))) < 1e-15


def test_time_invariant_adjustment_leaves_networks_identical(comparison):
    """The consequence: these networks cannot distinguish the models.

    Correlation is invariant under adding a constant, so every edge survives
    unchanged. This is the honest headline of the comparison -- the fit improves
    markedly while the network topology is provably untouched. A model with a
    time-varying adjustment would break this, and should.
    """
    ns_result = comparison.results[ModelSpec(ResidualModel.NS).name]
    afns_result = comparison.results[ModelSpec(ResidualModel.AFNS, n_refit=1).name]

    total_edges = 0
    for label in ns_result.label_order:
        ns_edges = {frozenset(e) for e in ns_result.graphs[label].edges()}
        afns_edges = {frozenset(e) for e in afns_result.graphs[label].edges()}
        assert ns_edges == afns_edges, f"layer {label} differs"
        total_edges += len(ns_edges)

    # Guard against the vacuous version of this test: empty graphs are trivially
    # equal, and would prove nothing.
    assert total_edges > 0, "networks are empty, so equality is meaningless"


def test_simulated_panel_yields_modular_networks(comparison):
    """The fixture must produce networks worth comparing.

    Neither empty (no shared structure) nor complete (shared structure swamping
    everything) -- both extremes make every model look identical for reasons
    that have nothing to do with the models.
    """
    metrics = comparison.networks
    edges = metrics.get_column("n_edges")
    nodes = metrics.get_column("n_nodes").max()
    complete = nodes * (nodes - 1) // 2

    assert edges.min() > 0, "networks are empty"
    assert edges.max() < complete, "networks are complete, so they say nothing"
    assert metrics.get_column("n_components").max() > 1, "no community structure"


def test_ragged_panel_does_not_break_the_harness(ragged_panel_long, curve_panel):
    """A panel with skipped and late-starting issuers still compares cleanly."""
    result = compare_residual_models(
        ragged_panel_long,
        curve_panel,
        NetworkKind.ISSUER_BY_TERM,
        "date",
        specs=[ModelSpec(ResidualModel.NS), ModelSpec(ResidualModel.AFNS, n_refit=1)],
    )
    assert result.residuals.height > 0
    assert result.factors.get_column("n_skipped").max() >= 1


def test_fits_record_the_tenors_they_used(ragged_panel_long, curve_panel):
    """Each fit knows its own tenors, because panels are ragged.

    An issuer missing the long end produces a shorter adjustment vector than the
    cube's union term axis. A consumer that assumes otherwise either raises or,
    worse, silently lines one issuer's long end up against another's short end.
    """
    result = compare_residual_models(
        ragged_panel_long,
        curve_panel,
        NetworkKind.ISSUER_BY_TERM,
        "date",
        specs=[ModelSpec(ResidualModel.AFNS, n_refit=1)],
    )

    fits = result.fits[ModelSpec(ResidualModel.AFNS, n_refit=1).name]
    lengths = set()
    for fit in fits.values():
        terms = fit.diagnostics["terms"]
        assert len(terms) == len(fit.adjustment)
        lengths.add(len(terms))

    # The ragged fixture drops two tenors for one issuer, so the lengths differ.
    assert len(lengths) > 1, "fixture is not actually ragged"


def test_adjustment_chart_handles_ragged_tenors(ragged_panel_long, curve_panel):
    """The adjustment chart builds when issuers carry different tenor counts."""
    from ycn.analysis.af_compare_viz import plot_yield_adjustment

    result = compare_residual_models(
        ragged_panel_long,
        curve_panel,
        NetworkKind.ISSUER_BY_TERM,
        "date",
        specs=[ModelSpec(ResidualModel.AFNS, n_refit=1)],
    )

    plot = plot_yield_adjustment(result)
    assert plot.data.height > 0
    assert set(plot.data.columns) >= {"model", "maturity_years", "adjustment_bp"}
    # Every tenor present on at least one issuer should be represented.
    assert plot.data.get_column("maturity_years").n_unique() == len(result.terms)
