"""Compare Nelson-Siegel against the arbitrage-free models on one panel.

Fits each requested model, prints the residual, factor, network and agreement
tables, and optionally writes the comparison charts.

    uv run python scripts/compare_af_models.py \
        --db data/ycs_fake.duckdb --table par_rates --models ns,afns,dtafns

Yields are auto-detected as percent or decimals; pass ``--yield-scale`` to force
it. That matters more than it looks: the arbitrage-free correction scales with
the square of volatility, so a hundredfold unit error becomes ten thousandfold
in the correction.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import polars as pl

from ycn.analysis.af_compare import compare_residual_models, summarize
from ycn.analysis.af_models import Estimator, ModelSpec, ResidualModel, SigmaScope
from ycn.analysis.af_references import ReferenceStatus, format_bibliography
from ycn.analysis.config import PipelineConfig
from ycn.analysis.yield_curve import NetworkKind, detect_panel, load_long_panel


def _build_specs(args: argparse.Namespace) -> list[ModelSpec]:
    """Expand the --models/--sigma-scope/--decays flags into specs."""
    scopes = [SigmaScope(s) for s in args.sigma_scope.split(",") if s]
    decays = [float(d) for d in args.decays.split(",") if d]
    estimator = Estimator(args.estimator)

    specs: list[ModelSpec] = []
    for raw in args.models.split(","):
        if not raw:
            continue
        model = ResidualModel(raw.strip())
        # Plain NS has no volatility term, so sweeping the scope would produce
        # duplicate fits under different names.
        model_scopes = scopes if model.is_arbitrage_free else scopes[:1]
        for scope in model_scopes:
            for decay in decays:
                specs.append(
                    ModelSpec(
                        model=model,
                        decay=decay,
                        estimator=(
                            estimator if model.is_arbitrage_free else Estimator.TWO_STEP
                        ),
                        sigma_scope=scope,
                        n_refit=args.refits,
                        correlated_sigma=args.correlated_sigma,
                        yield_scale=args.yield_scale,
                    )
                )
    return specs


def _use_utf8_stdout() -> None:
    """Print box-drawing characters on a Windows console without dying.

    Polars renders tables with Unicode box characters, and the default Windows
    console codepage (cp1252) cannot encode them. Reconfiguring is preferable to
    stripping the characters, which would make every table harder to read
    everywhere else.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass  # already redirected or not reconfigurable; not worth failing over


def _show(title: str, frame: pl.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if frame.is_empty():
        print("(empty)")
        return
    with pl.Config(tbl_rows=60, tbl_cols=14, tbl_width_chars=200):
        print(frame)


def main() -> None:
    _use_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True, help="DuckDB or SQLite file.")
    parser.add_argument("--table", default="par_rates", help="Table to read.")
    parser.add_argument("--date-column", default="date", help="Date column.")
    parser.add_argument(
        "--models",
        default="ns,afns,dtafns",
        help="Comma-separated model names (ns, afns, dtafns, neural_hjm).",
    )
    parser.add_argument(
        "--sigma-scope",
        default="per_issuer",
        help="Comma-separated volatility scopes (per_issuer, pooled).",
    )
    parser.add_argument(
        "--decays",
        default="1.0",
        help="Comma-separated decays, in the repo's scale convention.",
    )
    parser.add_argument(
        "--estimator", default="two_step", choices=[e.value for e in Estimator]
    )
    parser.add_argument(
        "--refits", type=int, default=1, help="Volatility refit passes."
    )
    parser.add_argument(
        "--correlated-sigma",
        action="store_true",
        help="Fit a full lower-triangular Sigma.",
    )
    parser.add_argument(
        "--yield-scale",
        type=float,
        default=None,
        help="Multiplier taking yields to decimals (0.01 for percent). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--kind",
        default=NetworkKind.ISSUER_BY_TERM.value,
        choices=[k.value for k in NetworkKind],
    )
    parser.add_argument("--threshold", type=float, default=0.3, help="Edge threshold.")
    parser.add_argument("--acf-lags", type=int, default=5, help="Autocorrelation lags.")
    parser.add_argument("--date-start", default=None, help="ISO start date.")
    parser.add_argument("--date-end", default=None, help="ISO end date.")
    parser.add_argument(
        "--out", type=Path, default=None, help="Directory for charts and CSVs."
    )
    parser.add_argument(
        "--bibliography",
        action="store_true",
        help="Print the citation registry and exit.",
    )
    args = parser.parse_args()

    if args.bibliography:
        print(format_bibliography())
        return

    panel = detect_panel(args.db, args.table, args.date_column)
    if not panel.is_usable:
        raise SystemExit(
            f"{args.table!r} in {args.db} does not look like a curve panel "
            "(need an issuer column and at least two maturity columns)."
        )

    cfg = PipelineConfig(
        db_path=args.db,
        table=args.table,
        date_column=args.date_column,
        # These describe the *long* frame load_long_panel produces, not the
        # stored wide table.
        name_column=panel.term_column,
        value_column=panel.rate_column,
        date_start=date.fromisoformat(args.date_start) if args.date_start else None,
        date_end=date.fromisoformat(args.date_end) if args.date_end else None,
        issuer_column=panel.issuer_column,
        term_columns=list(panel.term_columns),
    )
    long = load_long_panel(cfg, panel)
    print(
        f"Loaded {long.height:,} rows: "
        f"{long.get_column(panel.issuer_column).n_unique()} issuers x "
        f"{long.get_column(panel.term_column).n_unique()} terms"
    )

    specs = _build_specs(args)
    print(
        f"Comparing {len(specs)} specification(s): {', '.join(s.name for s in specs)}"
    )

    comparison = compare_residual_models(
        long,
        panel,
        NetworkKind(args.kind),
        args.date_column,
        specs=specs,
        threshold=args.threshold,
        acf_lags=args.acf_lags,
        status=lambda msg: print(f"  {msg}"),
    )

    _show("Summary (sorted by long-end RMSE)", summarize(comparison))
    _show("Estimated parameters", comparison.factors)
    _show(
        "Residual diagnostics by maturity",
        comparison.residuals.select(
            "model", "maturity_years", "n_obs", "rmse", "mae", "acf_1", "ljung_box_p"
        ),
    )
    _show(
        "Cross-model agreement",
        comparison.agreement.filter(pl.col("label") == "__all__").select(
            "model_a", "model_b", "residual_corr", "mean_abs_diff"
        ),
    )
    _show(
        "Network edge agreement by layer",
        comparison.agreement.filter(pl.col("label") != "__all__").select(
            "model_a", "model_b", "label", "edge_jaccard"
        ),
    )

    if any(comparison.residuals.get_column("filtered_residuals").to_list()):
        print(
            "\nNOTE: at least one model reports *filtered* in-sample residuals, which "
            "are not the same statistical object as the cross-sectional residuals of "
            "the others. Compare its RMSE column with that in mind."
        )

    if args.out is not None:
        _write_outputs(comparison, args.out)

    print("\n" + format_bibliography(ReferenceStatus.IMPLEMENTED))


def _write_outputs(comparison, out: Path) -> None:
    """Save the frames as CSV and the charts as PNG."""
    from ycn.analysis import af_compare_viz as viz

    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("summary", summarize(comparison)),
        ("residuals", comparison.residuals),
        ("factors", comparison.factors),
        ("networks", comparison.networks),
        ("agreement", comparison.agreement),
    ):
        if not frame.is_empty():
            frame.write_csv(out / f"{name}.csv")

    charts = {
        "rmse_by_maturity": viz.plot_rmse_by_maturity,
        "residual_acf": viz.plot_residual_acf,
        "acf_heatmap": viz.plot_acf_heatmap,
        "network_modularity": viz.plot_network_metric,
        "edge_agreement": viz.plot_edge_agreement,
        "yield_adjustment": viz.plot_yield_adjustment,
    }
    for name, builder in charts.items():
        try:
            builder(comparison).save(
                out / f"{name}.png", width=9, height=5, dpi=140, verbose=False
            )
        except Exception as exc:  # noqa: BLE001 -- one bad chart must not lose the rest
            print(f"  chart {name} skipped ({exc})")

    print(f"\nWrote frames and charts to {out}")


if __name__ == "__main__":
    main()
