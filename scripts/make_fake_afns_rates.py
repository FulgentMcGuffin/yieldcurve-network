"""Generate a fake ``par_rates`` panel from a known arbitrage-free model.

Companion to ``make_fake_par_rates.py``. That script generates curves from a
plain Nelson-Siegel process; this one generates them from the AFNS measurement
equation *including* the yield adjustment, so the panel carries genuine
convexity that a plain Nelson-Siegel fit cannot represent.

Useful for eyeballing the two frameworks side by side on data whose true
parameters are known:

    uv run python scripts/make_fake_afns_rates.py --out data/ycs_afns_fake.duckdb
    uv run python scripts/compare_af_models.py --db data/ycs_afns_fake.duckdb \
        --table par_rates --models ns,afns

The simulation itself lives in :mod:`ycn.analysis.af_simulate` so the tests can
import it without going through a file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import polars as pl

from ycn.analysis.af_simulate import DEFAULT_SIGMA, TERM_LABELS, simulate_afns_panel


def _to_wide(long: pl.DataFrame) -> pl.DataFrame:
    """Long ``(date, source, term, rate)`` to the wide schema ``detect_panel`` reads."""
    wide = long.pivot(on="term", index=["date", "source"], values="rate").sort(
        "date", "source"
    )
    ordered = [c for c in TERM_LABELS if c in wide.columns]
    return wide.select(["date", "source", *ordered])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/ycs_afns_fake.duckdb"),
        help="Destination DuckDB file.",
    )
    parser.add_argument(
        "--days", type=int, default=750, help="Business days to simulate."
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--decay",
        type=float,
        default=1.0,
        help="Decay in the repo's scale convention (loadings use exp(-tau/decay)).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        nargs=3,
        default=list(DEFAULT_SIGMA),
        metavar=("LEVEL", "SLOPE", "CURVE"),
        help="Annual factor volatilities, in decimals.",
    )
    args = parser.parse_args()

    long, truth = simulate_afns_panel(
        n_dates=args.days,
        decay=args.decay,
        sigma=tuple(args.sigma),
        seed=args.seed,
    )
    wide = _to_wide(long)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    panel = (
        wide.to_arrow()
    )  # noqa: F841 -- duckdb resolves `panel` from the local scope
    with duckdb.connect(str(args.out)) as con:
        con.execute("CREATE TABLE par_rates AS SELECT * FROM panel")
        # A second table so the GUI's "default to par_rates" pick is observable.
        con.execute("CREATE TABLE zero_rates AS SELECT * FROM panel")
        n_rows = con.execute("SELECT COUNT(*) FROM par_rates").fetchone()[0]

    issuers = long.get_column("source").n_unique()
    print(f"Wrote {args.out}")
    print(f"  par_rates: {n_rows:,} rows x {len(TERM_LABELS) + 2} columns")
    print(f"  {issuers} issuers, {args.days} days, decay={truth.decay:g} (scale)")
    print(f"  true sigma diag: {[f'{v:.4f}' for v in truth.sigma.diagonal()]}")
    print("  true adjustment at 30Y: " f"{truth.adjustment[-1] * 1e4:.1f} bp")


if __name__ == "__main__":
    main()
