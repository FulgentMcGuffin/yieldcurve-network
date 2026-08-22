"""Generate a fake ``par_rates`` panel for testing the GUI without real data.

Mirrors the structure of ``D:\\data\\duckdb\\ycs_data.duckdb``: a **wide** table
with ``date``, ``source`` and one numeric column per maturity term.

Rates come from a Nelson-Siegel curve whose level/slope/curvature factors follow
correlated random walks, plus a per-issuer credit spread -- enough structure
that the correlation networks have real communities rather than noise. Coverage
is deliberately ragged (some issuers lack long or short maturities, some start
late) so the User Filter grid has genuine holes to display.

Usage:
    uv run python scripts/make_fake_par_rates.py [--out PATH] [--seed N]
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

# Maturities in years, and the "0.5Y" style labels used as column names.
TERMS_YEARS: tuple[float, ...] = (0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30)

# Issuers grouped into blocs that share factor shocks, so community detection
# has something real to find.
BLOCS: dict[str, tuple[str, ...]] = {
    "core": ("DE", "NL", "AT", "FI"),
    "semi_core": ("FR", "BE", "IE"),
    "periphery": ("IT", "ES", "PT", "GR"),
    "non_emu": ("UK", "US", "CH", "JP"),
}

# Base credit spread in decimal yield added to every maturity of an issuer.
BLOC_SPREAD: dict[str, float] = {
    "core": 0.0000,
    "semi_core": 0.0025,
    "periphery": 0.0090,
    "non_emu": 0.0015,
}

# Issuers that do not quote the full curve, and the terms they omit. Produces
# empty cells in the User Filter grid.
MISSING_TERMS: dict[str, tuple[float, ...]] = {
    "GR": (0.5, 1, 20, 30),
    "PT": (20, 30),
    "IE": (15, 20),
    "CH": (0.5, 15, 20),
    "FI": (30,),
}

# Issuers whose history starts after the panel does.
LATE_START: dict[str, int] = {"GR": 400, "IE": 150}


def term_label(years: float) -> str:
    """``0.5`` -> ``"0.5Y"``, ``10`` -> ``"10Y"``."""
    return f"{years:g}Y"


def nelson_siegel(taus: np.ndarray, b0: float, b1: float, b2: float, lam: float = 1.8):
    """Nelson-Siegel yields for maturities ``taus`` (years)."""
    x = taus / lam
    # ``taus`` never contains 0, so the (1 - e^-x)/x limit at 0 is not needed.
    slope_load = (1.0 - np.exp(-x)) / x
    curve_load = slope_load - np.exp(-x)
    return b0 + b1 * slope_load + b2 * curve_load


def business_days(start: date, n: int) -> list[date]:
    """``n`` weekday dates starting at (or after) ``start``."""
    out: list[date] = []
    cursor = start
    while len(out) < n:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def build_panel(n_days: int, seed: int) -> pl.DataFrame:
    """Wide ``(date, source, <term columns>)`` frame."""
    rng = np.random.default_rng(seed)
    dates = business_days(date(2015, 1, 1), n_days)
    taus = np.array(TERMS_YEARS, dtype=float)
    labels = [term_label(t) for t in TERMS_YEARS]

    # One shared global factor path plus a path per bloc: issuers in a bloc move
    # together, blocs move together more weakly, which is the structure the
    # multi-layer community detection should recover.
    def walk(scale: float) -> np.ndarray:
        return np.cumsum(rng.normal(0.0, scale, size=(n_days, 3)), axis=0)

    global_path = walk(0.00035)
    bloc_paths = {name: walk(0.00025) for name in BLOCS}

    base = np.array([0.022, -0.012, 0.010])  # level, slope, curvature

    frames: list[pl.DataFrame] = []
    for bloc, issuers in BLOCS.items():
        for issuer in issuers:
            idio = walk(0.00018)
            factors = base + global_path + bloc_paths[bloc] + idio

            curves = np.vstack(
                [nelson_siegel(taus, b0, b1, b2) for b0, b1, b2 in factors]
            )
            curves += BLOC_SPREAD[bloc]
            # Small independent measurement noise per maturity.
            curves += rng.normal(0.0, 0.00012, size=curves.shape)

            frame = pl.DataFrame(
                {"date": dates, "source": [issuer] * n_days}
                | {label: curves[:, i] for i, label in enumerate(labels)}
            )

            skip = LATE_START.get(issuer, 0)
            if skip:
                frame = frame.slice(skip)

            for missing in MISSING_TERMS.get(issuer, ()):
                frame = frame.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias(term_label(missing))
                )

            frames.append(frame)

    return pl.concat(frames, how="vertical").sort("date", "source")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "ycs_fake.duckdb",
        help="output DuckDB file (created/overwritten)",
    )
    parser.add_argument("--days", type=int, default=1250, help="business days (~5y)")
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    panel = build_panel(args.days, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    con = duckdb.connect(str(args.out))
    try:
        con.register("panel", panel.to_arrow())
        con.execute("CREATE TABLE par_rates AS SELECT * FROM panel")
        # A second table so the GUI's "default to par_rates" pick is observable.
        con.execute("CREATE TABLE zero_rates AS SELECT * FROM panel")
        n_rows = con.execute("SELECT COUNT(*) FROM par_rates").fetchone()[0]
    finally:
        con.close()

    terms = [term_label(t) for t in TERMS_YEARS]
    issuers = [i for group in BLOCS.values() for i in group]
    print(f"Wrote {args.out}")
    print(f"  par_rates: {n_rows:,} rows x {len(terms) + 2} columns")
    print(f"  issuers ({len(issuers)}): {', '.join(issuers)}")
    print(f"  terms   ({len(terms)}): {', '.join(terms)}")
    print(f"  dates   : {panel['date'].min()} .. {panel['date'].max()}")


if __name__ == "__main__":
    main()
