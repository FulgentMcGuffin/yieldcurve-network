"""Wide yield-curve panel -> long ``(date, issuer, term, rate)`` frame.

The source tables (``par_rates``, ``zero_rates``) are stored **wide**: one row
per ``(date, issuer)`` and one numeric column per maturity term. Every network
in this application is built from the long form, so this module owns the
detection of which column plays which role and the reshape itself.

Two — and only two — multi-layer networks can be built from such a panel:

* ``ISSUER_BY_TERM`` — one layer per term, nodes are issuers.
* ``TERM_BY_ISSUER`` — one layer per issuer, nodes are terms.

Both are expressed by pointing ``PipelineConfig.name_column`` and
``MLNConfig.layer_column`` at the two non-date columns of the long frame, so
the MLN analysis code needs no knowledge of yield curves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import polars as pl

from ycn.backends.base import _normalize_sql_type

from .config import PipelineConfig
from .data_access import load_table, open_backend

# Canonical names for the two columns created by the unpivot. Uniquified against
# the source table's own column names by :func:`detect_panel`.
TERM_COLUMN = "term"
RATE_COLUMN = "rate"

# Types that can hold a rate. A term column must be one of these -- a text
# column whose *name* happens to parse as a term is not a rate series.
_NUMERIC_KINDS = {"REAL", "INTEGER"}

# Ordered term spellings, most specific first. Each maps a label to a maturity
# in years so terms sort numerically ("2Y" before "10Y") rather than
# alphabetically, which is the order that makes a yield curve read correctly.
_TERM_PATTERNS: tuple[tuple[re.Pattern[str], "object"], ...] = (
    # Zero-padded 'p'-for-point form used by the ycs_data tables: Y000p5, Y030p0.
    (
        re.compile(r"^Y(\d+)P(\d+)$", re.I),
        lambda m: float(f"{int(m.group(1))}.{m.group(2)}"),
    ),
    (
        re.compile(r"^(\d+(?:\.\d+)?)\s*Y(?:R|RS|EAR|EARS)?$", re.I),
        lambda m: float(m.group(1)),
    ),
    (
        re.compile(r"^(\d+(?:\.\d+)?)\s*M(?:O|OS|ONTH|ONTHS)?$", re.I),
        lambda m: float(m.group(1)) / 12.0,
    ),
    (
        re.compile(r"^(\d+(?:\.\d+)?)\s*W(?:K|KS|EEK|EEKS)?$", re.I),
        lambda m: float(m.group(1)) / 52.0,
    ),
    (
        re.compile(r"^(\d+(?:\.\d+)?)\s*D(?:AY|AYS)?$", re.I),
        lambda m: float(m.group(1)) / 365.0,
    ),
)


class NetworkKind(str, Enum):
    """Which of the two panel axes becomes the layer axis."""

    ISSUER_BY_TERM = "issuer_by_term"
    TERM_BY_ISSUER = "term_by_issuer"

    @property
    def label(self) -> str:
        return {
            NetworkKind.ISSUER_BY_TERM: "Issuer Network by Term",
            NetworkKind.TERM_BY_ISSUER: "Term Network by Issuer",
        }[self]


def parse_term_years(label: str) -> float | None:
    """Maturity of a term label in years, or None if it is not a term.

    Recognises ``0.5Y``/``10Y``, ``6M``, ``4W``, ``30D`` and the zero-padded
    ``Y000p5`` form. Used both to identify term columns and to order them.
    """
    text = str(label).strip()
    for pattern, to_years in _TERM_PATTERNS:
        match = pattern.match(text)
        if match is not None:
            try:
                return float(to_years(match))
            except (TypeError, ValueError):
                return None
    return None


def sort_terms(labels) -> list[str]:
    """Term labels in maturity order; unparseable ones sort last, alphabetically."""

    def key(label: str) -> tuple[int, float, str]:
        years = parse_term_years(label)
        return (1, 0.0, str(label)) if years is None else (0, years, str(label))

    return sorted((str(x) for x in labels), key=key)


def _uniquify(name: str, taken: set[str]) -> str:
    """A variant of ``name`` not already in ``taken``."""
    if name not in taken:
        return name
    i = 2
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


@dataclass(frozen=True)
class CurvePanel:
    """Role assignment for the columns of a wide yield-curve table.

    Args:
        date_column: Observation date column (chosen by the user).
        issuer_column: Column identifying the curve's issuer/source.
        term_columns: Wide rate columns, in maturity order.
        term_column: Name the unpivoted term labels get in the long frame.
        rate_column: Name the unpivoted rate values get in the long frame.
    """

    date_column: str
    issuer_column: str
    term_columns: tuple[str, ...]
    term_column: str = TERM_COLUMN
    rate_column: str = RATE_COLUMN

    @property
    def is_usable(self) -> bool:
        """True when there is enough structure to build any network at all."""
        return bool(self.issuer_column) and len(self.term_columns) >= 2

    def node_column(self, kind: NetworkKind) -> str:
        """Long-frame column whose values become graph nodes."""
        return (
            self.issuer_column
            if kind is NetworkKind.ISSUER_BY_TERM
            else self.term_column
        )

    def layer_column(self, kind: NetworkKind) -> str:
        """Long-frame column whose distinct values become MLN layers."""
        return (
            self.term_column
            if kind is NetworkKind.ISSUER_BY_TERM
            else self.issuer_column
        )


def detect_panel(
    db_path: Path | str,
    table: str,
    date_column: str,
) -> CurvePanel:
    """Infer issuer and term columns for a wide curve table.

    Term columns are the numeric columns whose *name* parses as a maturity
    (see :func:`parse_term_years`). The issuer column is the first remaining
    text column, falling back to the first remaining column of any type. Both
    the node-name and series-value pickers were removed from the GUI, so this
    inference is the only thing standing between a table and a network --
    it deliberately never raises, and callers check :attr:`CurvePanel.is_usable`.
    """
    with open_backend(db_path) as db:
        columns = [
            (c.name, _normalize_sql_type(c.type)) for c in db.get_schema(table).columns
        ]

    rest = [(name, kind) for name, kind in columns if name != date_column]
    terms = [
        name
        for name, kind in rest
        if kind in _NUMERIC_KINDS and parse_term_years(name) is not None
    ]
    non_terms = [(name, kind) for name, kind in rest if name not in set(terms)]

    issuer = ""
    for name, kind in non_terms:
        if kind == "TEXT":
            issuer = name
            break
    if not issuer and non_terms:
        issuer = non_terms[0][0]

    taken = {name for name, _ in columns}
    return CurvePanel(
        date_column=date_column,
        issuer_column=issuer,
        term_columns=tuple(sort_terms(terms)),
        term_column=_uniquify(TERM_COLUMN, taken),
        rate_column=_uniquify(RATE_COLUMN, taken),
    )


def load_long_panel(
    cfg: PipelineConfig,
    panel: CurvePanel,
    *,
    apply_cell_mask: bool = True,
) -> pl.DataFrame:
    """Load the wide table and reshape it to long ``(date, issuer, term, rate)``.

    Applies, in order: the SQL ``WHERE`` filter (pushed down to the query), the
    date range, the unpivot, null-dropping, and finally the user's cell mask.
    The mask is applied last by design -- the
    User Filter dialog is defined as operating on data that has already been
    loaded and filtered by the Optional Filter section, so it must see (and
    cut from) exactly this frame.

    Transforms are deliberately *not* applied: ``mln.build_layer_graphs``
    applies them per layer, which is the only correct order for per-node
    operations like ``daily_returns``.
    """
    cols = [cfg.date_column, panel.issuer_column, *panel.term_columns]
    cols = list(dict.fromkeys(c for c in cols if c))

    df = load_table(cfg.db_path, cfg.table, columns=cols, where_clause=cfg.where_clause)

    df = df.with_columns(pl.col(cfg.date_column).cast(pl.Date, strict=False))

    if cfg.date_start is not None:
        df = df.filter(pl.col(cfg.date_column) >= cfg.date_start)
    if cfg.date_end is not None:
        df = df.filter(pl.col(cfg.date_column) <= cfg.date_end)

    present = [c for c in panel.term_columns if c in df.columns]
    if not present:
        return pl.DataFrame(
            schema={
                cfg.date_column: pl.Date,
                panel.issuer_column: pl.Utf8,
                panel.term_column: pl.Utf8,
                panel.rate_column: pl.Float64,
            }
        )

    long = df.unpivot(
        on=present,
        index=[cfg.date_column, panel.issuer_column],
        variable_name=panel.term_column,
        value_name=panel.rate_column,
    ).with_columns(
        pl.col(panel.issuer_column).cast(pl.Utf8),
        pl.col(panel.rate_column).cast(pl.Float64, strict=False),
    )

    long = long.drop_nulls(
        subset=[
            cfg.date_column,
            panel.issuer_column,
            panel.term_column,
            panel.rate_column,
        ]
    )

    if apply_cell_mask and cfg.cell_mask is not None:
        long = filter_by_cell_mask(long, panel, cfg.cell_mask)

    return long.sort(cfg.date_column, panel.issuer_column, panel.term_column)


def filter_by_cell_mask(
    long: pl.DataFrame,
    panel: CurvePanel,
    mask,
) -> pl.DataFrame:
    """Keep only rows whose ``(term, issuer)`` pair is in ``mask``.

    An empty mask keeps nothing, which is the honest reading of "the user
    unchecked every cell" -- callers surface that as an error rather than
    silently building a network from the full panel.
    """
    pairs = {(str(t), str(i)) for t, i in mask}
    if not pairs:
        return long.clear()
    keep = pl.DataFrame(
        {
            panel.term_column: [t for t, _ in sorted(pairs)],
            panel.issuer_column: [i for _, i in sorted(pairs)],
        },
        schema={panel.term_column: pl.Utf8, panel.issuer_column: pl.Utf8},
    )
    return long.join(keep, on=[panel.term_column, panel.issuer_column], how="inner")


def available_cells(long: pl.DataFrame, panel: CurvePanel) -> set[tuple[str, str]]:
    """Distinct ``(term, issuer)`` pairs that actually carry data."""
    if long.is_empty():
        return set()
    unique = long.select(panel.term_column, panel.issuer_column).unique()
    return {
        (str(t), str(i))
        for t, i in zip(
            unique.get_column(panel.term_column).to_list(),
            unique.get_column(panel.issuer_column).to_list(),
        )
    }
