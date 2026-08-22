"""Configuration for a single-network analysis run."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class PipelineConfig:
    """User choices that drive the notebook-equivalent pipeline."""

    db_path: Path
    table: str
    date_column: str
    name_column: str
    value_column: str
    # Optional raw SQL boolean expression, inserted after WHERE in the load
    # query. None means no filtering. This is the only filter mode -- the older
    # "pin a column to one value" mode was a strict subset of it and is gone.
    where_clause: str | None = None
    transforms: list[str] = field(default_factory=list)
    measure: str = "distance_correlation"
    date_start: date | None = None
    date_end: date | None = None
    independent_threshold: float = 0.33
    title: str = "Distance Correlation Network"

    # --- Yield-curve panel ------------------------------------------------
    # Set when the source table is a wide curve panel (one column per term).
    # ``date_column``/``name_column``/``value_column`` then describe the *long*
    # frame produced by ``yield_curve.load_long_panel``, not the stored table.
    issuer_column: str = ""
    term_columns: list[str] = field(default_factory=list)
    network_kind: str = "issuer_by_term"
    # ``(term, issuer)`` pairs the user kept in the User Filter dialog.
    # None means "no manual filtering"; an empty collection means "nothing".
    cell_mask: list[tuple[str, str]] | None = None
