"""Ordered, pluggable dataframe transforms.

New transforms can be registered in ``TRANSFORMS`` without changing the GUI.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

TransformFn = Callable[[pl.DataFrame, str, str, list[str]], pl.DataFrame]


def daily_returns(
    df: pl.DataFrame,
    date_column: str,
    name_column: str,
    value_columns: list[str],
) -> pl.DataFrame:
    """Replace numeric series columns with per-node daily simple returns."""
    if not value_columns:
        return df
    return (
        df.sort(name_column, date_column)
        .with_columns(
            pl.col(col).pct_change().over(name_column).alias(col)
            for col in value_columns
        )
        .drop_nulls(subset=value_columns)
    )


TRANSFORMS: dict[str, TransformFn] = {
    "daily_returns": daily_returns,
}

TRANSFORM_LABELS: dict[str, str] = {
    "daily_returns": "Daily returns (per node)",
}


def available_transforms() -> list[tuple[str, str]]:
    """Return ``(id, label)`` pairs in a stable display order."""
    return [(key, TRANSFORM_LABELS.get(key, key)) for key in TRANSFORMS]


def apply_transforms(
    df: pl.DataFrame,
    transform_ids: list[str],
    date_column: str,
    name_column: str,
    value_columns: list[str],
) -> pl.DataFrame:
    """Apply transforms in the given order."""
    out = df
    for transform_id in transform_ids:
        fn = TRANSFORMS.get(transform_id)
        if fn is None:
            raise ValueError(f"Unknown transform: {transform_id!r}")
        out = fn(out, date_column, name_column, value_columns)
    return out
