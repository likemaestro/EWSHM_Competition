"""Duration-based preprocessing helpers."""

from __future__ import annotations

import pandas as pd

from aquinas_toolkit.io import AquinasReader


def filter_records_by_min_duration(
    reader: AquinasReader,
    min_duration_seconds: float = 10.0,
    quantity: str | None = "ACC",
    axis: str | None = "Z",
    deck: str | None = None,
) -> pd.DataFrame:
    """Return index-table rows that satisfy a minimum record duration.

    Parameters
    ----------
    reader:
        Dataset reader for one AQUINAS_SET folder.
    min_duration_seconds:
        Minimum record duration to retain.
    quantity:
        Optional sensor quantity filter, for example ``"ACC"`` or ``"STR"``.
    axis:
        Optional acceleration axis filter, for example ``"Z"``.
    deck:
        Optional deck filter such as ``"OLD"`` or ``"NEW"``.
    """
    filtered_tables: list[pd.DataFrame] = []
    sensor_summary = reader.summarize_sensor_records(quantity=quantity, axis=axis)
    if deck is not None:
        sensor_summary = sensor_summary[sensor_summary["deck"] == deck.upper()].reset_index(drop=True)

    for sensor_name in sensor_summary["sensor_name"]:
        index_df = reader.load_index_table(sensor_name).copy()
        duration_col = _require_duration_column(index_df)
        kept = index_df[index_df[duration_col] >= min_duration_seconds].copy()
        kept["sensor_name"] = sensor_name
        kept["dataset"] = reader.set_name
        filtered_tables.append(kept)

    if not filtered_tables:
        return pd.DataFrame()

    return pd.concat(filtered_tables, ignore_index=True)


def summarize_min_duration_filter(
    reader: AquinasReader,
    min_duration_seconds: float = 10.0,
    quantity: str | None = "ACC",
    axis: str | None = "Z",
    deck: str | None = None,
) -> pd.DataFrame:
    """Summarize keep/remove counts after minimum-duration filtering."""
    rows = []
    sensor_summary = reader.summarize_sensor_records(quantity=quantity, axis=axis)
    if deck is not None:
        sensor_summary = sensor_summary[sensor_summary["deck"] == deck.upper()].reset_index(drop=True)

    for sensor_row in sensor_summary.to_dict("records"):
        sensor_name = sensor_row["sensor_name"]
        index_df = reader.load_index_table(sensor_name)
        duration_col = _require_duration_column(index_df)
        kept_count = int((index_df[duration_col] >= min_duration_seconds).sum())
        total_count = int(len(index_df))
        removed_count = total_count - kept_count

        rows.append(
            {
                **sensor_row,
                "min_duration_seconds": float(min_duration_seconds),
                "kept_count": kept_count,
                "removed_count": removed_count,
                "kept_fraction": kept_count / total_count if total_count else 0.0,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _require_duration_column(index_df: pd.DataFrame) -> str:
    for candidate in ("Duration", "duration"):
        if candidate in index_df.columns:
            return candidate
    raise KeyError("Index table must contain a Duration column for minimum-duration filtering.")