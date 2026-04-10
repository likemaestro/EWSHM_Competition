"""Metadata-only helpers built on AQUINAS index tables."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from aquinas_toolkit.io.reader import AquinasReader


def load_sensor_metadata(
    readers: Sequence[AquinasReader],
    sensor_name: str,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Load one sensor's event metadata across one or more AQUINAS readers.

    This helper uses the per-sensor ``TABLE_*.json`` files only. It does not
    read raw waveform files.

    Parameters
    ----------
    readers:
        One or more ``AquinasReader`` instances to merge.
    sensor_name:
        Sensor name to load from each reader.
    columns:
        Optional subset of AQUINAS table columns to keep. The helper always adds
        a ``dataset`` column that records the source SET folder.
    """

    selected_columns = list(columns) if columns is not None else None
    frames: list[pd.DataFrame] = []

    for reader in readers:
        index_df = reader.load_index_table(sensor_name).copy()
        if selected_columns is not None:
            index_df = index_df.loc[:, selected_columns].copy()

        index_df["dataset"] = reader.set_name

        for column_name in ("Start_Time", "End_Time"):
            if column_name in index_df.columns:
                index_df[column_name] = pd.to_datetime(index_df[column_name])

        frames.append(index_df)

    if not frames:
        empty_columns = list(selected_columns or [])
        if "dataset" not in empty_columns:
            empty_columns.append("dataset")
        return pd.DataFrame(columns=empty_columns)

    metadata = pd.concat(frames, ignore_index=True)
    if "Start_Time" in metadata.columns:
        metadata = metadata.sort_values("Start_Time").reset_index(drop=True)
    else:
        metadata = metadata.reset_index(drop=True)

    return metadata
