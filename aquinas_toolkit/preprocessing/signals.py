"""Signal-level preprocessing helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from aquinas_toolkit.io import AquinasReader
from aquinas_toolkit.preprocessing.duration import filter_records_by_min_duration


def find_common_sensor_events(
    reader: AquinasReader,
    min_duration_seconds: float = 10.0,
    quantity: str | None = "ACC",
    axis: str | None = "Z",
    deck: str | None = None,
) -> pd.DataFrame:
    """Return events present in every selected sensor after duration filtering."""
    filtered = filter_records_by_min_duration(
        reader,
        min_duration_seconds=min_duration_seconds,
        quantity=quantity,
        axis=axis,
        deck=deck,
    )
    if filtered.empty:
        return pd.DataFrame()

    start_col = _require_column(filtered, ["Start_Time", "start_time"])
    end_col = _require_column(filtered, ["End_Time", "end_time"])

    expected_sensors = sorted(filtered["sensor_name"].unique())
    grouped = (
        filtered.groupby([start_col, end_col], as_index=False)
        .agg(
            sensor_count=("sensor_name", "nunique"),
            sensor_names=("sensor_name", lambda values: sorted(set(values))),
        )
        .rename(columns={start_col: "Start_Time", end_col: "End_Time"})
    )
    common = grouped[grouped["sensor_count"] == len(expected_sensors)].copy()
    common["dataset"] = reader.set_name
    common["deck"] = deck.upper() if deck is not None else "ALL"
    common["channel_count"] = len(expected_sensors)
    return common.sort_values(["Start_Time", "End_Time"]).reset_index(drop=True)


def load_common_event_waveform_matrix(
    reader: AquinasReader,
    start_time: str,
    end_time: str,
    sensor_names: Iterable[str],
) -> pd.DataFrame:
    """Load one aligned multichannel event matrix for the provided sensors."""
    trimmed_waveforms: list[pd.Series] = []
    output_columns: list[str] = []
    timestamp_series: pd.Series | None = None
    min_length: int | None = None

    for sensor_name in sensor_names:
        index_df = reader.load_index_table(sensor_name)
        start_col = _require_column(index_df, ["Start_Time", "start_time"])
        end_col = _require_column(index_df, ["End_Time", "end_time"])
        matches = index_df[(index_df[start_col] == start_time) & (index_df[end_col] == end_time)]
        if matches.empty:
            raise ValueError(
                f"No event found for sensor {sensor_name} with Start_Time={start_time} and End_Time={end_time}."
            )

        row_index = int(matches.index[0])
        _, waveform = reader.read_record(sensor_name=sensor_name, row_index=row_index)
        signal_col = _select_signal_column(waveform, sensor_name)
        signal = waveform[signal_col].reset_index(drop=True)

        if timestamp_series is None and "timestamp" in waveform.columns:
            timestamp_series = waveform["timestamp"].reset_index(drop=True)

        min_length = len(signal) if min_length is None else min(min_length, len(signal))
        trimmed_waveforms.append(signal)
        output_columns.append(sensor_name)

    if min_length is None:
        return pd.DataFrame()

    data = {
        sensor_name: series.iloc[:min_length].to_numpy()
        for sensor_name, series in zip(output_columns, trimmed_waveforms, strict=True)
    }
    matrix = pd.DataFrame(data)
    if timestamp_series is not None:
        matrix.insert(0, "timestamp", timestamp_series.iloc[:min_length].to_numpy())
    return matrix


def bandpass_filter_waveform_matrix(
    waveform_matrix: pd.DataFrame | np.ndarray,
    sampling_rate_hz: float = 100.0,
    low_hz: float = 0.5,
    high_hz: float = 20.0,
    order: int = 4,
) -> pd.DataFrame | np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter to a multichannel matrix."""
    if low_hz <= 0:
        raise ValueError("low_hz must be greater than 0 for band-pass filtering.")
    nyquist_hz = 0.5 * sampling_rate_hz
    if high_hz >= nyquist_hz:
        raise ValueError("high_hz must be below the Nyquist frequency.")
    if low_hz >= high_hz:
        raise ValueError("low_hz must be strictly smaller than high_hz.")

    if isinstance(waveform_matrix, pd.DataFrame):
        timestamp = waveform_matrix["timestamp"] if "timestamp" in waveform_matrix.columns else None
        numeric = waveform_matrix.drop(columns=["timestamp"], errors="ignore")
        sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sampling_rate_hz, output="sos")
        filtered = sosfiltfilt(sos, numeric.to_numpy(), axis=0)
        filtered_df = pd.DataFrame(filtered, columns=numeric.columns, index=waveform_matrix.index)
        if timestamp is not None:
            filtered_df.insert(0, "timestamp", timestamp)
        return filtered_df

    matrix = np.asarray(waveform_matrix, dtype=float)
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sampling_rate_hz, output="sos")
    return sosfiltfilt(sos, matrix, axis=0)


def _require_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lower_map = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    raise KeyError(f"None of the required columns were found: {candidates}")


def _select_signal_column(waveform: pd.DataFrame, sensor_name: str) -> str:
    if sensor_name in waveform.columns:
        return sensor_name

    non_timestamp = [column for column in waveform.columns if column != "timestamp"]
    if len(non_timestamp) == 1:
        return non_timestamp[0]
    if "value" in waveform.columns:
        return "value"
    raise KeyError(
        f"Could not determine signal column for sensor {sensor_name}. Available columns: {list(waveform.columns)}"
    )