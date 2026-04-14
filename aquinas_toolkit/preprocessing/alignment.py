"""
Organizer-faithful timestamp alignment helpers for AQUINAS preprocessing.

The implementation mirrors the ``Synchro()`` and two-pass shrinking
behavior used by ``AQUINAS_Explorer.R``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aquinas_toolkit.preprocessing.core import LoadedEventGroup


SYNCHRO_PASSES = 2


@dataclass(frozen=True)
class AlignedEvent:
    """Aligned multi-sensor waveform for one grouped event."""

    event_id: str
    set_name: str
    deck: str
    start_time_utc: pd.Timestamp
    end_time_utc: pd.Timestamp
    reference_sensor: str
    active_sensors: list[str]
    aligned_waveform: pd.DataFrame
    alignment_diagnostics: dict[str, Any]
    zeroing_method: str


def align_event_group(
    event_group: LoadedEventGroup,
    *,
    method: str = "r_synchro",
) -> AlignedEvent:
    """
    Align all sensors in an event group using organizer ``Synchro()`` logic.

    The first active sensor in organizer order becomes the reference
    seed. Each subsequent sensor shrinks the reference timestamps via
    the organizer's two-pass process.
    """
    if method != "r_synchro":
        raise ValueError(f"Unsupported alignment method: {method}")

    active_sensors = list(event_group.waveforms)
    if not active_sensors:
        raise ValueError("Cannot align an event group without active sensors.")

    organizer_frames = [
        (sensor_name, event_group.waveforms[sensor_name][1]) for sensor_name in active_sensors
    ]
    aligned_measures = organizer_align_sensor_frames(organizer_frames)
    rows_reference = int(len(organizer_frames[0][1]))
    rows_after_alignment = int(len(aligned_measures))
    reference_sensor = active_sensors[0]

    aligned_waveform = pd.DataFrame({"timestamp_utc": aligned_measures["timestamp"]}) if not aligned_measures.empty else pd.DataFrame(columns=["timestamp_utc"])
    match_stats: dict[str, dict[str, int]] = {}
    for sensor_name, frame in organizer_frames:
        n_sensor_rows = len(frame) if not frame.empty else 0
        matched_rows = rows_after_alignment
        unmatched_rows = n_sensor_rows - matched_rows if n_sensor_rows >= matched_rows else 0
        match_stats[sensor_name] = {
            "matched_rows": matched_rows,
            "unmatched_rows": unmatched_rows,
        }
        if not aligned_measures.empty:
            # Data is already numeric from load_raw_file_prepped
            aligned_waveform[sensor_name] = aligned_measures[sensor_name].to_numpy(dtype=float)
        else:
            aligned_waveform[sensor_name] = pd.Series(dtype=float)

    diagnostics: dict[str, Any] = {
        "method": method,
        "reference_sensor": reference_sensor,
        "reference_policy": "first_selected",
        "passes": SYNCHRO_PASSES,
        "rows_reference": rows_reference,
        "rows_after_alignment": rows_after_alignment,
        "match_stats": match_stats,
    }

    return AlignedEvent(
        event_id=event_group.event_id,
        set_name=event_group.set_name,
        deck=event_group.deck,
        start_time_utc=event_group.start_time_utc,
        end_time_utc=event_group.end_time_utc,
        reference_sensor=reference_sensor,
        active_sensors=active_sensors,
        aligned_waveform=aligned_waveform.reset_index(drop=True),
        alignment_diagnostics=diagnostics,
        zeroing_method=event_group.zeroing_method,
    )


def organizer_align_sensor_frames(
    sensor_frames: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Return organizer-style ``DataMesures`` for an ordered sensor selection.

    Empty frames are preserved as all-NaN columns (use ``pd.isna()`` to detect
    missing sensors). If the first selected sensor has no rows, the organizer
    helper reports ``NO DATA``; this implementation returns an empty DataFrame
    with the expected columns so callers can make the same decision.
    """
    if not sensor_frames:
        return pd.DataFrame(columns=["timestamp"])

    sensor_names = [name for name, _ in sensor_frames]
    all_columns = ["timestamp", *sensor_names]

    first_frame = sensor_frames[0][1].reset_index(drop=True)
    if first_frame.empty:
        return pd.DataFrame(columns=all_columns)

    # Work with numpy arrays internally to avoid DataFrame overhead.
    # ref_ts/ref_vals track the reference sensor's timestamps and values;
    # sensor_vals[i] holds the aligned value array for sensor i.
    ref_ts = _to_datetime64_ns(first_frame["timestamp"])
    ref_vals = first_frame[sensor_names[0]].to_numpy(dtype=float)

    # sensor_vals[0] will be rebuilt at the end from ref_vals
    sensor_vals: list[np.ndarray | None] = [None] * len(sensor_frames)
    sensor_vals[0] = ref_vals

    sensor_ts_arrays: list[np.ndarray | None] = [None] * len(sensor_frames)
    sensor_val_arrays: list[np.ndarray | None] = [None] * len(sensor_frames)
    for i in range(1, len(sensor_frames)):
        frame = sensor_frames[i][1].reset_index(drop=True)
        if frame.empty:
            sensor_ts_arrays[i] = None
            sensor_val_arrays[i] = None
        else:
            sensor_ts_arrays[i] = _to_datetime64_ns(frame["timestamp"])
            sensor_val_arrays[i] = frame[sensor_names[i]].to_numpy(dtype=float)

    for _ in range(SYNCHRO_PASSES):
        for i in range(1, len(sensor_frames)):
            if sensor_ts_arrays[i] is None:
                continue

            time_ref = synchro_indices_arrays(ref_ts, sensor_ts_arrays[i])
            non_zero_mask = time_ref != 0
            if not non_zero_mask.any():
                ref_ts = ref_ts[:0]
                ref_vals = ref_vals[:0]
                sensor_vals[0] = ref_vals
                sensor_ts_arrays[i] = None
                sensor_val_arrays[i] = None
                sensor_vals[i] = None
                continue

            non_zero = time_ref[non_zero_mask]
            unique_ref = pd.unique(non_zero)
            ref_indices = (unique_ref - 1).astype(int)
            ref_ts = ref_ts[ref_indices]
            ref_vals = ref_vals[ref_indices]
            sensor_vals[0] = ref_vals

            # Keep only first occurrence of each non-zero synchro index
            nz_positions = np.flatnonzero(non_zero_mask)
            _, first_idx = np.unique(time_ref[nz_positions], return_index=True)
            keep_positions = nz_positions[first_idx]

            sensor_vals[i] = sensor_val_arrays[i][keep_positions].astype(float)

    if len(ref_ts) == 0:
        return pd.DataFrame(columns=all_columns)

    # Build output DataFrame in one shot
    data: dict[str, np.ndarray] = {"timestamp": ref_ts}
    for i, name in enumerate(sensor_names):
        if sensor_vals[i] is not None and len(sensor_vals[i]) == len(ref_ts):
            data[name] = sensor_vals[i]
        else:
            data[name] = np.full(len(ref_ts), np.nan)

    return pd.DataFrame(data)


def _to_datetime64_ns(timestamps: pd.Series | list | np.ndarray) -> np.ndarray:
    """Convert timestamps to datetime64[ns] numpy array, skipping re-parsing when already datetime64."""
    if isinstance(timestamps, (pd.Series, pd.Index)):
        if pd.api.types.is_datetime64_any_dtype(timestamps):
            return timestamps.to_numpy(dtype="datetime64[ns]")
    elif isinstance(timestamps, np.ndarray) and timestamps.dtype.kind == "M":
        return timestamps.astype("datetime64[ns]")
    return pd.to_datetime(timestamps, utc=True).to_numpy(dtype="datetime64[ns]")


def synchro_indices_arrays(
    reference: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Core synchro logic on pre-converted datetime64[ns] numpy arrays.

    Uses ``np.searchsorted`` for an O(n log n) single-sort approach
    instead of the 3-sort merge used previously.
    """
    n_ref = len(reference)
    n_target = len(target)
    if n_target == 0:
        return np.zeros(0, dtype=int)
    if n_ref == 0:
        return np.zeros(n_target, dtype=int)

    ref_order = np.argsort(reference, kind="mergesort")
    ref_sorted = reference[ref_order]

    # For each target, find the rightmost reference <= target
    # searchsorted('right') gives index of first ref > target, so subtract 1
    pos = np.searchsorted(ref_sorted, target, side="right") - 1

    result = np.zeros(n_target, dtype=int)
    valid = pos >= 0
    if valid.any():
        # Map back from sorted position to original 1-based index
        result[valid] = ref_order[pos[valid]] + 1

    return result


def synchro_indices(
    reference_timestamps: pd.Series | list[pd.Timestamp],
    target_timestamps: pd.Series | list[pd.Timestamp],
) -> np.ndarray:
    """
    Replicate the organizer ``Synchro()`` helper.

    For each target timestamp, return the 1-based index of the latest
    reference timestamp that is not after it. Return ``0`` when no such
    reference sample exists.
    """
    reference = _to_datetime64_ns(reference_timestamps)
    target = _to_datetime64_ns(target_timestamps)
    return synchro_indices_arrays(reference, target)


def _empty_measure_frame(sensor_name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", sensor_name])
