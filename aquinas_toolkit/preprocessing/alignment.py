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
        (sensor_name, event_group.waveforms[sensor_name][1].copy()) for sensor_name in active_sensors
    ]
    aligned_measures = organizer_align_sensor_frames(organizer_frames)
    rows_reference = int(len(organizer_frames[0][1]))
    rows_after_alignment = int(len(aligned_measures))
    reference_sensor = active_sensors[0]

    aligned_waveform = pd.DataFrame({"timestamp_utc": aligned_measures["timestamp"]}) if not aligned_measures.empty else pd.DataFrame(columns=["timestamp_utc"])
    match_stats: dict[str, dict[str, int]] = {}
    for sensor_name, frame in organizer_frames:
        series = pd.to_numeric(frame[sensor_name], errors="coerce") if not frame.empty else pd.Series(dtype=float)
        matched_rows = rows_after_alignment
        unmatched_rows = int(len(series) - matched_rows) if len(series) >= matched_rows else 0
        match_stats[sensor_name] = {
            "matched_rows": matched_rows,
            "unmatched_rows": unmatched_rows,
        }
        if not aligned_measures.empty:
            aligned_waveform[sensor_name] = pd.to_numeric(
                aligned_measures[sensor_name],
                errors="coerce",
            ).to_numpy(dtype=float)
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

    dynamic = [(sensor_name, frame.reset_index(drop=True).copy()) for sensor_name, frame in sensor_frames]
    if dynamic[0][1].empty:
        columns = ["timestamp", *[sensor_name for sensor_name, _ in dynamic]]
        return pd.DataFrame(columns=columns)

    data_m: list[pd.DataFrame] = [_empty_measure_frame(sensor_name) for sensor_name, _ in dynamic]
    data_m[0] = dynamic[0][1].copy()
    for _ in range(SYNCHRO_PASSES):
        for index in range(1, len(dynamic)):
            sensor_name, frame = dynamic[index]
            if frame.empty:
                continue

            time_ref = synchro_indices(data_m[0]["timestamp"], frame["timestamp"])
            non_zero = time_ref[time_ref != 0]
            if len(non_zero) == 0:
                data_m[0] = data_m[0].iloc[0:0].copy()
                data_m[index] = _empty_measure_frame(sensor_name)
                continue

            unique_ref = pd.unique(pd.Series(non_zero))
            data_m[0] = data_m[0].iloc[[int(value) - 1 for value in unique_ref]].reset_index(drop=True)
            time_ref_series = pd.Series(time_ref)
            keep_mask = time_ref_series.ne(0) & ~time_ref_series.duplicated()
            measures = pd.to_numeric(
                frame.loc[keep_mask.to_numpy(), sensor_name],
                errors="coerce",
            ).reset_index(drop=True)
            data_m[index] = pd.DataFrame(
                {
                    "timestamp": data_m[0]["timestamp"].reset_index(drop=True),
                    sensor_name: measures,
                }
            )

    data_measures = pd.DataFrame({"timestamp": data_m[0]["timestamp"].reset_index(drop=True)})
    if data_measures.empty:
        return pd.DataFrame(columns=["timestamp", *[sensor_name for sensor_name, _ in dynamic]])

    for index, (sensor_name, frame) in enumerate(dynamic):
        if frame.empty:
            # NaN signals "no data for this sensor" — callers use pd.isna() to detect it.
            data_measures[sensor_name] = float("nan")
            continue
        data_measures[sensor_name] = data_m[index].iloc[:, 1].reset_index(drop=True)

    return data_measures.reset_index(drop=True)


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
    reference = pd.to_datetime(reference_timestamps, utc=True).to_numpy(dtype="datetime64[ns]")
    target = pd.to_datetime(target_timestamps, utc=True).to_numpy(dtype="datetime64[ns]")

    if len(target) == 0:
        return np.zeros(0, dtype=int)
    if len(reference) == 0:
        return np.zeros(len(target), dtype=int)

    reference_order = np.argsort(reference, kind="mergesort")
    target_order = np.argsort(target, kind="mergesort")

    combined = np.concatenate([reference, target])
    markers = np.concatenate(
        [
            np.zeros(len(reference), dtype=int),
            np.arange(1, len(target) + 1, dtype=int),
        ]
    )
    combined_order = np.argsort(combined, kind="mergesort")
    ordered_markers = markers[combined_order]
    cumulative_reference = np.cumsum(ordered_markers == 0)

    reference_positions = cumulative_reference[ordered_markers != 0]
    original_order_positions = np.empty(len(target), dtype=int)
    original_order_positions[target_order] = reference_positions

    result = original_order_positions.copy()
    non_zero = original_order_positions != 0
    if non_zero.any():
        result[non_zero] = reference_order[original_order_positions[non_zero] - 1] + 1

    return result


def _empty_measure_frame(sensor_name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", sensor_name])
