"""
Notebook-facing helpers for the preserved ACC_Z FDD workflow.

Original implementation by Mohsen Rezvani Alile.
Adapted into reusable helpers from the feature-extraction notebook workflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aquinas_toolkit.feature_extraction.fdd import (
    annotate_mode_shape_locations,
    frequency_domain_decomposition,
    summarize_fdd_mode_shapes,
)
from aquinas_toolkit.io import AquinasReader, parse_sensor_name
from aquinas_toolkit.preprocessing import (
    bandpass_filter_waveform_matrix,
    find_common_sensor_events,
    load_common_event_waveform_matrix,
)


@dataclass(frozen=True)
class FilteredEventCollection:
    """Filtered multichannel events prepared for FDD."""

    common_events: pd.DataFrame
    selected_events: pd.DataFrame
    filtered_events: list[pd.DataFrame]
    channel_names: list[str]


@dataclass(frozen=True)
class PreprocessedEventCollection:
    """Aligned event matrices collected from a preprocess stage store."""

    available_events: pd.DataFrame
    selected_events: pd.DataFrame
    aligned_events: list[pd.DataFrame]
    channel_names: list[str]
    detail: str = ""


def collect_filtered_event_matrices(
    reader: AquinasReader,
    *,
    min_duration_seconds: float = 10.0,
    quantity: str | None = "ACC",
    axis: str | None = "Z",
    deck: str | None = None,
    max_events: int | None = 5,
    sampling_rate_hz: float = 100.0,
    low_hz: float = 0.5,
    high_hz: float = 20.0,
    filter_order: int = 4,
) -> FilteredEventCollection:
    """Collect filtered event matrices for Mohsen's notebook workflow."""
    common_events = find_common_sensor_events(
        reader,
        min_duration_seconds=min_duration_seconds,
        quantity=quantity,
        axis=axis,
        deck=deck,
    )
    if common_events.empty:
        return FilteredEventCollection(
            common_events=common_events,
            selected_events=common_events,
            filtered_events=[],
            channel_names=[],
        )

    selected_events = common_events if max_events is None else common_events.head(max_events).copy()
    filtered_events: list[pd.DataFrame] = []
    channel_names: list[str] = []

    for event in selected_events.itertuples(index=False):
        waveform_matrix = load_common_event_waveform_matrix(
            reader,
            start_time=event.Start_Time,
            end_time=event.End_Time,
            sensor_names=event.sensor_names,
        )
        if not channel_names:
            channel_names = [column for column in waveform_matrix.columns if column != "timestamp"]
        filtered_matrix = bandpass_filter_waveform_matrix(
            waveform_matrix,
            sampling_rate_hz=sampling_rate_hz,
            low_hz=low_hz,
            high_hz=high_hz,
            order=filter_order,
        )
        filtered_events.append(filtered_matrix)

    return FilteredEventCollection(
        common_events=common_events,
        selected_events=selected_events,
        filtered_events=filtered_events,
        channel_names=channel_names,
    )


def collect_preprocessed_event_matrices(
    preprocess_store: Any,
    *,
    set_name: str | None = None,
    deck: str | None = None,
    quantity: str = "ACC",
    axis: str = "Z",
    min_common_events: int = 2,
    max_events: int | None = 5,
) -> PreprocessedEventCollection:
    """Collect aligned event matrices from preprocess outputs for notebook and stage reuse."""
    retained_events = preprocess_store.iter_retained_events(set_name=set_name, deck=deck)
    if retained_events.empty:
        return PreprocessedEventCollection(
            available_events=retained_events.copy(),
            selected_events=retained_events.copy(),
            aligned_events=[],
            channel_names=[],
            detail="no retained events",
        )

    candidate_rows: list[dict[str, Any]] = []
    candidate_sensor_sets: list[set[str]] = []
    candidate_sensor_orders: dict[str, int] = {}

    for event in retained_events.itertuples(index=False):
        event_sensors = preprocess_store.load_event_sensors(event.event_id)
        event_sensors = event_sensors.loc[event_sensors["sensor_status"] == "included"].copy()
        matching: list[tuple[str, int]] = []
        for sensor in event_sensors.itertuples(index=False):
            metadata = parse_sensor_name(str(sensor.sensor_name))
            if metadata["quantity"] == quantity and metadata["axis"] == axis:
                matching.append((str(sensor.sensor_name), int(sensor.sensor_order)))
        if len(matching) < 2:
            continue
        sensor_names = {sensor_name for sensor_name, _ in matching}
        candidate_rows.append(
            {
                "event_id": str(event.event_id),
                "set_name": str(event.set_name),
                "deck": str(event.deck),
                "start_time_utc": event.start_time_utc,
                "end_time_utc": event.end_time_utc,
                "available_channel_count": len(sensor_names),
            }
        )
        candidate_sensor_sets.append(sensor_names)
        for sensor_name, sensor_order in matching:
            candidate_sensor_orders.setdefault(sensor_name, sensor_order)

    available_events = pd.DataFrame(candidate_rows)
    if available_events.empty:
        return PreprocessedEventCollection(
            available_events=available_events,
            selected_events=available_events.copy(),
            aligned_events=[],
            channel_names=[],
            detail=f"insufficient common {quantity}_{axis} events",
        )

    if max_events is None:
        selected_candidates = available_events.copy()
        selected_sensor_sets = candidate_sensor_sets
    else:
        selected_candidates = available_events.head(max_events).copy()
        selected_sensor_sets = candidate_sensor_sets[: len(selected_candidates.index)]

    if len(selected_candidates.index) < min_common_events:
        return PreprocessedEventCollection(
            available_events=available_events,
            selected_events=selected_candidates,
            aligned_events=[],
            channel_names=[],
            detail=f"insufficient common {quantity}_{axis} events",
        )

    common_channels = sorted(
        set.intersection(*selected_sensor_sets),
        key=lambda sensor_name: candidate_sensor_orders[sensor_name],
    )
    if len(common_channels) < 2:
        return PreprocessedEventCollection(
            available_events=available_events,
            selected_events=selected_candidates,
            aligned_events=[],
            channel_names=common_channels,
            detail=f"insufficient common {quantity}_{axis} channels",
        )

    aligned_events: list[pd.DataFrame] = []
    used_rows: list[dict[str, Any]] = []
    for event in selected_candidates.itertuples(index=False):
        aligned_event = preprocess_store.load_aligned_event(event.event_id, sensor_names=common_channels)
        if aligned_event.empty:
            continue
        numeric = aligned_event[common_channels].dropna(axis=0, how="any").reset_index(drop=True)
        if len(numeric.index) < 2:
            continue
        aligned_events.append(numeric)
        used_rows.append(
            {
                "event_id": str(event.event_id),
                "set_name": str(event.set_name),
                "deck": str(event.deck),
                "start_time_utc": event.start_time_utc,
                "end_time_utc": event.end_time_utc,
                "channel_count": len(common_channels),
                "row_count": int(len(numeric.index)),
            }
        )

    selected_events = pd.DataFrame(used_rows)
    if len(aligned_events) < min_common_events:
        return PreprocessedEventCollection(
            available_events=available_events,
            selected_events=selected_events,
            aligned_events=[],
            channel_names=common_channels,
            detail=f"insufficient {quantity}_{axis} events after dropping missing rows",
        )

    return PreprocessedEventCollection(
        available_events=available_events,
        selected_events=selected_events,
        aligned_events=aligned_events,
        channel_names=common_channels,
    )


def summarize_fdd_results(
    fdd_result: dict[str, object],
    *,
    channel_names: Sequence[str],
    frequency_band_hz: tuple[float, float] = (0.5, 20.0),
    n_peaks: int = 3,
) -> dict[str, pd.DataFrame | list[str] | object]:
    """Convert FDD outputs into notebook-friendly tables."""
    peak_table, mode_shape_table = summarize_fdd_mode_shapes(
        fdd_result["frequencies_hz"],
        fdd_result["singular_values"],
        fdd_result["mode_shapes"],
        channel_names=channel_names,
        frequency_band_hz=frequency_band_hz,
        n_peaks=n_peaks,
    )
    mode_shape_locations = annotate_mode_shape_locations(mode_shape_table)

    amplitude_table = _pivot_mode_shape_table(
        mode_shape_locations,
        peak_table,
        value_column="mode_shape_amplitude",
    )
    signed_component_table = _pivot_mode_shape_table(
        mode_shape_locations,
        peak_table,
        value_column="mode_shape_signed_component",
    )
    phase_table = _pivot_mode_shape_table(
        mode_shape_locations,
        peak_table,
        value_column="mode_shape_phase_deg",
    )

    return {
        "frequencies_hz": fdd_result["frequencies_hz"],
        "singular_values": fdd_result["singular_values"],
        "first_singular_value": fdd_result["singular_values"][:, 0],
        "spectral_density_matrices": fdd_result["spectral_density_matrices"],
        "peak_table": peak_table,
        "mode_shape_table": mode_shape_table,
        "mode_shape_locations": mode_shape_locations,
        "amplitude_table": amplitude_table,
        "signed_component_table": signed_component_table,
        "phase_table": phase_table,
        "channel_names": list(channel_names),
    }


def run_acc_z_fdd_workflow(
    reader: AquinasReader,
    *,
    min_duration_seconds: float = 10.0,
    deck: str | None = None,
    max_events: int | None = 5,
    sampling_rate_hz: float = 100.0,
    low_hz: float = 0.5,
    high_hz: float = 20.0,
    filter_order: int = 4,
    nperseg: int = 1024,
    noverlap: int = 512,
    n_peaks: int = 3,
) -> dict[str, object]:
    """Run the preserved ACC_Z filtered-event FDD workflow for one reader/deck."""
    filtered = collect_filtered_event_matrices(
        reader,
        min_duration_seconds=min_duration_seconds,
        quantity="ACC",
        axis="Z",
        deck=deck,
        max_events=max_events,
        sampling_rate_hz=sampling_rate_hz,
        low_hz=low_hz,
        high_hz=high_hz,
        filter_order=filter_order,
    )
    if not filtered.filtered_events:
        raise ValueError("No common filtered events were found for the requested workflow.")

    summary = run_acc_z_fdd_from_event_matrices(
        filtered.filtered_events,
        channel_names=filtered.channel_names,
        sampling_rate_hz=sampling_rate_hz,
        low_hz=low_hz,
        high_hz=high_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        n_peaks=n_peaks,
    )
    summary.update(
        {
            "dataset": reader.set_name,
            "deck": deck.upper() if deck is not None else "ALL",
            "common_events": filtered.common_events,
            "selected_events": filtered.selected_events,
            "filtered_events": filtered.filtered_events,
        }
    )
    return summary


def run_acc_z_fdd_from_preprocess_store(
    preprocess_store: Any,
    *,
    set_name: str | None = None,
    deck: str | None = None,
    min_common_events: int = 2,
    max_events: int | None = 5,
    sampling_rate_hz: float = 100.0,
    low_hz: float = 0.5,
    high_hz: float = 20.0,
    nperseg: int = 1024,
    noverlap: int = 512,
    n_peaks: int = 3,
) -> dict[str, object]:
    """Run the preserved ACC_Z FDD workflow from preprocess-stage artifacts."""
    collection = collect_preprocessed_event_matrices(
        preprocess_store,
        set_name=set_name,
        deck=deck,
        quantity="ACC",
        axis="Z",
        min_common_events=min_common_events,
        max_events=max_events,
    )
    if not collection.aligned_events:
        raise ValueError(collection.detail or "No common aligned ACC_Z events were found in preprocess outputs.")

    summary = run_acc_z_fdd_from_event_matrices(
        collection.aligned_events,
        channel_names=collection.channel_names,
        sampling_rate_hz=sampling_rate_hz,
        low_hz=low_hz,
        high_hz=high_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        n_peaks=n_peaks,
    )
    summary.update(
        {
            "dataset": set_name if set_name is not None else "ALL",
            "deck": deck.upper() if deck is not None else "ALL",
            "available_events": collection.available_events,
            "selected_events": collection.selected_events,
            "aligned_events": collection.aligned_events,
            "detail": collection.detail,
        }
    )
    return summary


def run_acc_z_fdd_from_event_matrices(
    waveform_matrices: Sequence[pd.DataFrame | np.ndarray],
    *,
    channel_names: Sequence[str],
    sampling_rate_hz: float = 100.0,
    low_hz: float = 0.5,
    high_hz: float = 20.0,
    nperseg: int = 1024,
    noverlap: int = 512,
    n_peaks: int = 3,
) -> dict[str, object]:
    """Run the notebook-preserved ACC_Z FDD summary on prepared waveform matrices."""
    fdd_result = frequency_domain_decomposition(
        waveform_matrices,
        sampling_rate_hz=sampling_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return summarize_fdd_results(
        fdd_result,
        channel_names=channel_names,
        frequency_band_hz=(low_hz, high_hz),
        n_peaks=n_peaks,
    )


def _pivot_mode_shape_table(
    mode_shape_locations: pd.DataFrame,
    peak_table: pd.DataFrame,
    *,
    value_column: str,
) -> pd.DataFrame:
    table = (
        mode_shape_locations.pivot(index="position_label", columns="peak_rank", values=value_column)
        .rename_axis(index="Location", columns="Peak rank")
        .sort_index()
    )
    table.columns = [
        f"Peak {peak_rank} ({float(peak_table.iloc[peak_rank - 1]['frequency_hz']):.2f} Hz)"
        for peak_rank in table.columns
    ]
    return table


