"""Notebook-facing helpers for the preserved ACC_Z FDD workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from aquinas_toolkit.feature_extraction.fdd import (
    annotate_mode_shape_locations,
    frequency_domain_decomposition,
    summarize_fdd_mode_shapes,
)
from aquinas_toolkit.io import AquinasReader
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

    fdd_result = frequency_domain_decomposition(
        filtered.filtered_events,
        sampling_rate_hz=sampling_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    summary = summarize_fdd_results(
        fdd_result,
        channel_names=filtered.channel_names,
        frequency_band_hz=(low_hz, high_hz),
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
