"""Organizer-style timestamp query helpers for preprocessing parity checks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aquinas_toolkit.io import AquinasReader
from aquinas_toolkit.preprocessing.alignment import organizer_align_sensor_frames
from aquinas_toolkit.preprocessing.core import load_timestamp_query_frames, parse_utc_timestamp
from aquinas_toolkit.preprocessing.zeroing import zero_waveform


@dataclass(frozen=True)
class OrganizerQueryResult:
    """Aligned organizer-style output for one timestamp query."""

    set_name: str
    deck: str
    sensor_pattern: str
    timestamp_utc: pd.Timestamp
    selected_sensors: list[str]
    data_measures: pd.DataFrame


def run_organizer_query(
    reader: AquinasReader,
    *,
    timestamp: str | pd.Timestamp,
    deck: str,
    sensor_pattern: str,
    records: pd.DataFrame | None = None,
    zeroing_method: str = "linear_endpoints",
) -> OrganizerQueryResult:
    """
    Reproduce the organizer helper's timestamp-driven selection workflow.

    This mirrors the R helper's inputs: one dataset set, one deck
    selector, one sensor selector, one timestamp, and an optional
    zeroing method.
    """
    frames = load_timestamp_query_frames(
        reader,
        timestamp=timestamp,
        deck=deck,
        sensor_pattern=sensor_pattern,
        records=records,
    )
    zeroed_frames: list[tuple[str, pd.DataFrame]] = []
    for sensor_name, frame in frames:
        if frame.empty:
            zeroed_frames.append((sensor_name, frame.copy()))
            continue
        zeroed = frame.copy()
        zeroed[sensor_name] = zero_waveform(zeroed[sensor_name], method=zeroing_method)
        zeroed_frames.append((sensor_name, zeroed))

    data_measures = organizer_align_sensor_frames(zeroed_frames)
    return OrganizerQueryResult(
        set_name=reader.set_name,
        deck=deck,
        sensor_pattern=sensor_pattern,
        timestamp_utc=parse_utc_timestamp(timestamp),
        selected_sensors=[sensor_name for sensor_name, _ in zeroed_frames],
        data_measures=data_measures,
    )
