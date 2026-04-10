"""Organizer-style baseline-removal helpers for AQUINAS waveforms."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from aquinas_toolkit.preprocessing.alignment import AlignedEvent
from aquinas_toolkit.preprocessing.core import LoadedEventGroup


ZEROING_METHODS = {
    "none",
    "linear_endpoints",
}


def zero_waveform(values: pd.Series, *, method: str = "linear_endpoints") -> pd.Series:
    """Return an organizer-style zeroed copy of one waveform series."""
    if method not in ZEROING_METHODS:
        raise ValueError(
            f"Unsupported zeroing method: {method}. Supported methods are {sorted(ZEROING_METHODS)}."
        )

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if method == "none" or numeric.empty:
        return numeric
    if len(numeric) == 1:
        # R's formula produces NaN (0/0) for a single sample; returning 0 is more
        # robust and has the same semantic intent (zero-relative baseline removal).
        return numeric - float(numeric.iloc[0])

    values_array = numeric.to_numpy(dtype=float)
    baseline = np.linspace(values_array[0], values_array[-1], len(values_array))
    return pd.Series(values_array - baseline, index=numeric.index, name=numeric.name)


def zero_loaded_event_group(
    event_group: LoadedEventGroup,
    *,
    method: str = "linear_endpoints",
) -> LoadedEventGroup:
    """Apply organizer-style zeroing to each loaded sensor slice before alignment."""
    zeroed_waveforms: dict[str, tuple[pd.Series, pd.DataFrame]] = {}
    for sensor_name, (meta, waveform) in event_group.waveforms.items():
        zeroed = waveform.copy()
        zeroed[sensor_name] = zero_waveform(zeroed[sensor_name], method=method)
        zeroed_waveforms[sensor_name] = (meta.copy(), zeroed)

    return replace(event_group, waveforms=zeroed_waveforms, zeroing_method=method)


def zero_aligned_event(event: AlignedEvent, *, method: str = "linear_endpoints") -> AlignedEvent:
    """Apply baseline removal to each numeric sensor column in an aligned event."""
    aligned_waveform = event.aligned_waveform.copy()
    sensor_columns = [column for column in aligned_waveform.columns if column != "timestamp_utc"]
    for sensor_name in sensor_columns:
        aligned_waveform[sensor_name] = zero_waveform(aligned_waveform[sensor_name], method=method)
    return replace(event, aligned_waveform=aligned_waveform, zeroing_method=method)
