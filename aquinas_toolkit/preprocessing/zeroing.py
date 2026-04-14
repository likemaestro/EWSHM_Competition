"""Organizer-style baseline-removal helpers for AQUINAS waveforms."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from aquinas_toolkit.preprocessing.core import LoadedEventGroup


ZEROING_METHODS = {
    "none",
    "linear_endpoints",
}


def zero_waveform(values: pd.Series, *, method: str = "linear_endpoints") -> np.ndarray:
    """Return an organizer-style zeroed copy of one waveform series as a numpy array."""
    if method not in ZEROING_METHODS:
        raise ValueError(
            f"Unsupported zeroing method: {method}. Supported methods are {sorted(ZEROING_METHODS)}."
        )

    arr = values.to_numpy(dtype=float)
    if method == "none" or len(arr) == 0:
        return arr
    if len(arr) == 1:
        return arr - arr[0]

    baseline = np.linspace(arr[0], arr[-1], len(arr))
    return arr - baseline


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
        zeroed_waveforms[sensor_name] = (meta, zeroed)

    return replace(event_group, waveforms=zeroed_waveforms, zeroing_method=method)


