"""
Reusable plotting helpers for AQUINAS waveform data.

Original implementation by Zhenkun Li.
Adapted into reusable helpers from notebook 01_sensor_overview.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import pandas as pd
    from aquinas_toolkit.io.reader import AquinasReader


def plot_waveform(
    waveform: pd.DataFrame,
    title: str = "",
    ylabel: str = "Response",
    figsize: tuple[int, int] = (12, 4),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a single sensor waveform."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(waveform.iloc[:, 1])
    ax.set_title(title)
    ax.set_xlabel("Sample (100 Hz)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return ax


def plot_sensor_grid(
    reader: AquinasReader,
    event_idx: int = 0,
    ncols: int = 4,
    figsize_per_cell: tuple[int, int] = (5, 3),
) -> plt.Figure:
    """Plot the same event across all sensors in a grid."""
    sensors = reader.list_sensor_names()
    nrows = math.ceil(len(sensors) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, sensor in enumerate(sensors):
        ax = axes_flat[i]
        try:
            idx_df = reader.load_index_table(sensor)
            if event_idx >= len(idx_df):
                ax.set_title(f"{sensor}\nNo event {event_idx}")
                ax.axis("off")
                continue
            _, wf = reader.read_record(sensor_name=sensor, row_index=event_idx)
            ax.plot(wf.iloc[:, 1])
            ax.set_title(sensor, fontsize=8)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Response")
            ax.grid(True, alpha=0.3)
        except Exception:
            ax.set_title(f"{sensor}\nError")
            ax.axis("off")

    for j in range(len(sensors), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig


def plot_sensor_overlay(
    reader: AquinasReader,
    event_idx: int = 0,
    figsize: tuple[int, int] = (14, 6),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay all sensor waveforms for one event on a single plot."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sensors = reader.list_sensor_names()
    for sensor in sensors:
        try:
            idx_df = reader.load_index_table(sensor)
            if event_idx >= len(idx_df):
                continue
            _, wf = reader.read_record(sensor_name=sensor, row_index=event_idx)
            ax.plot(wf.iloc[:, 1], label=sensor)
        except Exception:
            pass

    ax.set_xlabel("Sample (100 Hz)")
    ax.set_ylabel("Response")
    ax.set_title(f"All sensors -- event {event_idx}")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)
    return ax
