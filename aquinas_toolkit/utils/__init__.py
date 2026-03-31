"""Shared utilities (plotting, helpers) for the AQUINAS toolkit."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["plot_waveform", "plot_sensor_grid", "plot_sensor_overlay"]


def __getattr__(name: str) -> Any:
    """Load plotting helpers lazily to avoid importing matplotlib on CLI startup."""
    if name in __all__:
        plotting = import_module("aquinas_toolkit.utils.plotting")
        return getattr(plotting, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
