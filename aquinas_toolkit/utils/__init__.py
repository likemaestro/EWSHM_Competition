"""Shared utilities (plotting, helpers) for the AQUINAS toolkit."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from aquinas_toolkit.utils.dataset_paths import (
    find_dataset_root,
    find_repo_root,
    find_workspace_root,
    list_dataset_dirs,
)

__all__ = [
    "plot_waveform",
    "plot_sensor_grid",
    "plot_sensor_overlay",
    "find_repo_root",
    "find_workspace_root",
    "find_dataset_root",
    "list_dataset_dirs",
]


def __getattr__(name: str) -> Any:
    """Load plotting helpers lazily to avoid importing matplotlib on CLI startup."""
    if name in __all__:
        plotting = import_module("aquinas_toolkit.utils.plotting")
        return getattr(plotting, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
