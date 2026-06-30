"""
AQUINAS Toolkit
===============
Data access, preprocessing, and analysis tools for the
EWSHM 2026 structural health monitoring competition.

The toolkit follows a pipeline architecture:

    reader  -->  preprocessing  -->  feature_extraction  -->  training  -->  scoring
    (done)       (done)              (done v1)              (done)          (done)

Quick start::

    from aquinas_toolkit import AquinasReader

    reader = AquinasReader("AQUINAS_DATASET/AQUINAS_SET1_2022_07")
    print(reader.summary())
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["AquinasReader", "plot_waveform", "plot_sensor_grid", "plot_sensor_overlay"]
__version__ = "0.2.0"


def __getattr__(name: str) -> Any:
    """Load public exports lazily so CLI startup avoids plotting imports."""
    if name == "AquinasReader":
        return import_module("aquinas_toolkit.io").AquinasReader

    if name in {"plot_waveform", "plot_sensor_grid", "plot_sensor_overlay"}:
        plotting = import_module("aquinas_toolkit.utils.plotting")
        return getattr(plotting, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
