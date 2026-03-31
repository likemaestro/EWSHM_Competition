# utils/

## Purpose

Shared utility functions used across the toolkit and notebooks.
This package currently provides:
- Reusable plotting helpers for AQUINAS waveform exploration and quick visual diagnostics
- Run-management helpers used by the CLI to create, resolve, and track pipeline runs

## Status

Partially complete. Plotting and run-management helpers are implemented;
additional utility modules can be added as the pipeline grows.

## Key functions

| Function | What it does |
|---|---|
| `plot_waveform(waveform, ...)` | Plot a single waveform with consistent axis labels and style |
| `plot_sensor_grid(reader, event_idx=..., ...)` | Plot the same event across all sensors in a subplot grid |
| `plot_sensor_overlay(reader, event_idx=..., ...)` | Overlay all available sensor waveforms for one event |
| `create_run(name=...)` | Create `results/<run_id>/`, snapshot `config.yaml`, initialize `metadata.json`, and update `results/latest.json` |
| `resolve_run(run_id=...)` | Resolve an existing run explicitly or from `results/latest.json` |
| `validate_stage_can_run(run_dir, stage)` | Enforce stage order and status prerequisites before stage execution |
| `mark_stage_started/completed/failed(...)` | Persist stage lifecycle transitions in `metadata.json` |

Run-management implementation is in `aquinas_toolkit/utils/run_management.py` and is consumed by `aquinas run`.

## Public import

```python
from aquinas_toolkit import plot_waveform, plot_sensor_grid, plot_sensor_overlay
# or
from aquinas_toolkit.utils import plot_waveform, plot_sensor_grid, plot_sensor_overlay
```

Run-management helpers are internal-to-CLI utilities and are imported directly from:

```python
from aquinas_toolkit.utils.run_management import create_run, resolve_run
```

## Attribution

Plotting implementation originally by Zhenkun Li, adapted into
reusable helpers from notebook exploration code.
