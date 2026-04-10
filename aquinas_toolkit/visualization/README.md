# visualization/

## Purpose

Build an offline, analytical 3D bridge viewer for AQUINAS runs.
This package converts dataset metadata and run outputs into a stable
visualization schema, then packages a portable static viewer bundle.

The viewer is intended to answer three practical questions:

- Where is each sensor located on the bridge?
- How is that sensor behaving across the available monthly datasets?
- How do accelerometers and strain sensors compare across decks,
  spans, sections, and homologous sensor locations?

## Status

Implemented.

Current capabilities:

- Parse AQUINAS sensor names into a canonical spatial registry
- Enforce the released-layout rule that `ACC_Y` appears on `DO` only
- Build normalized bridge geometry for `OLD` and `NEW` decks, spans,
  piers, and section anchors
- Export a portable visualization bundle with:
  `manifest.json`, `bridge_geometry.json`, `sensor_layout.json`,
  `sensor_metrics.json`, `sensor_trends.json`, `event_groups.json`,
  `correlations.json`, and optional waveform previews
- Package a static offline viewer from
  `aquinas_toolkit/visualization/viewer_assets/`
- Drive the bundle from the CLI via `aquinas viz build`
- Refresh the bundle automatically from `aquinas run ...` when the run
  config points to a locally available AQUINAS dataset tree

Current limitation:

- Until the pipeline scoring stages are implemented, the viewer uses
  proxy metrics derived from AQUINAS index-table fields such as
  `Range`, `Mean_Value`, `Duration`, `Temperature`, and event count.
- Spatial semantics come from the AQUINAS dataset README and handbook,
  but the shear-section placement is still an analytical approximation:
  the handbook only says those sensors are located "near pier 1", so the
  viewer places `S1_SHE` and `S2_SHE` close to the pier rather than
  claiming an exact surveyed position.

## Interface

- **Primary command:** `aquinas viz build --run-id <id> [--set <set>] [--output <path>] [--include-waveforms]`
- **Open bundle:** `aquinas viz open --run-id <id> [--host <host>] [--port <port>]`
- **Automatic refresh:** `aquinas run ...` rebuilds the bundle for the
  resolved run when visualization inputs are available
- **Input:** a resolved run context, its snapshotted `config.yaml`,
  AQUINAS dataset folders, and optional waveform previews read through
  `AquinasReader`
- **Output:** a static visualization bundle under
  `results/<run_id>/visualization/`

Operational note:

- `aquinas viz open` serves the bundle over local HTTP instead of
  opening `index.html` directly from `file://`. This is required because
  the viewer loads JSON artifacts dynamically and browsers typically
  block those requests on the `file://` origin.
- `aquinas viz open` keeps the local server process running until you
  stop it with `Ctrl+C`.

Core Python entry points:

- `build_sensor_layout(...)` -- derive the canonical sensor registry
- `build_bridge_geometry()` -- generate normalized analytical geometry
- `build_visualization_artifacts(...)` -- export the full viewer bundle
