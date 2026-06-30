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

**Work in progress (WIP).** A WIP badge is shown in the viewer topbar
as a reminder that proxy metrics are in use and viewer integration with
the notebook-backed health-score narrative remains separate.

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

Current WIP boundary:

- The viewer uses proxy metrics derived from AQUINAS index-table fields such as
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

---

## 3D Viewer UI

### Overview

The viewer is a single-page application (`index.html` + `viewer.js` + `viewer.css`)
that loads the exported JSON artifacts and renders an interactive Three.js scene.
It has three tabs: **3D View**, **Sensor Analysis**, and **Datasets**.

### Tabs

#### 3D View

The main tab. A sidebar on the left contains controls; the right panel
contains the 3D canvas.

**Sidebar controls:**

| Control | Description |
|---------|-------------|
| **Family** segmented toggle | Filter visible sensors by measurement family: `ALL`, `ACC` (accelerometers), or `STR` (strain gauges). |
| **Compare** dropdown | `Single` shows one sensor at a time. `Homologous` highlights the counterpart sensor on the other deck. |
| **Show correlations** checkbox | Overlay correlation arcs between pairs of sensors selected in the active metric. |
| **Reset filters** button | Clear all filter checkboxes and show every sensor. |
| **Selection** panel | Shows the sensor ID, metric value, and mount location of the currently selected glyph. Click **Open analysis** to jump to the Sensor Analysis tab. |
| **Filters** collapsible | Narrow the visible set by Deck, Span, Side, Section, and Axis / Fibre. All groups are AND-combined; an empty group means "no filter on this dimension". |
| **Legend** collapsible | Colour key for accelerometer glyphs (red), strain glyphs (blue), high-value amber, and deck colours. |

**Topbar controls:**

| Control | Description |
|---------|-------------|
| **Dataset** dropdown | Switch between the available monthly AQUINAS sets. Metric values update immediately. |
| **Metric** dropdown | Choose which proxy metric to colour-scale: event count, mean range, mean absolute mean value, mean duration, or mean temperature. |
| **Reset view** button | Return the camera to its default position. |

**3D scene interactions:**

| Action | Effect |
|--------|--------|
| Left-drag | Orbit the camera around the bridge. |
| Scroll | Zoom in / out. |
| Right-drag | Pan the scene. |
| Click a sensor glyph | Pin the sensor. The Selection panel updates and the glyph scales up. |
| Click empty space | Deselect the current sensor. |
| Double-click a glyph | Isolate the local neighbourhood (same deck, span, and section). Double-click again on empty space to exit isolation. |
| Hover a glyph | Tooltip shows the sensor ID, metric value, unit, and deck. |

#### Sensor Analysis

Appears when a sensor is selected (or when you click **Open analysis** in the
Selection panel). Shows:

- Current metric value and unit
- Trend sparkline across all available datasets
- Homologous partner comparison
- Waveform preview (if `--include-waveforms` was used during `viz build`)
- Correlation table for the selected sensor and metric

#### Datasets

A grid of buttons for every available AQUINAS set in the run, with the
active set highlighted. Click any button to switch the active dataset
across the whole viewer.

---

### 3D Scene

#### Bridge geometry

The bridge is an analytical 45 m × 2-span box-girder model. Dimensions
are normalized to span length (1.0 = 45 m) and converted to world-space
metres by the `meters()` function at runtime.

Key cross-section parameters (`layout.py` constants → `bridge_geometry.json`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Span length | 45 m | One span; two spans total (90 m bridge) |
| Total depth | 2.0 m | Box girder full height |
| Top slab width | 7.5 m | Includes 1.75 m overhangs on each side |
| Web outer width at top | 4.7 m | Widest point of web zone (top) |
| Web outer width at bottom | 4.0 m | Narrowest point of web zone (bottom) |
| Bottom slab width | 4.7 m | Matches web outer top |
| Web thickness | 0.35 m | Constant through height |
| Slab thickness | 0.30 m | Top and bottom slabs |

The web profile tapers correctly: **wider at the top, narrower at the bottom**,
matching the trapezoidal box-girder cross-section in the AQUINAS handbook.
The inner void also tapers the same way (4.0 m at top → 3.3 m at bottom).

The `OLD` deck (tan/brown) and `NEW` deck (teal/glass) are separated by
a fixed transverse offset (compact layout). Their identity labels —
**OLD deck** and **NEW deck** — are painted as canvas textures flat on
the top slab surface, readable from the default camera angle.

Piers are rendered at the two abutments (x = 0 and x = 90 m) and the
mid-span pier (x = 45 m) under each deck.

#### Sensor glyphs

Each sensor is rendered as a 3D glyph placed on an **exterior surface** of
the deck, so it is always visible and raycaster-reachable:

| Sensor type | Glyph shape | Placement |
|-------------|-------------|-----------|
| `ACC_Z` (vertical acceleration) | Arrow pointing up | Outer face of bottom slab edge, pushed outward |
| `ACC_Y` (transverse acceleration) | Arrow pointing transversely | Same outer edge, nudged slightly along span from ACC_Z |
| `SUP_STR` (upper strain, superior fibre) | Double-headed horizontal bar | Top exterior of top slab, pushed upward |
| `INF_STR` (lower strain, inferior fibre) | Double-headed horizontal bar | Bottom exterior of bottom slab, pushed downward |
| `SHE_STR` (shear strain) | Double-headed horizontal bar | Outer face of web at mid-height |

Glyph colour encodes metric status:
- **Blue** — strain sensor, normal range
- **Red** — accelerometer, normal range
- **Amber** — any sensor with a high relative value for the selected metric
- **Dark navy** — currently selected sensor

A small invisible sphere (`SphereGeometry`) is added to each glyph as the
raycaster pick target to give a generous click area.

#### Coordinate system

```
X  →  span progression (0 m at abutment 1, 90 m at abutment 2)
Y  ↑  vertical elevation (0 at deck mid-height)
Z  ←→ upstream / downstream transverse axis
```

Sensor local positions (`local_position` in `sensor_layout.json`) are
in normalized span units. The `meters()` function converts them by
multiplying by `45.0` (the `meters_per_normalized_unit` field in
`bridge_geometry.json`).

---

### Source files

| File | Role |
|------|------|
| `viewer_assets/index.html` | HTML shell: topbar, tabbar, sidebar controls, canvas, panels |
| `viewer_assets/viewer.css` | All styling (Manrope font, panel layout, segmented toggles, legend, tooltip) |
| `viewer_assets/viewer.js` | Three.js scene: geometry build, glyph rendering, raycasting, UI wiring, analysis charts |
| `layout.py` | Pure Python: sensor coordinate derivation, bridge geometry constants, cross-section parameters |
| `exporter.py` | I/O layer: reads AQUINAS data, calls layout helpers, writes all JSON artifacts, copies viewer assets |

### JSON artifacts

| File | Contents |
|------|----------|
| `manifest.json` | Schema version, available datasets, default dataset, metric catalog, file paths |
| `bridge_geometry.json` | Cross-section parameters, deck mesh segments, pier anchors, span boundaries, label anchors, view mode deck centres |
| `sensor_layout.json` | One row per sensor with: parsed name fields, 3D exterior position, surface normal, glyph orientation, mount surface name, compact/exploded Z values, homologous partner ID |
| `sensor_metrics.json` | One row per (sensor, dataset, metric): value, unit, status band (`low` / `normal` / `high`) |
| `sensor_trends.json` | Per-sensor trend direction and slope across the selected datasets |
| `event_groups.json` | Deck-scoped event windows with optional waveform preview paths |
| `correlations.json` | Pairwise correlation rows for overlay rendering |

### Regenerating artifacts after code changes

If you modify `layout.py` (sensor positions, geometry constants), re-run
`aquinas viz build` which always runs the full export from scratch.
