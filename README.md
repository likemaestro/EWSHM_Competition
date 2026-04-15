<p align="center">
  <img src="docs/logo.png" alt="Aquinas Toolkit" width="100%"/>
</p>

<h1 align="center">AQUINAS Toolkit — EWSHM 2026 Challenge</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-%E2%89%A53.11-blue" alt="Python ≥3.11">
  <img src="https://img.shields.io/badge/version-0.1.0-orange" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/license-private-lightgrey" alt="License">
</p>

<p align="center">
  Offline, unsupervised structural health monitoring for the AQUINAS viaduct dataset.<br/>
  Built for Challenge 1 of the <a href="https://www.ewshm2026.com/">13th European Workshop on Structural Health Monitoring (EWSHM 2026)</a>, sponsored by <a href="https://www.osmos-group.com/">OSMOS Group</a>.<br/>
</p>

<p align="center">
  <b>48 sensors (24 ACC + 24 STR)</b> · <b>100 Hz trigger-based records</b> · <b>5 monthly datasets</b> · <b>offline batch pipeline</b> · <b>reproducible run artifacts</b>
</p>

<p align="center">
  <a href="#the-challenge">The challenge</a> ·
  <a href="#dataset">Dataset</a> ·
  <a href="#repository-structure">Repository structure</a> ·
  <a href="#getting-started">Getting started</a> ·
  <a href="#release">Release</a> ·
  <a href="#current-status">Current status</a> ·
  <a href="#what-preprocessing-now-does">Preprocessing</a> ·
  <a href="#evaluation-criteria">Evaluation criteria</a> ·
  <a href="#timeline">Timeline</a>
</p>

<p align="center">
  By Amir Zare Beiranvand, Liv Breivik, Mohsen Rezvani Alile, Murat Güven, Tommaso Panigati, and Zhenkun Li
</p>

## At a glance

- Research competition entry for an unsupervised, data-driven structural health score built from raw bridge measurements under the EWSHM Challenge 1 constraints
- Python toolkit with implemented data access, preprocessing, feature-store generation, CLI workflows, and offline visualization for repeatable offline analysis
- Current emphasis is robust preprocessing and feature extraction; training and global scoring remain intentionally stubbed while the run layout under `results/` is already reproducible

## Release

Current milestone release: [`v0.1.0`](https://github.com/likemaestro/EWSHM_Competition/releases/tag/v0.1.0)

This release covers the implemented reader, preprocessing, feature extraction, CLI workflow, and offline visualization bundle. `training/` and `scoring/` remain stubs, so the repository is released as an early milestone rather than a complete end-to-end competition pipeline.

## The challenge

A prestressed concrete box-girder viaduct in France is monitored by
**48 sensors** (24 acceleration + 24 strain) sampling at **100 Hz**.
Recordings are trigger-based: each record captures a few seconds of
bridge response as a vehicle crosses.

The goal is to develop a **data-driven, unsupervised** algorithm that:

1. Processes the raw sensor data across all 48 channels
2. Detects trends, anomalies, or shifts in the structural behaviour
3. Produces a **synthetic health score** indicating whether the bridge's
   mechanical response is stable, improving, or degrading

No labels are provided. No numerical (FEM) models may be used.
The algorithm must run on a standard office computer.

## Dataset

The **AQUINAS Dataset** (Available QUantities INtended for Analysis and
Science) contains five monthly snapshots spanning two years:

| SET | Period | Folder |
|---|---|---|
| SET1 | July 2022 | `AQUINAS_SET1_2022_07` |
| SET2 | April 2023 | `AQUINAS_SET2_2023_04` |
| SET3 | August 2023 | `AQUINAS_SET3_2023_08` |
| SET4 | January 2024 | `AQUINAS_SET4_2024_01` |
| SET5 | June 2024 | `AQUINAS_SET5_2024_06` |

Each SET contains 48 JSON index tables and 48 sensor directories with
raw waveform files. See `AQUINAS_DATASET/README.md` and
`docs/Aquinas-Dataset-Handbook.pdf` for full details.

## Repository structure

```text
EWSHM_Competition/
│
├── aquinas_toolkit/          Core Python package
│   ├── io/                   Data I/O (AquinasReader)              [implemented]
│   ├── cli/                  CLI commands (aquinas run/info/viz)   [implemented]
│   ├── preprocessing/        Signal preprocessing                  [implemented]
│   ├── feature_extraction/   Feature extraction                    [implemented]
│   ├── training/             Unsupervised anomaly models           [stub]
│   ├── utils/                Shared utilities (plotting)           [implemented]
│   ├── scoring/              Health score synthesis                [stub]
│   └── visualization/        Offline 3D bridge viewer              [implemented]
│
├── AGENTS.md                 Instructions for coding agents
├── pyproject.toml            Package metadata and CLI entry point
├── tests/                    Pytest test suite
├── configs/                  Pipeline configuration (YAML)
│
├── notebooks/                Exploration & presentation
│   ├── 01_sensor_overview.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_anomaly_detection.ipynb
│   ├── 05_health_scoring.ipynb
│   └── misc/
│       └── A_temperature_correlations.ipynb
│
├── docs/                     Challenge rules & dataset handbook (PDFs)
├── results/                  Output figures and data (git-ignored)
└── AQUINAS_DATASET/          Raw data (git-ignored, user-supplied)
```
## Current status

| Area | Status | Notes |
|---|---|---|
| `aquinas_toolkit.io` | Implemented | `AquinasReader` loads index tables and raw waveforms |
| `aquinas_toolkit.utils` | Implemented | Plotting helpers available through the public API |
| `aquinas_toolkit.cli` | Implemented | Run lifecycle, metadata, resume, preprocess and features dispatch; train and score pending |
| `aquinas_toolkit.visualization` | Implemented | Offline bridge viewer with proxy metrics, trends, correlations, and waveform previews |
| `aquinas_toolkit.preprocessing` | Implemented | Band-pass filtering -> zeroing -> alignment pipeline with manifests and QC artifacts |
| `aquinas_toolkit.feature_extraction` | Implemented | FDD modal analysis plus per-sensor waveform statistics and SQLite feature storage |
| `aquinas_toolkit.training` | Stub | Unsupervised anomaly and trend detection |
| `aquinas_toolkit.scoring` | Stub | Global health score aggregation |

## Organizer-Driven Preprocessing Notes

The preprocessing stage now reflects organizer guidance shared on
April 9, 2026 through the `AQUINAS_Explorer.R` helper script and a
follow-up email from Francois-Baptiste Cartiaux:

- the `AQUINAS_Explorer.R` helper script shaped the current event
  lookup, synchronization, zeroing, and aligned-export API
- the April 9, 2026 email warned that one sensor became damaged between
  SET3 and SET4 and should be kept for SET1-SET3 but discarded for
  SET4-SET5
- the repository implements that email guidance as a config-driven
  exclusion for `OLD_S1_UP_SUP_STR` in `AQUINAS_SET4_2024_01` and
  `AQUINAS_SET5_2024_06`
- the issue is not that the late raw waveform files become zero; the
  key failure is that the TABLE metadata reports `Range = 0`
  throughout SET4/SET5 while the raw waveform still varies and its
  baseline shifts sharply

## What preprocessing now does

Pipeline order: **band-pass filter -> zeroing -> alignment**

- groups events by deck and exact event window (`Start_Time` / `End_Time`)
- queries organizer-style timestamp windows with strict containment
- applies a zero-phase Butterworth band-pass filter (default 0.5-20 Hz) to
  each raw waveform before any baseline or timing correction
- applies per-sensor linear-endpoint zeroing (baseline removal) after filtering
- aligns sensors with the organizer `Synchro()` workflow:
  first selected sensor as reference, two shrinking passes, no interpolation
- writes event manifests, sensor-record status tables, aligned exports,
  summary diagnostics, a damaged-sensor QC report, and a local
  Python-vs-R parity harness

Team-facing details and rationale live in:

- [configs/README.md](configs/README.md) for the canonical config
  glossary and where to verify that a run used a given setting
- [docs/README.md](docs/README.md) for the organizer-email record
- [aquinas_toolkit/preprocessing/README.md](aquinas_toolkit/preprocessing/README.md) for the exact preprocessing semantics, evidence, artifacts, API behavior such as `timestamp` containment and `sensor_pattern` matching, and the Python vs `AQUINAS_Explorer.R` adaptation notes
- [notebooks/README.md](notebooks/README.md) for notebook-specific example choices, including organizer-style substitute timestamps used in `02_preprocessing`
- [aquinas_toolkit/feature_extraction/README.md](aquinas_toolkit/feature_extraction/README.md) for the downstream constraint

## Getting started

### 1. Clone and set up

```bash
git clone https://github.com/likemaestro/EWSHM_Competition.git
cd EWSHM_Competition
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e .
```

This project targets Python 3.11 or newer.

### 2. Bootstrap the dataset

Preferred:

```bash
aquinas data fetch
```

This downloads the static archive source, verifies SHA256, and extracts
to `AQUINAS_DATASET/` (or your configured `data.dataset_root`).

If the local dataset copy is corrupted and you want to replace it:

```bash
aquinas data fetch --force
```

Manual extraction is still supported. The expected folder structure is:

```text
AQUINAS_DATASET/
├── README.md
├── AQUINAS_SET1_2022_07/
├── AQUINAS_SET2_2023_04/
├── AQUINAS_SET3_2023_08/
├── AQUINAS_SET4_2024_01/
└── AQUINAS_SET5_2024_06/
```

### 3. Explore

```python
from aquinas_toolkit import AquinasReader, plot_waveform
from aquinas_toolkit.io import load_sensor_metadata

reader = AquinasReader("AQUINAS_DATASET/AQUINAS_SET1_2022_07")
print(reader.summary())

# Read a single event waveform
meta, waveform = reader.read_record("NEW_S1_DO_MID_ACC_Z", row_index=0)
plot_waveform(waveform, title="Single event preview")

# Or load metadata only from one or more readers
metadata = load_sensor_metadata(
    [reader],
    "NEW_S1_DO_MID_ACC_Z",
    columns=["File", "Start_Time", "End_Time", "Mean_Value", "Temperature"],
)
print(metadata.head())
```

Use `load_sensor_metadata(...)` for table-only analyses and notebooks.
Use `read_record(...)` when you need the raw waveform slice itself.

Or inspect the dataset summary from the CLI:

```bash
aquinas info
```

Then launch the notebooks:

```bash
jupyter lab notebooks/
```

Notebook layout:

- Top-level numbered notebooks (`01_` to `05_`) are the main project storyline.
- `notebooks/misc/` contains supporting analyses with alphabetical prefixes
  to distinguish them from the numbered main storyline.
- Current supporting notebook: `misc/A_temperature_correlations.ipynb`.

### 4. Run the pipeline

```bash
aquinas run                        # create a new run and execute the full pipeline
aquinas run --name baseline        # same, with an optional human-readable run name
aquinas run preprocess             # create a new run and run preprocessing only
aquinas run preprocess --name prep # same, with an optional human-readable run name
aquinas run features               # continue from results/latest.json
aquinas run train                  # continue from results/latest.json
aquinas run score                  # continue from results/latest.json
aquinas run features --run-id 2026-03-31T21-45-00Z  # resume an older run explicitly
aquinas viz build                  # explicitly rebuild the viewer for the active run
aquinas viz build --include-waveforms
aquinas viz open                   # serve the viewer locally and open it in the default browser
aquinas info                       # dataset summary
aquinas data fetch                 # download + verify + extract dataset archive
aquinas data status                # human-readable dataset readiness summary
aquinas data verify                # strict dataset completeness check
aquinas data path                  # print the resolved dataset root
aquinas about                      # toolkit metadata and maintainers
aquinas --version                  # installed CLI version
```

Each new run creates a readable UTC folder, snapshots the active config,
and updates the convenience pointer:

```text
results/
  latest.json
  2026-03-31T21-45-00Z/
    config.yaml
    metadata.json
    visualization/
    stages/
      preprocess/
      features/
      train/
      score/
```

Implemented behavior:

- Edit `configs/default.yaml` before creating a new run. v1 does not
  expose a separate `--config` flag.
- `--name` is optional and only applies when creating a new run.
- `aquinas run` and `aquinas run preprocess` validate dataset availability
  before creating a fresh run.
- `aquinas run features|train|score` use `--run-id` when provided,
  otherwise they resolve `results/latest.json`.
- New runs snapshot `configs/default.yaml` into
  `results/<run_id>/config.yaml`.
- Downstream stages always use the selected run's `config.yaml`, never
  the current workspace config.
- When the configured AQUINAS dataset tree is available locally,
  `aquinas run ...` refreshes `results/<run_id>/visualization/`
  automatically.
- Dataset availability means every configured `data.sets` folder exists
  under `data.dataset_root`, not merely that the root path exists.
- If dataset folders are missing, `aquinas info`, `aquinas run`, and
  `aquinas run preprocess` prompt to bootstrap with `aquinas data fetch`
  in interactive terminals before any new run is created.
- In non-interactive terminals, missing or incomplete dataset inputs
  fail clearly before run creation and point users to `aquinas data fetch`
  or `aquinas data fetch --force`.
- Stage prerequisites are enforced:
  `preprocess -> features -> train -> score`.
- Re-running a completed stage in the same run is intentionally not
  supported in v1; create a new run instead.

The metadata file records the run name, creation time, git state, and
per-stage status (`not_started`, `running`, `completed`, `failed`).

Current limitation:

- The `features` stage is fully wired and will execute FDD modal analysis and
  time-domain feature extraction. The `train` and `score` stages enforce stage
  order and update metadata but the corresponding algorithms are not yet wired
  into the CLI.

### 5. Rebuild or open the viewer

Each run now refreshes its visualization bundle automatically when you
use `aquinas run ...` and the configured dataset inputs are available
locally.

Use the explicit commands below only when you want to rebuild the bundle
manually, add waveform previews, or reopen the viewer server:

```bash
aquinas viz build
aquinas viz build --run-id 2026-03-31T21-45-00Z
aquinas viz build --set AQUINAS_SET2_2023_04 --set AQUINAS_SET5_2024_06
aquinas viz build --include-waveforms
aquinas viz open
aquinas viz open --port 8765
```

The build creates a static bundle under:

```text
results/
  <run_id>/
    visualization/
      index.html
      viewer.css
      viewer.js
      manifest.json
      bridge_geometry.json
      sensor_layout.json
      sensor_metrics.json
      sensor_trends.json
      event_groups.json
      correlations.json
      waveforms/               # only when --include-waveforms is used
```

What the viewer currently shows:

- Analytical 3D bridge geometry for the `OLD` and `NEW` decks rendered
  as a correct trapezoidal box-girder cross-section (wider at top,
  narrower at bottom), matching the AQUINAS handbook structural drawings
- `OLD deck` / `NEW deck` identity labels painted as canvas textures
  directly on the top slab surface of each deck
- Sensor glyphs placed on **exterior** structural surfaces so they are
  always visible and clickable:
  - `SUP_STR` - top of top slab (pushed upward)
  - `INF_STR` - bottom of bottom slab (pushed downward)
  - `ACC_Z / ACC_Y` - outer edge of bottom slab (pushed outward)
  - `SHE_STR` - outer face of web at mid-height
- A top-level `ALL | ACC | STR` family toggle
- Proxy metrics from AQUINAS index tables:
  event count, mean range, mean absolute mean value, mean duration,
  and mean temperature
- Sensor trends across the exported AQUINAS sets
- Homologous sensor comparisons and capped correlation overlays
- Deck-scoped event previews keyed by `dataset + deck + Start_Time + End_Time`

Current limitation / WIP:

- Until the scoring stages are implemented, the viewer uses metadata-derived
  proxy metrics rather than final structural health scores. A **WIP** badge
  is shown in the viewer topbar as a reminder.
- `aquinas viz open` serves the bundle over local HTTP and keeps the
  process running until you stop it with `Ctrl+C`. This avoids browser
  `file://` CORS restrictions when loading JSON artifacts.
- See `aquinas_toolkit/visualization/README.md` for full UI documentation.

## Timeline

| Date | Milestone |
|---|---|
| 2026-03-02 | Dataset released |
| 2026-04-01 | Deadline for submitting questions to OSMOS |
| 2026-07-01 | Two-page methodology + results summary due |
| 2026-07-09 | Presentation during plenary session |

## Evaluation criteria

| Weight | Criterion |
|---|---|
| 40% | Quality of scientific approach, presentation, and discussion |
| 40% | Quality of results and published code |
| 20% | Innovation and expected impact |

```text
 █████╗  ██████╗ ██╗   ██╗██╗███╗   ██╗ █████╗ ███████╗
██╔══██╗██╔═══██╗██║   ██║██║████╗  ██║██╔══██╗██╔════╝
███████║██║   ██║██║   ██║██║██╔██╗ ██║███████║███████╗
██╔══██║██║▄▄ ██║██║   ██║██║██║╚██╗██║██╔══██║╚════██║
██║  ██║╚██████╔╝╚██████╔╝██║██║ ╚████║██║  ██║███████║
╚═╝  ╚═╝ ╚══▀▀═╝  ╚═════╝ ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
                                        ████████╗ ██████╗  ██████╗ ██╗     ██╗  ██╗██╗████████╗
                                        ╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██║ ██╔╝██║╚══██╔══╝
                                           ██║   ██║   ██║██║   ██║██║     █████╔╝ ██║   ██║
                                           ██║   ██║   ██║██║   ██║██║     ██╔═██╗ ██║   ██║
                                           ██║   ╚██████╔╝╚██████╔╝███████╗██║  ██╗██║   ██║
                                           ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝   ╚═╝
```
