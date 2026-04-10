# EWSHM 2026 -- Structural Health Monitoring Challenge

```text
 █████╗  ██████╗ ██╗   ██╗██╗███╗   ██╗ █████╗ ███████╗    ████████╗ ██████╗  ██████╗ ██╗     ██╗  ██╗██╗████████╗
██╔══██╗██╔═══██╗██║   ██║██║████╗  ██║██╔══██╗██╔════╝    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██║ ██╔╝██║╚══██╔══╝
███████║██║   ██║██║   ██║██║██╔██╗ ██║███████║███████╗       ██║   ██║   ██║██║   ██║██║     █████╔╝ ██║   ██║
██╔══██║██║▄▄ ██║██║   ██║██║██║╚██╗██║██╔══██║╚════██║       ██║   ██║   ██║██║   ██║██║     ██╔═██╗ ██║   ██║
██║  ██║╚██████╔╝╚██████╔╝██║██║ ╚████║██║  ██║███████║       ██║   ╚██████╔╝╚██████╔╝███████╗██║  ██╗██║   ██║
╚═╝  ╚═╝ ╚══▀▀═╝  ╚═════╝ ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝   ╚═╝
```

Entry for **Challenge 1** of the [13th European Workshop on Structural Health Monitoring](https://www.ewshm2026.com/) (EWSHM 2026), sponsored by [OSMOS Group](https://www.osmos-group.com).

## Team

- Amir Zare Beiranvand
- Liv Breivik
- Mohsen Rezvani Alile
- Murat Guven
- Tommaso Panigati
- Zhenkun Li

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

```
EWSHM_Competition/
│
├── aquinas_toolkit/          Core Python package
│   ├── io/                   Data I/O (AquinasReader)          [implemented]
│   ├── cli/                  CLI commands (aquinas run/info)   [implemented]
│   ├── preprocessing/        Signal preprocessing              [implemented]
│   ├── feature_extraction/   Feature extraction                [stub]
│   ├── training/             Unsupervised anomaly models       [stub]
│   ├── utils/                Shared utilities (plotting)       [implemented]
│   └── scoring/              Health score synthesis            [stub]
│
├── AGENTS.md                 Instructions for coding agents
├── pyproject.toml            Package metadata and CLI entry point
├── tests/                    Pytest test suite
├── configs/                  Pipeline configuration (YAML)
│
├── notebooks/                Exploration & presentation
│   ├── 01_sensor_overview    Dataset exploration & waveform plots
│   ├── 02_preprocessing      Preprocessing demo                [implemented]
│   ├── 03_feature_extraction Feature extraction demo           [stub]
│   ├── 04_anomaly_detection  Unsupervised model demo           [stub]
│   └── 05_health_scoring     Final health score demo           [stub]
│
├── docs/                     Challenge rules & dataset handbook (PDFs)
├── results/                  Output figures and data (git-ignored)
└── AQUINAS_DATASET/          Raw data (git-ignored, user-supplied)
```

## Current status

| Area | Status | Notes |
|---|---|---|
| `aquinas_toolkit.io` | Implemented | `AquinasReader` loads index tables and raw waveforms |
| `aquinas_toolkit.utils` | Implemented | Plotting helpers are available through the public package API |
| `aquinas_toolkit.cli` | Implemented | Run lifecycle, metadata, resume behavior, and preprocess-stage dispatch are complete; feature/train/score registration pending |
| `aquinas_toolkit.preprocessing` | Implemented | Event grouping, timestamp alignment, zeroing, and preprocess-stage artifacts |
| `aquinas_toolkit.feature_extraction` | Stub | Time- and frequency-domain features |
| `aquinas_toolkit.training` | Stub | Unsupervised anomaly and trend detection |
| `aquinas_toolkit.scoring` | Stub | Global health score aggregation |

## Organizer-Driven Preprocessing Notes

The preprocessing stage now reflects organizer guidance shared on
April 9, 2026 through the `AQUINAS_Explorer.R` helper script and a
follow-up email from François-Baptiste Cartiaux:

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

- groups events by deck and exact event window
- queries organizer-style timestamp windows with strict containment
- aligns sensors with the organizer `Synchro()` workflow:
  first selected sensor, two shrinking passes, no interpolation
- applies per-sensor endpoint zeroing before alignment by default
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
git clone https://github.com/likeaestro/EWSHM_Competition.git
cd EWSHM_Competition
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e .
```

### 2. Place the dataset

Download the AQUINAS dataset and extract it so the folder structure is:

```
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

reader = AquinasReader("AQUINAS_DATASET/AQUINAS_SET1_2022_07")
print(reader.summary())

# Read a single event
meta, waveform = reader.read_record("NEW_S1_DO_MID_ACC_Z", row_index=0)
plot_waveform(waveform, title="Single event preview")
```

Or launch the notebooks:

```bash
jupyter lab notebooks/
```

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
aquinas info                       # dataset summary
```

Each new run creates a readable UTC folder, snapshots the active config,
and updates the convenience pointer:

```text
results/
  latest.json
  2026-03-31T21-45-00Z/
    config.yaml
    metadata.json
    stages/
      preprocess/
      features/
      train/
      score/
```

Behavior:

- Edit `configs/default.yaml` before creating a new run. v1 does not
  expose a separate `--config` flag.
- `--name` is optional and only applies when creating a new run.
- `aquinas run` and `aquinas run preprocess` always create a fresh run.
- `aquinas run features|train|score` use `--run-id` when provided,
  otherwise they resolve `results/latest.json`.
- New runs snapshot `configs/default.yaml` into
  `results/<run_id>/config.yaml`.
- Downstream stages always use the selected run's `config.yaml`, never
  the current workspace config.
- Stage prerequisites are enforced:
  `preprocess -> features -> train -> score`.
- Re-running a completed stage in the same run is intentionally not
  supported in v1; create a new run instead.

The metadata file records the run name, creation time, git state, and
per-stage status (`not_started`, `running`, `completed`, `failed`).

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
