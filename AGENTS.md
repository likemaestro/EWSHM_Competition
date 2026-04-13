# AGENTS.md -- Instructions for AI agents working on this repository

## What this project is

An entry for the EWSHM 2026 Challenge 1 (OSMOS Group): build an
unsupervised, data-driven algorithm that turns raw strain and
acceleration measurements from a French viaduct into a synthetic
structural health score.

This is a **research competition**, not a production system. Prioritise
clarity, correctness, and reproducibility over enterprise patterns.

## Hard constraints (from the challenge rules)

- **Unsupervised only** -- the dataset has no labels.
- **Data-driven only** -- no numerical (FEM) models of the structure.
- **Offline computation** -- batch processing over the whole dataset,
  not real-time streaming.
- **All 48 sensors** must be used and combined.
- **Standard office computer** -- no GPU requirement, reasonable RAM.

## Architecture

```
reader       -->  preprocessing  -->  feature_extraction  -->  training  -->  scoring
[implemented]     [implemented]       [partial]            [stub]       [stub]
```

- **Library code** lives in `aquinas_toolkit/`. Every reusable function
  goes here -- never define pipeline logic inside notebooks.
- **I/O** (`aquinas_toolkit/io/`) -- data loading via `AquinasReader`.
- **CLI** (`aquinas_toolkit/cli/`) -- `aquinas run preprocess`,
  `aquinas run features`, `aquinas run train`, `aquinas run score`,
  `aquinas info`, `aquinas viz build`, and `aquinas viz open`.
  Thin wrappers that call library code.
- **Pipeline stages** are subpackages: `preprocessing/`, `feature_extraction/`,
  `training/`, `scoring/`. Each has a `README.md` describing its purpose
  and expected interface.
- **Shared utilities** live in `aquinas_toolkit/utils/` (for example
  plotting helpers and run-management helpers used by notebooks and the
  public package API).
- **Notebooks** (`notebooks/`) are for exploration, visualisation, and
  jury presentation. They import from `aquinas_toolkit`.
- **Configs** (`configs/*.yaml`) hold all tunable parameters. Do not
  hardcode hyperparameters or file paths in library code.
- **Tests** (`tests/`) -- run with `pytest`.
- **Results** go to `results/` (git-ignored except `.gitkeep`).
- **Run layout** -- new pipeline runs live under `results/<run_id>/`
  with a snapshotted `config.yaml`, `metadata.json`, and lazy
  `stages/<stage>/` directories. When dataset inputs are available, the
  run command also refreshes `visualization/` for the same run.
  `results/latest.json` is only a convenience pointer to the active run.
- **Run IDs** use the readable UTC folder format
  `YYYY-MM-DDTHH-MM-SSZ` (for example `2026-03-31T21-45-00Z`).
- **Config source** -- in v1, new runs always snapshot
  `configs/default.yaml`; downstream stages must read the selected
  run's `config.yaml`, never the current workspace config.
- **CLI contract** -- `aquinas run [stage] [--name NAME] [--run-id ID]`.
  Use `--name` only when creating a new run. Use `--run-id` only for
  `features`, `train`, or `score`. Visualization commands live under
  `aquinas viz ...`.
- **Config CLI scope** -- v1 does not expose `--config`; users edit
  `configs/default.yaml` before creating a new run.
- **Stage policy** -- stage prerequisites are enforced
  (`preprocess -> features -> train -> score`), and re-running a
  completed stage inside the same run is out of scope for v1.

## Key packages

| Package | Status | Purpose |
|---|---|---|
| `io/` | Implemented | `AquinasReader` -- load index tables and raw waveforms |
| `cli/` | Implemented | Run lifecycle, metadata, latest-pointer resolution, and full stage dispatch (preprocess + features wired; train and score stubs) |
| `preprocessing/` | Implemented | Deck-aware event grouping, timestamp alignment, zeroing, and preprocess-stage artifacts |
| `feature_extraction/` | Partial | FDD modal analysis done (pipeline.py, fdd.py, workflow.py); time-domain features pending |
| `training/` | Stub | Unsupervised anomaly/trend detection models |
| `utils/` | Implemented | Shared utilities: plotting helpers (`plotting.py`) and run-management helpers (`run_management.py`) used by the CLI and notebooks |
| `scoring/` | Stub | Aggregate per-sensor scores into a global health score |

## Current preprocessing contract

- Group events per deck by exact `Start_Time` / `End_Time`.
- Synchronize sensors using organizer `Synchro()` alignment: the first
  selected sensor is the reference, two shrinking passes narrow to the
  common timestamp window; no interpolation in v1.
- Keep zeroing configurable; the current default is the organizer-shared
  endpoint-line subtraction (`linear_endpoints`).
- Read preprocessing behavior from the selected run's snapshotted
  `config.yaml`, not from hardcoded workspace assumptions.
- Keep preprocess logic in `aquinas_toolkit/preprocessing/`; notebooks
  should only consume the package API.
- `export.format` is active and supports `csv.gz` (default) and `csv`.

## Preprocessing public API

```python
from aquinas_toolkit.preprocessing import (
    AlignedEvent,
    LoadedEventGroup,
    OrganizerQueryResult,
    align_event_group,
    export_aligned_event,
    find_events,
    load_event_group,
    load_timestamp_query_frames,
    run_preprocessing,
    run_organizer_query,
    synchro_indices,
    zero_loaded_event_group,
    zero_waveform,
)
```

| Symbol | Kind | Purpose |
|---|---|---|
| `find_events()` | function | Group records by `set + deck + Start_Time + End_Time`; optional timestamp-containment and sensor-pattern filters |
| `load_event_group()` | function | Load all raw waveforms belonging to one grouped event |
| `load_timestamp_query_frames()` | function | Organizer-style deck/sensor selection for one timestamp query |
| `run_organizer_query()` | function | Return organizer-style aligned `DataMesures` output for one query |
| `align_event_group()` | function | Two-pass `Synchro()` alignment, first-selected reference, no interpolation |
| `synchro_indices()` | function | Low-level helper: compute the shared row indices from one synchronization pass |
| `zero_loaded_event_group()` | function | Apply baseline removal to each raw sensor slice before alignment |
| `zero_waveform()` | function | Apply a zeroing method to a single waveform array |
| `export_aligned_event()` | function | Export one aligned event as a CSV or CSV.GZ artifact |
| `run_preprocessing()` | function | Execute the full preprocess stage for a snapped pipeline run |
| `AlignedEvent` | dataclass | Output of `align_event_group()` |
| `LoadedEventGroup` | dataclass | Output of `load_event_group()` |
| `OrganizerQueryResult` | dataclass | Output of `run_organizer_query()` |

## Damaged sensor policy

François-Baptiste Cartiaux stated in an April 9, 2026 email that one
sensor was damaged between SET3 and SET4 but still emitted erroneous
data in SET4 and SET5.

- Current repo policy excludes `OLD_S1_UP_SUP_STR` only for
  `AQUINAS_SET4_2024_01` and `AQUINAS_SET5_2024_06`.
- The same sensor remains valid for SET1-SET3 and should not be treated
  as globally bad.
- Do not reintroduce that excluded channel downstream for SET4/SET5,
  either from raw waveforms or TABLE-derived features.
- Treat this as a documented data-integrity exception, not as a license
  to drop sensors arbitrarily. The overall methodology still needs to
  use the full sensor network across the dataset.
- See `docs/README.md` for the organizer record and
  `aquinas_toolkit/preprocessing/README.md` for the implemented
  evidence and artifact contract.

## Dataset facts (from the handbook)

- **48 sensors**: 24 acceleration (ACC) + 24 strain (STR)
- **Sampling rate**: 100 Hz, trigger-based (not continuous)
- **5 monthly datasets**: Jul 2022, Apr 2023, Aug 2023, Jan 2024, Jun 2024
- **Units**: acceleration in g (9.81 m/s^2), strain in permille (mm/m),
  temperature in deg C
- **Index tables**: 15 features per record (Record_UID, File, Start_Row,
  End_Row, Start_Time, End_Time, Duration, Start_Value, End_Value,
  Diff_Value, Min_Value, Max_Value, Mean_Value, Range, Temperature)
- **Record_UID is sensor-specific** -- to match events across sensors,
  use `Start_Time` and `End_Time`
- **Row numbering is 1-based** in Start_Row / End_Row
- Full reference: `AQUINAS_DATASET/README.md` and
  `docs/Aquinas-Dataset-Handbook.pdf`

## IO performance contract

Each sensor's raw data is stored in numbered sequential batch files named
`{SENSOR_NAME}_SET{N}_{NUMBER}.json`. Each file is roughly 8.8 MB and covers
between 2 and 226 index-table records. Without caching, the same file would
be read and re-parsed from JSON once per event that references it.

`AquinasReader.load_raw_file()` caches parsed DataFrames in memory, keyed on
`(sensor_name, raw_filename)`. Do not call `_load_json_file` directly; always
go through `load_raw_file`.

- The cache is scoped to the reader instance. `run_preprocessing()` creates a
  new `AquinasReader` per SET, so memory is released between sets.
- Any caller that needs to modify the returned DataFrame must call `.copy()`
  first. All existing internal callers (`_load_waveform_from_record`,
  `_load_waveform_slice`, `read_record`) already do this.

## Progress reporting

`run_preprocessing()` uses `rich.progress` with the shared `get_console()` so
`NO_COLOR` and the CLI theme are respected automatically. The display has four
phases per SET:

1. A persistent header line printed via `progress.console.print()` that stays
   in terminal history (e.g. `SET 1/5  AQUINAS_SET1_2022_07`).
2. An indeterminate spinner labeled `Reading sensor records...` while index
   tables are loaded and QC reports are built (typically ~30 seconds for
   SET1).
3. A single event progress bar (`Processing events  312/10738  3%  ...`) that
   fills as events are aligned and exported.
4. A persistent one-line summary after each set completes (e.g.
   `done  9,841 retained  897 discarded`).

There is no set-level progress bar. The persistent print calls provide the
history so completed sets remain visible while later sets run.

Progress output is suppressed in pytest because stdout is not a terminal.

## Code conventions

- **Python 3.11+**, type hints on public functions, docstrings on
  public classes and functions.
- **Formatting**: `ruff` with line length 100.
- **Imports**: `from aquinas_toolkit import AquinasReader` (not relative).
- **Plotting imports**: use `from aquinas_toolkit import plot_waveform`
  or `from aquinas_toolkit.utils.plotting import ...`, never
  `aquinas_toolkit.plotting`.
- **Editable install**: use `pip install -e .`.
- **Notebook naming**: `NN_descriptive_name.ipynb` with markdown headers
  between code cells.
- **No secrets or data in git**: the `AQUINAS_DATASET/` folder and
  `results/` outputs are git-ignored.

## Pipeline run conventions

- **Create new runs only from the front of the pipeline**: `aquinas run`
  and `aquinas run preprocess` always create a fresh run directory.
- **Resume later stages explicitly or via latest**:
  `features`, `train`, and `score` resolve the target run from
  `--run-id` or `results/latest.json`.
- **Refresh the viewer bundle from the run command**: when the dataset
  referenced by the run config is available locally, `aquinas run ...`
  refreshes `results/<run_id>/visualization/` automatically.
- **Keep metadata authoritative**: update `metadata.json` stage status
  as `not_started`, `running`, `completed`, or `failed`.
- **Keep writes atomic** for `metadata.json` and `latest.json` so an
  interrupted command does not leave a partially written file behind.
- **Serve the viewer over HTTP**: `aquinas viz open` runs a temporary
  local HTTP server for the bundle; do not assume the browser can load
  the viewer correctly from `file://`.
- **Do not add session folders, symlinks, or alternate history layers**
  unless the user explicitly asks for a design change. The run folders
  themselves are the history.

## Forward-looking notes (not yet implemented)

- **Temperature normalization** -- fiber-optic strain sensors are not
  hardware-compensated; temperature metadata is preserved but active
  normalization is deferred.
- **OMA on short records** -- the organizer confirmed that
  baseline-corrected short records can be concatenated into a
  pseudo-continuous signal for Operational Modal Analysis. This does not
  force OMA as the competition method; aligned preprocess exports are kept
  clean so the option stays open. Organizer methodology reference:
  DOI `10.1007/978-3-031-96106-9_22` (EVACES 2025 Volume 2).
- **Expected bridge frequencies** -- typically 2-10 Hz for this
  structure, which justifies the simple non-interpolating synchronization
  strategy in v1.

## Attribution

The data reader (`io/reader.py`) was originally written by **Zhenkun Li**
and migrated into `aquinas_toolkit` with minimal changes. When modifying
this file, preserve the attribution docstring at the top.
