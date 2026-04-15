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
[implemented]     [implemented]       [implemented]        [stub]       [stub]
```

- **Library code** lives in `aquinas_toolkit/`. Every reusable function
  goes here -- never define pipeline logic inside notebooks.
- **I/O** (`aquinas_toolkit/io/`) -- data loading via `AquinasReader`.
- **CLI** (`aquinas_toolkit/cli/`) -- `aquinas run preprocess`,
  `aquinas run features`, `aquinas run train`, `aquinas run score`,
  `aquinas info`, `aquinas data fetch`, `aquinas viz build`, and
  `aquinas viz open`.
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
- Dataset archive source metadata for `aquinas data fetch`
  (`share_url`, `sha256`) is intentionally static and lives in
  `aquinas_toolkit/dataset_source.py`, not in user config.
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
| `preprocessing/` | Implemented | Deck-aware event grouping, timestamp alignment, zeroing, bandpass filtering, SQLite storage, and preprocess-stage artifacts |
| `feature_extraction/` | Implemented | FDD modal analysis, per-sensor waveform statistics (mean, std, RMS, min, max, peak-to-peak, energy, crest factor, zero crossing rate, skewness, kurtosis), SQLite feature store |
| `training/` | Stub | Unsupervised anomaly/trend detection models |
| `utils/` | Implemented | Shared utilities: plotting helpers (`plotting.py`) and run-management helpers (`run_management.py`) used by the CLI and notebooks |
| `scoring/` | Stub | Aggregate per-sensor scores into a global health score |

## Current preprocessing contract

- Group events per deck by exact `Start_Time` / `End_Time`.
- Synchronize sensors using organizer `Synchro()` alignment: the first
  selected sensor is the reference, two shrinking passes narrow to the
  common timestamp window; no interpolation in v1.
- Alignment uses `np.searchsorted` for O(n log n) synchro index
  computation and works on numpy arrays internally to avoid DataFrame
  overhead in the inner loop.
- Keep zeroing configurable; the current default is the organizer-shared
  endpoint-line subtraction (`linear_endpoints`).
- Signal filtering is configurable via `signal_filter.method`:
  `butterworth_bandpass` (default, 0.5–20 Hz) or `none`.
- Read preprocessing behavior from the selected run's snapshotted
  `config.yaml`, not from hardcoded workspace assumptions.
- Keep preprocess logic in `aquinas_toolkit/preprocessing/`; notebooks
  should only consume the package API.
- **Primary storage is SQLite** (`preprocess.sqlite`), written via
  `PreprocessStoreWriter`. CSV exports are an optional secondary output
  controlled by `exports.aligned_waveforms.enabled`.
- `export.format` supports `csv.gz` (default) and `csv` for optional CSV
  exports.

## Preprocessing public API

```python
from aquinas_toolkit.preprocessing import (
    AlignedEvent,
    LegacyPreprocessCsvReader,
    LoadedEventGroup,
    OrganizerQueryResult,
    PreprocessStoreReader,
    SIGNAL_FILTER_METHODS,
    align_event_group,
    bandpass_filter_waveform_matrix,
    export_aligned_event,
    filter_loaded_event_group,
    filter_records_by_min_duration,
    find_common_sensor_events,
    find_events,
    load_common_event_waveform_matrix,
    load_event_group,
    load_timestamp_query_frames,
    open_preprocess_store,
    run_preprocessing,
    run_organizer_query,
    summarize_min_duration_filter,
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
| `filter_loaded_event_group()` | function | Apply signal filtering (e.g. bandpass) to each sensor in a loaded event group |
| `bandpass_filter_waveform_matrix()` | function | Apply bandpass filter to a multichannel waveform DataFrame or ndarray |
| `filter_records_by_min_duration()` | function | Filter sensor records by minimum event duration |
| `find_common_sensor_events()` | function | Find events present in every selected sensor after duration filtering |
| `load_common_event_waveform_matrix()` | function | Load one aligned multichannel event matrix |
| `summarize_min_duration_filter()` | function | Summarize the effect of a minimum duration filter |
| `export_aligned_event()` | function | Export one aligned event as a CSV or CSV.GZ artifact |
| `open_preprocess_store()` | function | Open a preprocess store (SQLite or legacy CSV) as a context manager |
| `run_preprocessing()` | function | Execute the full preprocess stage for a snapped pipeline run |
| `AlignedEvent` | dataclass | Output of `align_event_group()` |
| `LoadedEventGroup` | dataclass | Output of `load_event_group()` |
| `OrganizerQueryResult` | dataclass | Output of `run_organizer_query()` |
| `PreprocessStoreReader` | class | Read-only accessor for `preprocess.sqlite` |
| `LegacyPreprocessCsvReader` | class | Fallback reader for old CSV-only preprocess artifacts |
| `SIGNAL_FILTER_METHODS` | set | Supported signal filter methods: `{"none", "butterworth_bandpass"}` |

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

`AquinasReader` uses **orjson** for JSON loading (`_load_json_file` reads
bytes and calls `orjson.loads()`), which is significantly faster than
stdlib `json`.

There are two caching layers, both scoped to the reader instance:

1. **`load_raw_file()`** -- caches the parsed DataFrame in
   `self._raw_file_cache`, keyed on `(sensor_name, raw_filename)`.
2. **`load_raw_file_prepped()`** -- an additional cache in
   `self._prepped_cache` that pre-parses timestamps to
   `datetime64[ns, UTC]` and converts sensor values to numeric. This
   avoids repeated timestamp parsing across event slices (~240K calls
   reduced to ~2K calls).

Do not call `_load_json_file` directly; always go through `load_raw_file`
or `load_raw_file_prepped`.

- `run_preprocessing()` creates a new `AquinasReader` per SET, so memory
  is released between sets.
- Any caller that needs to modify the returned DataFrame must call `.copy()`
  first, or slice first then copy (`.iloc[start:end].to_numpy()`).
  All existing internal callers already do this.
- **Critical**: when slicing cached DataFrames, always slice first then
  convert (`.iloc[start:end].to_numpy()`), never convert then slice
  (`.to_numpy()[start:end]`). The latter copies the entire 100K+ row
  column before slicing, causing severe performance regression.

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

## Documentation conventions

- Preserve the project branding as **`AQUINAS Toolkit`** in user-facing
  documentation; do not silently switch the project name back to
  `Aquinas Toolkit`.
- Keep README links aligned to the actual repository owner and workflow
  paths. The canonical GitHub remote is
  `https://github.com/likemaestro/EWSHM_Competition`.
- When editing the README hero, prioritize accurate repo state over
  aspirational claims. Current public positioning should reflect:
  implemented data access, preprocessing, feature extraction, CLI
  workflow, offline visualization, and reproducible run artifacts; the
  `training/` and `scoring/` stages remain stubs.
- Keep README quick-access links pointed only at sections that actually
  exist in the current document.
- Avoid leaving broken status badges in the README. Remove or fix any
  badge whose target repository, workflow path, or visibility does not
  resolve correctly.

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
  structure, which justifies the 0.5-20 Hz default bandpass and the
  simple non-interpolating synchronization strategy in v1.

## Performance notes

- **Preprocessing throughput**: ~8.7 events/sec on a standard office
  machine (~16 min for 10,738 events across 5 SETs).
- **Threading**: Python's GIL prevents CPU parallelism for numpy/scipy
  workloads. The preprocessing pipeline uses serial event processing.
  `ThreadPoolExecutor` was tried and removed -- it provides zero speedup
  for CPU-bound numpy/scipy work.
- **Key optimizations**: `np.searchsorted`-based synchro alignment,
  numpy-internal alignment loop (no DataFrame overhead), pre-parsed
  timestamp/numeric caching in `load_raw_file_prepped`, vectorized
  store preparation, pre-computed Butterworth SOS coefficients shared
  across events.
- **Data integrity**: 14 dedicated edge-case tests
  (`TestSearchsortedSynchroFidelity`, `TestZeroDataLoss`) verify that
  no valid data is silently dropped through the load-filter-zero-align-sample
  pipeline.

## Attribution

The data reader (`io/reader.py`) was originally written by **Zhenkun Li**
and migrated into `aquinas_toolkit` with minimal changes. When modifying
this file, preserve the attribution docstring at the top.
