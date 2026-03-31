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
reader  -->  preprocessing  -->  feature_extraction  -->  training  -->  scoring
(done)       (TODO)              (TODO)         (TODO)       (TODO)
```

- **Library code** lives in `aquinas_toolkit/`. Every reusable function
  goes here -- never define pipeline logic inside notebooks.
- **I/O** (`aquinas_toolkit/io/`) -- data loading via `AquinasReader`.
- **CLI** (`aquinas_toolkit/cli/`) -- `aquinas run preprocess`,
  `aquinas run features`, `aquinas run train`, `aquinas run score`,
  and `aquinas info`. Thin wrappers that call library code.
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
  `stages/<stage>/` directories. `results/latest.json` is only a
  convenience pointer to the active run.
- **Run IDs** use the readable UTC folder format
  `YYYY-MM-DDTHH-MM-SSZ` (for example `2026-03-31T21-45-00Z`).
- **Config source** -- in v1, new runs always snapshot
  `configs/default.yaml`; downstream stages must read the selected
  run's `config.yaml`, never the current workspace config.
- **CLI contract** -- `aquinas run [stage] [--name NAME] [--run-id ID]`.
  Use `--name` only when creating a new run. Use `--run-id` only for
  `features`, `train`, or `score`.
- **Config CLI scope** -- v1 does not expose `--config`; users edit
  `configs/default.yaml` before creating a new run.
- **Stage policy** -- stage prerequisites are enforced
  (`preprocess -> features -> train -> score`), and re-running a
  completed stage inside the same run is out of scope for v1.

## Key packages

| Package | Status | Purpose |
|---|---|---|
| `io/` | Done | `AquinasReader` -- load index tables and raw waveforms |
| `cli/` | In progress | Run lifecycle, metadata, latest-pointer resolution, and stage dispatch |
| `preprocessing/` | TODO | Filtering, normalisation, cross-sensor alignment |
| `feature_extraction/` | TODO | Time-domain and frequency-domain feature extraction |
| `training/` | TODO | Unsupervised anomaly/trend detection models |
| `utils/` | Done | Shared utilities such as plotting helpers |
| `scoring/` | TODO | Aggregate per-sensor scores into a global health score |

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
- **Keep metadata authoritative**: update `metadata.json` stage status
  as `not_started`, `running`, `completed`, or `failed`.
- **Keep writes atomic** for `metadata.json` and `latest.json` so an
  interrupted command does not leave a partially written file behind.
- **Do not add session folders, symlinks, or alternate history layers**
  unless the user explicitly asks for a design change. The run folders
  themselves are the history.

## Attribution

The data reader (`io/reader.py`) was originally written by **Zhenkun Li**
and migrated into `aquinas_toolkit` with minimal changes. When modifying
this file, preserve the attribution docstring at the top.
