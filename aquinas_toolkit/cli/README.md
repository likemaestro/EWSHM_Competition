# cli/

## Purpose

Command-line interface for running the analysis pipeline and
inspecting the dataset. Registered as the `aquinas` console
script in `pyproject.toml`.

## Commands

| Command | What it does |
|---|---|
| `aquinas run [--name NAME]` | Create a new run and execute the full pipeline, then refresh the visualization bundle when dataset inputs are available |
| `aquinas run preprocess [--name NAME]` | Create a new run and execute only preprocessing, then refresh the visualization bundle when dataset inputs are available |
| `aquinas run features [--run-id ID]` | Run feature extraction in an existing run, using `latest.json` if `--run-id` is omitted, then refresh the visualization bundle |
| `aquinas run train [--run-id ID]` | Prepare deterministic train/validation/test split indices and normalization stats for NN inputs, then refresh the visualization bundle |
| `aquinas run score [--run-id ID]` | Run health score computation in an existing run, using `latest.json` if `--run-id` is omitted, then refresh the visualization bundle |
| `aquinas info` | Show dataset summary (sensors, events, date ranges) |
| `aquinas data fetch [--force] [--assume-yes] [--keep-zip]` | Download static dataset archive with Rich progress, verify SHA256, and extract to `data.dataset_root` |
| `aquinas data status` | Show a human-readable summary of dataset readiness |
| `aquinas data verify` | Strictly validate that the configured dataset root is complete |
| `aquinas data path` | Print the resolved dataset root path |
| `aquinas viz build [--run-id ID]` | Explicitly rebuild the offline visualization bundle for a run |
| `aquinas viz open [--run-id ID] [--host HOST] [--port PORT]` | Serve the visualization bundle over local HTTP and open it in the default browser |
| `aquinas preprocess quicklook [--run-id ID]` | Inspect split NN input tensors and write quicklook plots for preprocess artifacts |
| `aquinas --about` / `aquinas about` | Show toolkit metadata and maintainers |
| `aquinas --version` / `aquinas version` | Show installed CLI version |

## Run storage

- Each new run snapshots `configs/default.yaml` into `results/<run_id>/config.yaml`.
- `results/latest.json` is only a convenience pointer to the active run.
- Downstream stages always use the selected run's `config.yaml`, not the current workspace config.
- When dataset inputs are present locally, `aquinas run ...` refreshes
  `results/<run_id>/visualization/` automatically.
- `aquinas viz open` serves the bundle over local HTTP and keeps running
  until interrupted so the browser can load JSON artifacts without
  `file://` CORS problems.
- Dataset availability means all configured `data.sets` directories exist
  under `data.dataset_root`.
- `aquinas run` and `aquinas run preprocess` check dataset availability
  before creating a new run.
- If dataset folders are missing, `aquinas info`, `aquinas run`, and
  `aquinas run preprocess` can bootstrap data via `aquinas data fetch`
  in interactive terminals.
- In non-interactive terminals, missing or incomplete dataset inputs fail
  before new-run creation and point users to `aquinas data fetch` or
  `aquinas data fetch --force`.
- A placeholder dataset root containing only stub files such as
  `README.md` or `.gitkeep` is treated as an empty bootstrap destination,
  not as a destructive overwrite/repair case.

## Progress reporting

- `aquinas data fetch` shows archive download progress with transferred
  bytes, download speed, elapsed time, and ETA when the server exposes a
  total size.
- `aquinas data status` is intended for human inspection.
- `aquinas data verify` is intended for strict readiness checks and scripting.
- `aquinas data path` prints only the resolved dataset root path.
- `aquinas run` prints pipeline checkpoint lines after each completed stage
  (`1/4 completed`, `2/4 completed`, ...) while stage-level progress remains visible.
- Stage implementations own their inner progress details (for example,
  preprocess set/event progress and features extraction/modal/write phases).
- `aquinas run <stage>` (single-stage invocations) shows only that stage's
  inner progress. Pipeline checkpoint lines are not shown.
- Visualization refresh remains a post-run action and is not counted as a
  pipeline stage total.
- `aquinas run --verbose` prints detailed stage timing summaries to the console.
- Every run writes `results/<run_id>/debug.log` with lifecycle entries, timing
  lines, and failure traces (even when `--verbose` is not used).

## Structure

- `main.py` -- top-level dispatcher
- `run.py` -- pipeline execution and run lifecycle management
- `info.py` -- dataset inspection
- `data.py` -- dataset bootstrap (`fetch`) command
- `viz.py` -- visualization build and local serving commands
- `terminal.py` -- shared Rich terminal rendering helpers and theme
