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
| `aquinas run train [--run-id ID]` | Run model training in an existing run, using `latest.json` if `--run-id` is omitted, then refresh the visualization bundle |
| `aquinas run score [--run-id ID]` | Run health score computation in an existing run, using `latest.json` if `--run-id` is omitted, then refresh the visualization bundle |
| `aquinas info` | Show dataset summary (sensors, events, date ranges) |
| `aquinas viz build [--run-id ID]` | Explicitly rebuild the offline visualization bundle for a run |
| `aquinas viz open [--run-id ID] [--host HOST] [--port PORT]` | Serve the visualization bundle over local HTTP and open it in the default browser |

## Run storage

- Each new run snapshots `configs/default.yaml` into `results/<run_id>/config.yaml`.
- `results/latest.json` is only a convenience pointer to the active run.
- Downstream stages always use the selected run's `config.yaml`, not the current workspace config.
- When dataset inputs are present locally, `aquinas run ...` refreshes
  `results/<run_id>/visualization/` automatically.
- `aquinas viz open` serves the bundle over local HTTP and keeps running
  until interrupted so the browser can load JSON artifacts without
  `file://` CORS problems.

## Structure

- `main.py` -- top-level dispatcher
- `run.py` -- pipeline execution and run lifecycle management
- `info.py` -- dataset inspection
- `viz.py` -- visualization build and local serving commands
- `terminal.py` -- shared Rich terminal rendering helpers and theme
