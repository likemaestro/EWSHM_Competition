# cli/

## Purpose

Command-line interface for running the analysis pipeline and
inspecting the dataset. Registered as the `aquinas` console
script in `pyproject.toml`.

## Commands

| Command | What it does |
|---|---|
| `aquinas run [--name NAME]` | Create a new run and execute the full pipeline |
| `aquinas run preprocess [--name NAME]` | Create a new run and execute only preprocessing |
| `aquinas run features [--run-id ID]` | Run feature extraction in an existing run, using `latest.json` if `--run-id` is omitted |
| `aquinas run train [--run-id ID]` | Run model training in an existing run, using `latest.json` if `--run-id` is omitted |
| `aquinas run score [--run-id ID]` | Run health score computation in an existing run, using `latest.json` if `--run-id` is omitted |
| `aquinas info` | Show dataset summary (sensors, events, date ranges) |

## Run storage

- Each new run snapshots `configs/default.yaml` into `results/<run_id>/config.yaml`.
- `results/latest.json` is only a convenience pointer to the active run.
- Downstream stages always use the selected run's `config.yaml`, not the current workspace config.

## Structure

- `main.py` -- top-level dispatcher
- `run.py` -- pipeline execution and run lifecycle management
- `info.py` -- dataset inspection
- `terminal.py` -- shared Rich terminal rendering helpers and theme
