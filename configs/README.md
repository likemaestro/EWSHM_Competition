# configs/

Pipeline configuration files in YAML format.

## How it works

The v1 CLI reads `configs/default.yaml` as the active workspace config.
When you start a new run with `aquinas run` or `aquinas run preprocess`,
that file is copied into the run folder as `config.yaml`.

Downstream stages (`features`, `train`, `score`) always use the selected
run's snapshot instead of the current workspace config. This keeps resume
behavior deterministic even if `configs/default.yaml` changes later.

## Files

| File | Purpose |
|---|---|
| `default.yaml` | Active working configuration used when creating new runs |

## Working with variants

If you want to keep multiple config variants, store them next to
`default.yaml` and copy the desired variant over `configs/default.yaml`
before starting a new run.

## What goes in a config

- **Data** -- which AQUINAS SETs to include, path to dataset root
- **Preprocessing** -- filter settings, normalisation toggles
- **Features** -- which feature extractors to run
- **Model** -- model type and hyperparameters
- **Scoring** -- aggregation method, trend window
- **Output** -- where to write run folders and `latest.json`
