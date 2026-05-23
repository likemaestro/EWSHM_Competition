# NN Inputs and Training Prep Implementation Summary

## What changed

- Replaced the old single flattened `neural_inputs.npy` contract with split
  event-level tensors under `results/<run_id>/stages/preprocess/nn_inputs/`.
- Added first-class preprocess quicklook inspection through:
  `aquinas preprocess quicklook`.
- Implemented `aquinas run train` as deterministic training-data preparation:
  split indices plus train-only normalization statistics.

## Preprocess artifacts

The preprocess stage now writes:

- `nn_inputs/strain_inputs.npy` with shape `(N_events, strain_samples, strain_channels)`
- `nn_inputs/acc_inputs.npy` with shape `(N_events, frequency_bins, acc_channels)`
- `nn_inputs/temperature_inputs.npy` with shape `(N_events, 1)`
- `nn_inputs/event_ids.npy` with shape `(N_events,)`

The first dimension is always the event axis. For any row index `i`,
`strain_inputs[i]`, `acc_inputs[i]`, `temperature_inputs[i]`, and
`event_ids[i]` describe the same event.

Current default NN sensor selection is:

- strain: `INF_STR` and `SUP_STR`
- acceleration: `ACC_Z`
- excluded from NN inputs: `SHE_STR` and `ACC_Y`

`preprocessing.acc.axis` is still configurable for tests and future
experiments, but the default competition NN contract remains `ACC_Z`.

## Metadata

The NN input directory keeps only the model-facing arrays and row IDs at the
top level. Metadata lives under `nn_inputs/metadata/`:

- `metadata/manifest.csv`
- `metadata/nn_inputs_manifest.json`
- `metadata/sensor_map.csv`
- `metadata/sensor_ids.json`
- `metadata/input_shapes.json`
- `metadata/frequency_bins.npy`
- `metadata/valid_lengths.npy`
- `metadata/temperature_metadata.csv`

The existing `report/` directory also receives compatibility copies of the
main metadata files for notebook and human inspection workflows.

## Quicklook

Use:

```bash
aquinas preprocess quicklook --run-id RUN_ID --event-index 42
```

Useful variants:

```bash
aquinas preprocess quicklook --summary
aquinas preprocess quicklook --sensor-map
aquinas preprocess quicklook --random 12
```

Plots are written by default under:

```text
results/<run_id>/stages/preprocess/nn_inputs/quicklook/
```

## Training preparation

`aquinas run train` now reads the split NN tensors and writes:

- `stages/train/splits/train_indices.npy`
- `stages/train/splits/val_indices.npy`
- `stages/train/splits/test_indices.npy`
- `stages/train/splits/split_manifest.json`
- `stages/train/normalization_stats.npz`

The default split is `70% / 20% / 10%`, using `training.random_seed`.
Normalization statistics are fit on training rows only and saved separately
for strain, ACC, and temperature inputs.

Full NN model architecture training is intentionally still deferred.
