# training/

## Purpose

Prepare reproducible training data artifacts for the neural-network experiments.
The challenge remains unsupervised: no labels are introduced here.

## Status

Partially implemented. `aquinas run train` currently performs deterministic
data preparation only; full NN architecture training is still a later step.

## Current Interface

- **Input:** split event tensors from
  `results/<run_id>/stages/preprocess/nn_inputs/`, with sensor and frequency
  metadata from `results/<run_id>/stages/preprocess/nn_inputs/metadata/`
- **Output:** train/validation/test index arrays and train-only
  normalization statistics under `results/<run_id>/stages/train/`

The stage writes:

- `splits/train_indices.npy`
- `splits/val_indices.npy`
- `splits/test_indices.npy`
- `splits/split_manifest.json`
- `normalization_stats.npz`

The split indices are applied to every input array so row `i` stays the same
event across strain, ACC, temperature, and event-id artifacts.

## Deferred Model Work

The later model-training implementation can consume the prepared artifacts for:

- attention-based architectures
- latent-space plus temperature-informed architectures
- reconstruction-error metrics such as MSE and MAE
- latent-space visualization and downstream health-score design

## Organizer Notes Relevant To Future OMA Work

- The organizer stated that Operational Modal Analysis can still work on
  short traffic records when they are baseline-corrected and concatenated
  into a longer pseudo-continuous signal.
- This is forward-looking guidance only. It does not force OMA as the
  competition method, and it does not change the v1 preprocessing scope.
- The preprocess stage preserves event-clean aligned outputs so later
  experiments can concatenate them if OMA becomes a selected path.
- The organizer also noted that bridge frequencies for this type of
  structure are typically around 2-10 Hz, which supports the current
  simple synchronization strategy for v1 preprocessing.
- Methodology reference supplied by the organizer:
  `10.1016/j.prostr.2024.09.248`.
