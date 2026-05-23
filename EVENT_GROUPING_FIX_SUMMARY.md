# Event Grouping Fix Summary

## What Was Found

The low NN input count came from exact `Start_Time + End_Time` grouping. Some
physically same vehicle passages have the same `Start_Time` across sensors but
slightly different `End_Time` values, so the old grouping split one passage into
multiple event groups before alignment.

Observed SET1 OLD example:

- 15 selected sensors started at the same time and ended at `02:43:19`.
- 1 selected sensor, `OLD_S1_UP_INF_STR`, had the same start but ended at
  `02:43:20`.
- Exact-window grouping produced two event groups, so neither group had complete
  selected sensor coverage for NN packaging.

## Why Alignment Did Not Fix It

Alignment runs after event grouping. Once exact-window grouping placed the
one-second-longer sensor into a different event group, `align_event_group()` only
received the sensors inside each already-split group. It could trim unequal
durations to common timestamps, but it could not recover sensors that were never
passed into the same event.

## How It Was Fixed

Preprocessing now supports `preprocessing.event_grouping.method`:

- `shared_start` is the default for preprocessing and NN runs.
- `exact_window` remains available for legacy comparisons and tests.

`shared_start` groups by `set + deck + Start_Time`, records the grouped event end
as the maximum grouped `End_Time`, and keeps each sensor row's original raw
start/end values in the event-sensor audit records. Alignment is unchanged: it
still receives the grouped records and trims them to the common timestamp
intersection.

## What Did Not Change

Mohsen's NN input design is unchanged:

- `strain_inputs.npy`: `(N_events, 200, 8)`
- `acc_inputs.npy`: `(N_events, frequency_bins, 8)`
- `temperature_inputs.npy`: `(N_events, 1)`
- no zero-fill
- no missing-sensor masks
- no `ACC_Y`
- no `SHE_STR`

## Validation

Previous exact-window SET1 OLD NN output:

- complete selected NN events: `1,352`
- `strain_inputs`: `(1352, 200, 8)`
- `acc_inputs`: `(1352, 578, 8)`
- `temperature_inputs`: `(1352, 1)`
- NaN counts: `0`

Fresh `shared_start` SET1 OLD run:

- run ID: `2026-05-23T11-49-59Z`
- preprocess retained events: `3,979`
- complete selected sensor coverage: `3,900`
- incomplete selected sensor coverage: `79`
- packaging-constraint exclusions: `1`
- retained NN events: `3,899`
- `strain_inputs`: `(3899, 200, 8)`
- `acc_inputs`: `(3899, 820, 8)`
- `temperature_inputs`: `(3899, 1)`
- `event_ids`: `(3899,)`
- NaN counts: `0` for strain, ACC, and temperature arrays

The ACC frequency-bin count changed from `578` to `820` because the existing
configuration uses `fft_length: max_retained_length`; after the grouping fix,
the retained complete event set includes longer aligned ACC records. The model
contract remains split ACC/strain/temperature with 8 channels each for strain
and ACC_Z.

Quicklooks generated for validation:

- `results/2026-05-23T11-49-59Z/stages/preprocess/nn_inputs/quicklook/event_0103.png`
- `results/2026-05-23T11-49-59Z/stages/preprocess/nn_inputs/quicklook/event_0697.png`
- `results/2026-05-23T11-49-59Z/stages/preprocess/nn_inputs/quicklook/event_3319.png`

## Verification Commands

```bash
python -m pytest tests/test_preprocessing.py tests/test_quicklook.py -q
ruff check aquinas_toolkit tests
python -m pytest -q
python -c "import sys; from aquinas_toolkit.cli import main; sys.argv=['aquinas','run','preprocess','--name','shared_start_event_grouping_verification']; main()"
```
