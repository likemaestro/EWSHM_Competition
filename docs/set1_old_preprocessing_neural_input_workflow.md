# SET1 OLD Preprocessing and Neural Input Workflow

## Scope of This Note

This note documents the preprocessing workflow used for the current run:

- run folder: `results/2026-05-15T16-40-01Z`
- dataset: `AQUINAS_SET1_2022_07`
- deck: `OLD`

The active settings come from the snapped run config at
`results/2026-05-15T16-40-01Z/config.yaml`.

## Selected Sensors for This Run

This run uses only the `OLD` deck and only the following selected sensor groups:

- strain sensors at locations `INF`, `SHE`, and `SUP`
- acceleration sensors on axis `Z` only

This means:

- `ACC_Y` is excluded from this run
- the selected neural-input sensor set contains 20 channels total
  - 12 strain channels
  - 8 ACC_Z channels

## Event Definition

Events are grouped by exact:

- `deck`
- `Start_Time`
- `End_Time`

So one event is a shared event window across sensors, not one single sensor record.

## Preprocessing Pipeline Order

For each event, preprocessing follows this order:

1. load raw sensor waveforms for the event
2. filter
3. zeroing
4. alignment
5. save aligned outputs

The stage is run once per event. The whole preprocess run was executed once when
`aquinas run preprocess` created `results/2026-05-15T16-40-01Z`.

## Step-by-Step: Strain Sensors

For selected strain sensors, this is what happens:

1. The raw waveform slice is loaded from the dataset.
2. Filtering step:
   - method: `none`
   - strain signals are not band-pass filtered in this run.
3. Zeroing step:
   - method: `linear_endpoints`
   - a straight baseline connecting the first and last retained sample is subtracted.
4. Alignment step:
   - the strain timestamps are aligned together with the other active sensors in the event.
   - method: `r_synchro`
   - no interpolation is used.
5. Save step:
   - the aligned strain values are written into the event waveform output.

## Step-by-Step: ACC_Z Sensors

For selected acceleration sensors, this is what happens:

1. The raw waveform slice is loaded from the dataset.
2. Sensor selection rule:
   - only `ACC_Z` enters this run.
   - `ACC_Y` is not processed here.
3. Filtering step:
   - method: `butterworth_bandpass`
   - low cutoff: `0.5 Hz`
   - high cutoff: `20.0 Hz`
   - order: `4`
4. Zeroing step:
   - method: `linear_endpoints`
   - baseline removal is applied after filtering.
5. Alignment step:
   - the filtered and zeroed ACC_Z signals are aligned with the active event sensors.
   - method: `r_synchro`
   - no interpolation is used.
6. Minimum aligned-length rule:
   - `min_aligned_samples = 500`
   - short ACC_Z records below this threshold are excluded from neural-input packaging.
7. Save step:
   - the aligned ACC_Z values are written into the event waveform output.

## What `r_synchro` Means

`r_synchro` is the organizer-faithful timestamp alignment method used in preprocessing.

Its job is to make the retained sensors for one event share a common aligned timestamp grid.

It does this by:

1. taking the first active sensor as the reference timeline
2. comparing the other sensors against that reference
3. trimming to shared timestamps
4. repeating that trimming process one more time

So `r_synchro` is called once per event, but it performs **two internal shrinking passes**.

That is what "two-pass alignment" means:

- alignment is not run twice as separate preprocess stages
- the alignment helper itself makes two internal passes over the timestamp matching

## What Is Saved Per Event

For each retained aligned event, the run stores per-event waveform artifacts.

Each event has:

- one `.npy` file containing the aligned waveform matrix
- one `.meta.json` file containing metadata

The `.meta.json` records:

- `event_id`
- `sensor_names`
- `timestamps_utc`

The `.npy` stores the aligned values.

For row `i`:

- `timestamps_utc[i]` is the timestamp for that row
- `.npy[i, j]` is the value of sensor `sensor_names[j]` at that timestamp

## Current Run Counts

For this `SET1 / OLD` run:

- retained preprocess events: `6681`
- preprocess discarded events: `0`
- events with complete selected-sensor coverage: `1352`
- events excluded from neural packaging because of incomplete selected-sensor coverage: `5329`

## Canonical Neural-Input Event ID File

The canonical event ID artifact for the current run is:

- `results/2026-05-15T16-40-01Z/stages/preprocess/report/event_ids.npy`

That file contains the retained neural-input `event_id` values in the same row order as
`neural_inputs.npy`.

## Final Note for Neural Network Usage

Events that have all `20` selected sensors available are the ones used for the neural network input packaging in this run.

In this run, packaging exclusions are `0`, so the `1352` events with complete selected-sensor coverage are exactly the events listed in:

- `results/2026-05-15T16-40-01Z/stages/preprocess/report/event_ids.npy`
