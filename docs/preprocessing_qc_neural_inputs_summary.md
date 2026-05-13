# Preprocessing, QC, and Neural Input Work Summary

## Purpose

We revised the AQUINAS Toolkit preprocessing stage so its primary output is a neural-network-ready dataset for the EWSHM competition. The work keeps reusable logic in `aquinas_toolkit/` and leaves notebooks as thin experiment/inspection layers.

## Main Pipeline Changes

- Added neural input packaging under the preprocess stage.
- The preprocess stage now writes one canonical `neural_inputs.npy`.
- Verbose metadata and QC artifacts are written beside it in folders:
  - `report/` for tensor metadata such as sensor map, input slices, event IDs, frequency bins, and temperature metadata.
  - `qc_outputs/` for event QC, sensor QC, discarded events, QC summary, and flagged plots.
- Flexible preprocessing is preserved: technically valid partial-sensor events remain in the preprocess store, while `neural_inputs.npy` only uses events with complete selected-sensor coverage.
- Limited the current default dataset scope to `AQUINAS_SET1_2022_07` and the `OLD` deck.
- Excluded ACC-Y from neural preprocessing and retained only ACC-Z.

## Signal-Specific Preprocessing

- Replaced misleading global filter/zeroing intent with signal-specific preprocessing settings.
- Strain-type sensors (`INF_STR`, `SUP_STR`, `SHE_STR`):
  - no band-pass filtering
  - endpoint-line zeroing
  - peak-window clipping for neural input packaging
- ACC-Z sensors:
  - time-domain Butterworth band-pass filter at 0.5-20 Hz
  - endpoint-line zeroing after filtering
  - no clipping of high-amplitude but valid vehicle responses
  - zero-padding to the longest retained ACC-Z event before FFT
  - FFT magnitude conversion with retained 0.5-20 Hz frequency bins

## Strain Window Logic

- Replaced separate before/after peak settings with one symmetric variable:
  - `preprocessing.strain.peak_window_half_samples`
- Total strain window length is `2 * peak_window_half_samples`.
- Peaks near the start or end are no longer discarded if the full fixed-length window can fit somewhere in the aligned record.
- Edge windows are shifted inside signal bounds:
  - near start: window starts at sample 0
  - near end: window ends at the final sample
- `strain_window_out_of_bounds` now means the aligned strain record is shorter than the required fixed window length.

## QC Logic

- Added deterministic record-level QC before neural input packaging.
- QC expands each retained preprocess event across the selected neural sensor set. If a selected sensor has no record for a retained event, that row is marked `not_available_for_global_event`.
- `not_available_for_global_event` is an availability/coverage label, not evidence of a broken waveform or sensor malfunction.
- Hard discard reasons include:
  - `not_available_for_global_event`
  - `missing_row_range`
  - `invalid_row_range`
  - `waveform_load_failed`
  - `nan_values`
  - `timestamp_error`
  - `flat_signal`
  - `strain_window_out_of_bounds`
  - `acc_short_duration`
- MAD robust z-score is used only for warnings/reporting.
- MAD does not remove records.
- Added `qc.mad_used_for_removal: false` to make this explicit in config.
- A high record-level discard rate can therefore be caused by global event-grid coverage expansion. Interpret it alongside `coverage_missing_rate` and `true_failure_rate`.

## Sensor-Level QC Correction

- Renamed event-grid missing records from `missing_record` to `not_available_for_global_event`.
- Sensor status no longer treats global event-grid coverage misses as sensor malfunction.
- `sensor_qc_report.csv` now separates:
  - coverage missing rate
  - true technical QC failure rate
- `sensor_status = exclude` is based on true QC failures, not missing coverage.
- `report/neural_input_summary.json` records:
  - total retained preprocess events checked
  - events with complete selected-sensor coverage
  - events excluded from neural packaging due to incomplete selected-sensor coverage
  - events excluded due to true QC failures

## Sensor Numbering

- Added deterministic neural model channel numbering in `sensor_map.csv`.
- Included strain channels are numbered first, then ACC-Z channels.
- Sensor map includes source sensor order plus model channel IDs and global model channel indices.

## Config Updates

- Added documented preprocessing variables in `configs/default.yaml` and `configs/full_pipeline.yaml`.
- Important active settings now include:
  - `preprocessing.sampling_rate_hz`
  - `preprocessing.sensor_selection.decks`
  - `preprocessing.strain.locations`
  - `preprocessing.strain.filter.method`
  - `preprocessing.strain.zeroing.method`
  - `preprocessing.strain.peak_window_half_samples`
  - `preprocessing.acc.axis`
  - `preprocessing.acc.min_aligned_samples`
  - `preprocessing.acc.filter.*`
  - `preprocessing.acc.zeroing.method`
  - `preprocessing.acc.time_padding.method`
  - `preprocessing.acc.frequency_transform.*`
  - `preprocessing.qc.*`

## Notebook Work

- Renamed and populated the trial notebook:
  - `notebooks/misc/E_preprocessing_qc_neural_inputs.ipynb`
- The notebook is an experiment harness for:
  - inspecting active config
  - optionally running preprocess
  - loading `neural_inputs.npy`
  - reviewing sensor numbering
  - reviewing QC reports and flagged plots

## Verification

The current checks passed after the implemented changes:

```text
pytest tests/test_preprocessing.py -q
pytest tests/test_imports.py -q
ruff check aquinas_toolkit tests
```

At the latest verification point:

```text
tests/test_preprocessing.py: 92 passed
tests/test_imports.py: 13 passed
ruff: all checks passed
```
