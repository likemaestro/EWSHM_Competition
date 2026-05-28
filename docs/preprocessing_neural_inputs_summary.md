# Preprocessing and Neural Input Work Summary

## Purpose

We revised the AQUINAS Toolkit preprocessing stage so its primary output is a neural-network-ready dataset for the EWSHM competition. The work keeps reusable logic in `aquinas_toolkit/` and leaves notebooks as thin experiment/inspection layers.

## Main Pipeline Changes

- Added neural input packaging under the preprocess stage.
- The preprocess stage now writes split event-level tensors under `nn_inputs/`: `strain_inputs.npy`, `acc_inputs.npy`, `temperature_inputs.npy`, and `event_ids.npy`.
- Verbose metadata artifacts are written under `nn_inputs/metadata/` and `report/`, including sensor map, input shapes, frequency bins, valid lengths, and temperature metadata.
- Flexible preprocessing is preserved: technically valid partial-sensor events remain in the preprocess store, while the split NN arrays only use events with complete selected-sensor coverage.
- Limited the current default dataset scope to `AQUINAS_SET1_2022_07` and the `OLD` deck.
- Excluded ACC-Y from neural preprocessing and retained only ACC-Z.

## Signal-Specific Preprocessing

- Replaced misleading global filter/zeroing intent with signal-specific preprocessing settings.
- Strain-type sensors (`INF_STR`, `SUP_STR`):
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

## Neural-Input Event Selection

- Neural input packaging first checks whether each retained preprocess event contains all selected neural sensors.
- Events missing one or more required selected sensors are excluded as incomplete coverage.
- Events with full selected-sensor coverage are then checked for packaging constraints:
  - finite strain and ACC_Z values
  - strain window long enough for the fixed 200-sample clip
  - ACC_Z aligned length at or above `min_aligned_samples`
- In the current fresh run, incomplete selected-sensor coverage excluded `5329` events and packaging constraints excluded `0` events.

## Sensor Override Summary

- Configured sensor exclusions such as the documented damaged sensor in SET4 and SET5 are recorded through `sensor_records.csv`, `event_manifest.csv`, and `summary.json`.
- `report/neural_input_summary.json` records:
  - total retained preprocess events checked
  - events with complete selected-sensor coverage
  - events excluded from neural packaging due to incomplete selected-sensor coverage
  - events excluded by packaging constraints

## Sensor Numbering

- Added deterministic neural model channel numbering in `sensor_map.csv`.
- Included strain channels are numbered first, then ACC-Z channels.
- Sensor map includes source sensor order plus model channel IDs and global model channel indices.

## Config Updates

- Added documented preprocessing variables in `configs/default.yaml` and deck-specific all-set configs.
- Important active settings now include:
  - `preprocessing.sampling_rate_hz`
  - `preprocessing.sensor_selection.decks`
  - `preprocessing.strain.locations`
  - `preprocessing.strain.filter.method`
  - `preprocessing.strain.zeroing.method`
  - `preprocessing.strain.peak_window_half_samples`
  - `preprocessing.acc.min_aligned_samples`
  - `preprocessing.acc.filter.*`
  - `preprocessing.acc.zeroing.method`
  - `preprocessing.acc.frequency_transform.*`
  - `preprocessing.storage.backend`
  - `preprocessing.exports.aligned_waveforms.*`

## Notebook Work

- Renamed and populated the trial notebook:
  - `notebooks/misc/E_preprocessing_neural_inputs.ipynb`
- The notebook is an experiment harness for:
  - inspecting active config
  - optionally running preprocess
  - loading the split NN input arrays
  - reviewing sensor numbering
  - reviewing retained event coverage and tensor layout metadata

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
