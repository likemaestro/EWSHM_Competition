# preprocessing/

## Purpose

Prepare raw 100 Hz waveforms for feature extraction. This stage sits
between the reader (raw data in) and the feature extractor (compact
vectors out).

## Status

Empty -- not yet implemented.

Planned work:

- Minimum-duration filtering for acceleration records when short windows
  cannot resolve the target structural frequency range
- Baseline removal (subtract mean or fitted trend)
- Band-pass filtering (remove drift and high-frequency noise)
- Temperature normalisation (compensate strain drift with ambient temp)
- Cross-sensor time alignment (match events via `Start_Time` / `End_Time`,
  since `Record_UID` is sensor-specific)
- Resampling / zero-padding to a consistent record length

## Interface

- **Input:** raw waveform DataFrames from `AquinasReader.read_record()`
  and index-table metadata (especially `Temperature`)
- **Output:** cleaned, aligned DataFrames ready for feature extraction

## Implemented helpers

- `filter_records_by_min_duration(...)` -- filter index-table rows by a
  minimum `Duration` threshold, optionally restricted to a sensor subset
  such as `ACC_Z`
- `summarize_min_duration_filter(...)` -- report how many records are
  kept and excluded per sensor after the duration filter
- `find_common_sensor_events(...)` -- align surviving records across
  sensors using `Start_Time` / `End_Time`
- `load_common_event_waveform_matrix(...)` -- load one aligned event as a
  multichannel waveform matrix
- `bandpass_filter_waveform_matrix(...)` -- apply a zero-phase Butterworth
  band-pass filter such as `0.5-20 Hz`
