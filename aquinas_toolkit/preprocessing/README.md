# preprocessing/

## Purpose

Prepare raw 100 Hz waveforms for feature extraction. This stage sits
between the reader (raw data in) and the feature extractor (compact
vectors out).

## Status

Empty -- not yet implemented.

Planned work:

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
