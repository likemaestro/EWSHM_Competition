# feature_extraction/

## Purpose

Convert preprocessed waveforms into compact feature vectors suitable
for unsupervised learning.

## Status

Stub — not yet implemented.

Planned work:

- Time-domain statistics (RMS, peak-to-peak, kurtosis, skewness,
  crest factor, zero-crossing rate)
- Frequency-domain features (dominant frequencies via FFT/PSD,
  spectral centroid, energy in frequency bands)
- Frequency Domain Decomposition (FDD) for modal peak extraction from
  multichannel acceleration response data
- Cross-sensor features (correlation between co-located sensors)
- Index-table features (the 15 pre-computed values already in the
  TABLE JSON: Duration, Range, Mean_Value, Temperature, etc.)

## Interface

- **Input:** preprocessed waveform DataFrames + index-table metadata
- **Output:** a feature matrix (rows = events, columns = named features)

## Implemented helpers

- `frequency_domain_decomposition(...)` -- compute singular-value spectra
  from a multichannel waveform matrix or a sequence of matrices
- `summarize_fdd_peaks(...)` -- extract dominant modal peaks from the
  first singular-value curve inside a target frequency band
- `summarize_fdd_mode_shapes(...)` -- report normalized mode-shape
  amplitudes and phases at selected FDD peak frequencies
<!-- TODO: consider writing preprocessing output and feature vectors into a
     SQLite database instead of CSV/CSV.GZ. Feature extraction, training, and
     scoring all query the same data by event, sensor, set, and deck. Indexed
     SQL lookups would be faster than scanning compressed CSVs on each stage. -->

## Damaged-Sensor Constraint

Preprocessing now supports config-driven sensor exclusions for
set-specific data integrity issues. Future feature extraction should
inherit that contract:

- excluded sensors must be absent from feature generation for the
  affected SETs
- corrupted TABLE-derived features from excluded sensors must not be
  reintroduced downstream
- the organizer-provided damaged sensor (`OLD_S1_UP_SUP_STR`) should be
  kept for SET1-SET3 and excluded for SET4-SET5 unless the policy is
  intentionally revised later with supporting evidence

This rule comes from the organizer's April 9, 2026 email. The reason is
not that the late raw files are flat; it is that the late TABLE
metadata becomes inconsistent with the raw waveform while the baseline
also shifts sharply. Feature extraction should therefore trust the
preprocess exclusion contract and avoid silently rebuilding features for
that sensor from either raw or TABLE sources in SET4/SET5.
