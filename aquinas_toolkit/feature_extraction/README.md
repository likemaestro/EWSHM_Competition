# feature_extraction/

## Purpose

Convert preprocessed waveforms into compact feature vectors suitable
for unsupervised learning.

## Status

Empty -- not yet implemented.

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
