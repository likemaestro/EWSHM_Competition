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
- Cross-sensor features (correlation between co-located sensors)
- Index-table features (the 15 pre-computed values already in the
  TABLE JSON: Duration, Range, Mean_Value, Temperature, etc.)

## Interface

- **Input:** preprocessed waveform DataFrames + index-table metadata
- **Output:** a feature matrix (rows = events, columns = named features)
