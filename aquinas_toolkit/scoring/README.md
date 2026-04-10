# scoring/

## Purpose

Aggregate per-sensor anomaly scores into a single synthetic health
score for the viaduct. This is the final deliverable required by
the challenge rules.

## Status

Stub — not yet implemented.

Planned work:

- Per-sensor score aggregation (summarise each sensor's anomaly
  history into a time series)
- Cross-sensor combination (weight and merge all 48 sensors into
  a global health indicator)
- Trend quantification (stable / improving / degrading over the
  five monthly datasets)
- Confidence bounds alongside point estimates

## Interface

- **Input:** per-event anomaly scores and trend indicators from the
  training stage
- **Output:** a time series of structural health scores with
  uncertainty estimates, covering all five monthly datasets
