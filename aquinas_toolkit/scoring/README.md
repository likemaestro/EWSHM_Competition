# scoring/

## Purpose

Aggregate per-sensor anomaly scores into a single synthetic health
score for the viaduct. This is the final deliverable required by
the challenge rules.

## Status

Implemented as a notebook-backed scoring narrative. The current final
interpretation is documented in `notebooks/05_health_scoring.ipynb` and the
cross-SET evaluation scripts under `notebooks/azrmirz_fncs/`.

Current work:

- Per-sensor score aggregation (summarise each sensor's anomaly
  history into a time series)
- Cross-sensor combination (weight and merge all 48 sensors into
  a global health indicator)
- Trend quantification (stable / improving / degrading over the
  five monthly datasets)
- Confidence bounds alongside point estimates

## Interface

- **Input:** per-event anomaly scores and trend indicators from the
  neural reconstruction-error experiments
- **Output:** a time series of structural health scores with
  uncertainty estimates, covering all five monthly datasets
