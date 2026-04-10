# training/

## Purpose

Detect anomalies and long-term trends in the feature data using
unsupervised methods. The dataset has no labels -- all approaches
must be unsupervised (challenge rules).

## Status

Stub — not yet implemented.

Planned work:

- Dimensionality reduction (PCA, kernel PCA, autoencoders)
- Statistical process control (Hotelling T^2, EWMA control charts)
- Outlier detection (Isolation Forest, Local Outlier Factor)
- Clustering-based approaches

## Interface

- **Input:** feature matrix from the features stage
- **Output:** per-event anomaly scores and per-sensor trend indicators

## Organizer Notes Relevant To Future OMA Work

- The organizer stated that Operational Modal Analysis can still work on
  short traffic records when they are baseline-corrected and concatenated
  into a longer pseudo-continuous signal.
- This is forward-looking guidance only. It does not force OMA as the
  competition method, and it does not change the v1 preprocessing scope.
- The preprocess stage now preserves event-clean aligned outputs so later
  experiments can concatenate them if OMA becomes a selected path.
- The organizer also noted that bridge frequencies for this type of
  structure are typically around 2-10 Hz, which supports the current
  simple synchronization strategy for v1 preprocessing.
- Methodology reference supplied by the organizer:
  `10.1007/978-3-031-96106-9_22` (EVACES 2025 Volume 2 chapter).
