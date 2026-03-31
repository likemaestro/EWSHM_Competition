# training/

## Purpose

Detect anomalies and long-term trends in the feature data using
unsupervised methods. The dataset has no labels -- all approaches
must be unsupervised (challenge rules).

## Status

Empty -- not yet implemented.

Planned work:

- Dimensionality reduction (PCA, kernel PCA, autoencoders)
- Statistical process control (Hotelling T^2, EWMA control charts)
- Outlier detection (Isolation Forest, Local Outlier Factor)
- Clustering-based approaches

## Interface

- **Input:** feature matrix from the features stage
- **Output:** per-event anomaly scores and per-sensor trend indicators
