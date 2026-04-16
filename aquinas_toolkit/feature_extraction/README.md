# feature_extraction/

## Purpose

Convert preprocessed waveforms into compact feature vectors suitable
for unsupervised learning.

## Status

Partially implemented for v1.

Current implemented scope:

- Per-sensor per-event statistical features written to `features.sqlite`,
  including waveform mean, std, RMS, min, max, peak-to-peak, energy,
  crest factor, zero-crossing rate, skewness, and kurtosis
- Index-table-derived context fields stored alongside those sensor-event
  rows, including duration, start/end/diff/min/max/mean/range values,
  and temperature
- Frequency Domain Decomposition (FDD) for modal peak extraction from
  multichannel acceleration response data
- Peak picking on the first singular-value curve inside a target band
- Mode-shape summarization and structural-location annotation
- Thin notebook-facing wrappers that preserve Mohsen's
  filtered-event-to-FDD workflow without leaving reusable logic in
  notebook cells

Deferred beyond this pass:

- Frequency-domain features beyond FDD (dominant frequencies via FFT/PSD,
  spectral centroid, energy in frequency bands)
- Cross-sensor features (correlation between co-located sensors)

## Interface

- **Input:** filtered, zeroed, and aligned waveform DataFrames from the
  preprocessing stage (band-pass filtered, baseline-corrected,
  timestamp-synchronized)
- **Output:** a SQLite-backed feature store rooted at `features.sqlite`.
  The canonical tables are:
  `sensor_event_features` (per-sensor per-event statistical features and
  TABLE-derived context),
  `deck_modal_peaks` (deck-level FDD peak summaries),
  `deck_mode_shape_components` (per-sensor mode-shape components), and
  `feature_family_status` (completed/skipped status per feature family).

## Stage Progress

`run_features()` now reports stage-owned progress phases with the shared CLI console:

- loading preprocess artifacts
- extracting per-sensor features across retained events
- per `(set, deck)` modal analysis progress while checking retained events for complete configured acceleration-axis coverage
- configured-axis acceleration FDD execution (or a clear `skipped` reason when requirements are not met)
- writing `features.sqlite`

This inner stage progress is displayed both when running `aquinas run features`
and when `features` is executed inside `aquinas run` with the outer pipeline bar.

## Implemented helpers

- `frequency_domain_decomposition(...)` -- compute singular-value spectra
  from a multichannel waveform matrix or a sequence of matrices
- `summarize_fdd_peaks(...)` -- extract dominant modal peaks from the
  first singular-value curve inside a target frequency band
- `summarize_fdd_mode_shapes(...)` -- report normalized mode-shape
  amplitudes and phases at selected FDD peak frequencies
- `annotate_mode_shape_locations(...)` -- parse AQUINAS channel names
  into deck/span/side/location fields for plotting and reporting
- `collect_filtered_event_matrices(...)` -- gather per-deck filtered
  ACC_Z multichannel events using preprocessing helpers
- `collect_preprocessed_event_matrices(...)` -- gather preprocess-backed
  ACC_Z aligned events, requiring the full configured ACC_Z channel set
  for the selected set/deck by default
- `run_acc_z_fdd_workflow(...)` -- run Mohsen's preserved filtered
  ACC_Z FDD workflow for one set/deck
- `summarize_fdd_results(...)` -- convert FDD outputs into notebook-ready
  peak, amplitude, signed-component, and phase tables

## Current ACC_Z FDD Selection Rule

- The preprocess-backed FDD path works per `(set, deck)` pair.
- By default it now accepts only events where every configured included
  `ACC_Z` sensor for that set/deck is present simultaneously.
- This replaces the earlier permissive fallback that could run with only
  two common channels and gives mode-shape estimates full spatial coverage.
- The fallback can still be re-enabled by setting
  `features.modal_analysis.require_full_channel_set: false` in the run config.

The intended ownership split is:

- `aquinas_toolkit.preprocessing` handles all signal conditioning:
  band-pass filtering, baseline zeroing, timestamp alignment, duration
  filtering, common-event loading, and the batch preprocess pipeline
- `aquinas_toolkit.feature_extraction` derives modal and statistical
  features from the conditioned waveforms produced by preprocessing
- `training` should consume the canonical feature store and may emit a
  separate dense model-ready matrix artifact later if a specific
  unsupervised method benefits from it

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

## Attribution

FDD implementation and ACC_Z feature-extraction workflow originally by `Mohsen Rezvani Alile`.
Adapted into reusable helpers from the feature-extraction notebook.
